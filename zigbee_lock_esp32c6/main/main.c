/*
 * ==============================================================================
 * MyHome Zigbee 门锁光耦控制器 (ESP32-C6 Super Mini + PC817)
 * 基于 ESP-Zigbee-SDK 1.6.x 官方标准实现
 * ==============================================================================
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "esp_pm.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "nvs_flash.h"
#include "ha/esp_zigbee_ha_standard.h"

#define UNLOCK_GPIO_PIN       GPIO_NUM_1   /* 串联 1kΩ 限流电阻接 PC817 第 1 脚 */
#define PAIR_BUTTON_GPIO      GPIO_NUM_9   /* 板载 BOOT 键，短接 GND 触发 */
#define MYHOME_ENDPOINT       1

#define UNLOCK_PULSE_MS       300          /* PC817 光耦闭合导通时长 (ms) */
#define UNLOCK_COOLDOWN_MS    2000         /* 防重防抖冷却保护期 (ms) */

static const char *TAG = "MYHOME_ZB_LOCK";

static TimerHandle_t s_pulse_timer = NULL;
static TimerHandle_t s_cooldown_timer = NULL;
static bool s_in_cooldown = false;
static int s_button_press_ticks = 0;

/* PC817 点动脉冲复位：300ms 后拉低引脚，模拟松开按键 */
static void pulse_timer_callback(TimerHandle_t xTimer)
{
    gpio_set_level(UNLOCK_GPIO_PIN, 0);

    /* 门锁状态自动回到上锁闭合态 */
    uint8_t lock_state = 0x01; /* Locked */
    esp_zb_zcl_set_attribute_val(MYHOME_ENDPOINT,
                                 ESP_ZB_ZCL_CLUSTER_ID_DOOR_LOCK,
                                 ESP_ZB_ZCL_CLUSTER_SERVER_ROLE,
                                 ESP_ZB_ZCL_ATTR_DOOR_LOCK_LOCK_STATE_ID,
                                 &lock_state, false);

    s_in_cooldown = true;
    xTimerStart(s_cooldown_timer, 0);
    ESP_LOGI(TAG, "Door lock pulse finished, restored to locked state");
}

static void cooldown_timer_callback(TimerHandle_t xTimer)
{
    s_in_cooldown = false;
}

static void trigger_unlock_pulse(void)
{
    if (s_in_cooldown) {
        return;
    }
    ESP_LOGI(TAG, "Triggering unlock pulse (300ms)...");
    gpio_set_level(UNLOCK_GPIO_PIN, 1);
    xTimerStart(s_pulse_timer, 0);
}

/* 接收 Zigbee ZCL 指令回调 */
static esp_err_t zb_attribute_handler(const esp_zb_zcl_set_attr_value_message_t *message)
{
    if (!message || message->info.status != ESP_ZB_ZCL_STATUS_SUCCESS) {
        return ESP_OK;
    }
    if (message->info.dst_endpoint == MYHOME_ENDPOINT) {
        if (message->info.cluster == ESP_ZB_ZCL_CLUSTER_ID_DOOR_LOCK) {
            trigger_unlock_pulse();
        } else if (message->info.cluster == ESP_ZB_ZCL_CLUSTER_ID_ON_OFF) {
            if (*(uint8_t *)message->attribute.data.value == 1) {
                trigger_unlock_pulse();
            }
        }
    }
    return ESP_OK;
}

static esp_err_t zb_action_handler(esp_zb_core_action_callback_id_t callback_id, const void *message)
{
    if (callback_id == ESP_ZB_CORE_SET_ATTR_VALUE_CB_ID) {
        return zb_attribute_handler((const esp_zb_zcl_set_attr_value_message_t *)message);
    }
    return ESP_OK;
}

static void bdb_start_top_level_commissioning_cb(uint8_t mode_mask)
{
    esp_zb_bdb_start_top_level_commissioning(mode_mask);
}

void esp_zb_app_signal_handler(esp_zb_app_signal_t *signal_struct)
{
    uint32_t *p_sg_p = signal_struct->p_app_signal;
    esp_err_t err_status = signal_struct->esp_err_status;
    esp_zb_app_signal_type_t sig_type = (esp_zb_app_signal_type_t)*p_sg_p;

    switch (sig_type) {
    case ESP_ZB_ZDO_SIGNAL_SKIP_STARTUP:
        esp_zb_bdb_start_top_level_commissioning(ESP_ZB_BDB_MODE_INITIALIZATION);
        break;
    case ESP_ZB_BDB_SIGNAL_DEVICE_FIRST_START:
    case ESP_ZB_BDB_SIGNAL_DEVICE_REBOOT:
        if (err_status == ESP_OK) {
            if (esp_zb_bdb_is_factory_new()) {
                esp_zb_bdb_start_top_level_commissioning(ESP_ZB_BDB_MODE_NETWORK_STEERING);
            }
        } else {
            esp_zb_scheduler_alarm((esp_zb_callback_t)bdb_start_top_level_commissioning_cb,
                                   ESP_ZB_BDB_MODE_INITIALIZATION, 1000);
        }
        break;
    case ESP_ZB_BDB_SIGNAL_STEERING:
        if (err_status != ESP_OK) {
            esp_zb_scheduler_alarm((esp_zb_callback_t)bdb_start_top_level_commissioning_cb,
                                   ESP_ZB_BDB_MODE_NETWORK_STEERING, 1000);
        }
        break;
    default:
        break;
    }
}

static void button_task(void *pvParameters)
{
    while (1) {
        if (gpio_get_level(PAIR_BUTTON_GPIO) == 0) {
            s_button_press_ticks++;
            if (s_button_press_ticks == 30) {
                esp_zb_factory_reset();
            }
        } else {
            if (s_button_press_ticks > 0 && s_button_press_ticks < 30) {
                trigger_unlock_pulse();
            }
            s_button_press_ticks = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static void esp_zb_task(void *pvParameters)
{
    esp_zb_cfg_t zb_nwk_cfg = {
        .esp_zb_role = ESP_ZB_DEVICE_TYPE_ED,
        .install_code_policy = false,
        .nwk_cfg.zed_cfg = {
            .ed_timeout = ESP_ZB_ED_0_MINUTES,
            .keep_alive = 3000,
        },
    };
    esp_zb_init(&zb_nwk_cfg);

    /* 发射功率优化：限制在 +10dBm */
    esp_zb_set_tx_power(10);

    /* 创建标准 Door Lock 端点 */
    esp_zb_door_lock_cfg_t lock_cfg = ESP_ZB_DEFAULT_DOOR_LOCK_CONFIG();
    esp_zb_ep_list_t *door_lock_ep = esp_zb_door_lock_ep_create(MYHOME_ENDPOINT, &lock_cfg);
    esp_zb_device_register(door_lock_ep);

    esp_zb_core_action_handler_register(zb_action_handler);
    esp_zb_set_primary_network_channel_set(ESP_ZB_TRANSCEIVER_ALL_CHANNELS_MASK);
    esp_zb_start(false);
    esp_zb_stack_main_loop();
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

#ifdef CONFIG_PM_ENABLE
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 80,
        .min_freq_mhz = 40,
        .light_sleep_enable = true
    };
    esp_pm_configure(&pm_config);
#endif

    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << UNLOCK_GPIO_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(UNLOCK_GPIO_PIN, 0);

    gpio_config_t btn_conf = {
        .pin_bit_mask = (1ULL << PAIR_BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn_conf);

    s_pulse_timer = xTimerCreate("pulse_tmr", pdMS_TO_TICKS(UNLOCK_PULSE_MS), pdFALSE, NULL, pulse_timer_callback);
    s_cooldown_timer = xTimerCreate("cd_tmr", pdMS_TO_TICKS(UNLOCK_COOLDOWN_MS), pdFALSE, NULL, cooldown_timer_callback);

    xTaskCreate(button_task, "btn_task", 2048, NULL, 5, NULL);
    xTaskCreate(esp_zb_task, "zb_task", 4096, NULL, 5, NULL);
}
