/*
 * ==============================================================================
 * MyHome Zigbee 门锁光耦控制器 (ESP32-C6 Super Mini + PC817)
 * 极致低功耗设计：
 *   1. 彻底断电 Wi-Fi / 蓝牙，CPU 降频至 40MHz
 *   2. 802.15.4 发射功率限制在 +10dBm (峰值电流从 150mA 砍至 50mA)
 *   3. 常态 1.5 秒休眠轮询 (1500ms)，预警期 200ms 高速轮询，平时 Light Sleep (~35μA)
 *   4. 收到开锁指令后，GPIO 1 输出高电平驱动 PC817 导通 300ms 后自动复位断开
 *   5. 板载 BOOT 按键 (GPIO 9) 长按 3 秒重置 Zigbee 配网
 * ==============================================================================
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "esp_pm.h"
#include "driver/gpio.h"
#include "nvs_flash.h"
#include "esp_zigbee_core.h"

/* 硬件引脚分配 */
#define UNLOCK_GPIO_PIN       GPIO_NUM_1   /* 串联 1kΩ 电阻接 PC817 第 1 脚 */
#define PAIR_BUTTON_GPIO      GPIO_NUM_9   /* Super Mini 板载 BOOT 键，短接 GND 触发 */

/* 动作与时序参数 */
#define UNLOCK_PULSE_MS       300          /* PC817 光耦导通闭合时长 (ms) */
#define UNLOCK_COOLDOWN_MS    2000         /* 防重触发保护冷却时间 (ms) */
#define NORMAL_POLL_MS        1500         /* 常态休眠轮询间隔 (1.5 秒) */
#define FAST_POLL_MS          200          /* 预警加速轮询间隔 (200 毫秒) */

#define MYHOME_ENDPOINT       1

static TimerHandle_t s_pulse_timer = NULL;
static TimerHandle_t s_cooldown_timer = NULL;
static TimerHandle_t s_fast_poll_timer = NULL;
static bool s_in_cooldown = false;
static int s_button_press_ticks = 0;

/* ============================================================================
 * PC817 光耦点动脉冲控制逻辑
 * ============================================================================ */

/* 脉冲到期回调：300ms 后自动拉低，光耦断开，模拟松开室内开门按钮 */
static void pulse_timer_callback(TimerHandle_t xTimer)
{
    gpio_set_level(UNLOCK_GPIO_PIN, 0);

    /* 更新 Zigbee 门锁属性状态为上锁闭合态，保持状态干净 */
    uint8_t lock_state = 0x01; /* Locked */
    uint8_t on_off = 0;        /* Off */
    esp_zb_zcl_set_attribute_val(MYHOME_ENDPOINT, ESP_ZB_ZCL_CLUSTER_ID_DOOR_LOCK,
                                 ESP_ZB_ZCL_CLUSTER_SERVER_ROLE,
                                 ESP_ZB_ZCL_ATTR_DOOR_LOCK_LOCK_STATE_ID,
                                 &lock_state, false);
    esp_zb_zcl_set_attribute_val(MYHOME_ENDPOINT, ESP_ZB_ZCL_CLUSTER_ID_ON_OFF,
                                 ESP_ZB_ZCL_CLUSTER_SERVER_ROLE,
                                 ESP_ZB_ZCL_ATTR_ON_OFF_ON_OFF_ID,
                                 &on_off, false);

    /* 开启 2 秒防重冷却保护 */
    s_in_cooldown = true;
    xTimerStart(s_cooldown_timer, 0);
}

static void cooldown_timer_callback(TimerHandle_t xTimer)
{
    s_in_cooldown = false;
}

/* 预警高速轮询恢复回调：15 秒后恢复为 1.5 秒常态低功耗休眠 */
static void fast_poll_timeout_callback(TimerHandle_t xTimer)
{
    esp_zb_zdo_pim_set_long_poll_interval(NORMAL_POLL_MS);
}

/* 执行开门物理动作 */
static void trigger_unlock_pulse(void)
{
    if (s_in_cooldown) {
        return;
    }

    /* 1. 导通 PC817 (高电平) */
    gpio_set_level(UNLOCK_GPIO_PIN, 1);

    /* 2. 启动 300ms 单次定时器，到期自动断开 */
    xTimerStart(s_pulse_timer, 0);
}

/* 切换为 200ms 极速响应模式 (维持 15 秒) */
static void enter_fast_poll_mode(void)
{
    esp_zb_zdo_pim_set_long_poll_interval(FAST_POLL_MS);
    xTimerStart(s_fast_poll_timer, 0);
}

/* ============================================================================
 * Zigbee ZCL 指令回调拦截处理
 * ============================================================================ */

static esp_err_t zb_action_handler(esp_zb_core_action_callback_id_t callback_id, const void *message)
{
    esp_err_t ret = ESP_OK;

    switch (callback_id) {
    case ESP_ZB_CORE_SET_ATTR_VALUE_CB_ID: {
        const esp_zb_zcl_set_attr_value_message_t *msg = (const esp_zb_zcl_set_attr_value_message_t *)message;
        if (msg->info.cluster == ESP_ZB_ZCL_CLUSTER_ID_ON_OFF) {
            if (msg->attribute.id == ESP_ZB_ZCL_ATTR_ON_OFF_ON_OFF_ID && *(uint8_t *)msg->attribute.data.value == 1) {
                trigger_unlock_pulse();
            }
        }
        break;
    }
    case ESP_ZB_CORE_CMD_CUSTOM_CLUSTER_REQ_CB_ID:
    case ESP_ZB_CORE_CMD_DEFAULT_RESP_CB_ID:
        break;
    default:
        break;
    }
    return ret;
}

/* 接收 Door Lock 专用 ZCL 命令 (Unlock Door: 0x01) */
static uint8_t zb_door_lock_cluster_handler(const esp_zb_zcl_cmd_t *cmd)
{
    if (cmd->cluster_id == ESP_ZB_ZCL_CLUSTER_ID_DOOR_LOCK) {
        if (cmd->command.id == 0x01) { /* Unlock Door */
            trigger_unlock_pulse();
            return true;
        } else if (cmd->command.id == 0x00) { /* Lock Door (进入加速响应模式) */
            enter_fast_poll_mode();
            return true;
        }
    }
    return false;
}

/* Zigbee 信号事件处理 */
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
            /* 启动成功，配置常态 1.5 秒轮询休眠 */
            esp_zb_zdo_pim_set_long_poll_interval(NORMAL_POLL_MS);
            esp_zb_zdo_pim_set_fast_poll_interval(FAST_POLL_MS);
        } else {
            /* 未配网，启动搜网 */
            esp_zb_bdb_start_top_level_commissioning(ESP_ZB_BDB_MODE_NETWORK_STEERING);
        }
        break;

    case ESP_ZB_BDB_SIGNAL_STEERING:
        if (err_status == ESP_OK) {
            /* 成功入网！锁定 1.5 秒低功耗轮询 */
            esp_zb_zdo_pim_set_long_poll_interval(NORMAL_POLL_MS);
        } else {
            /* 搜网失败，10 秒后重试 */
            vTaskDelay(pdMS_TO_TICKS(10000));
            esp_zb_bdb_start_top_level_commissioning(ESP_ZB_BDB_MODE_NETWORK_STEERING);
        }
        break;

    default:
        break;
    }
}

/* ============================================================================
 * 配网按键轮询与本地点动测试 (GPIO 9 BOOT 键)
 * ============================================================================ */

static void button_task(void *pvParameters)
{
    while (1) {
        if (gpio_get_level(PAIR_BUTTON_GPIO) == 0) {
            s_button_press_ticks++;
            if (s_button_press_ticks == 30) { /* 持续按住 3 秒：重置 Zigbee 配网 */
                esp_zb_factory_reset();
            }
        } else {
            if (s_button_press_ticks > 0 && s_button_press_ticks < 30) {
                /* 短按松开：本地直接触发一次开门脉冲测试 */
                trigger_unlock_pulse();
            }
            s_button_press_ticks = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

/* ============================================================================
 * Zigbee 协议栈主任务与外设初始化
 * ============================================================================ */

static void esp_zb_task(void *pvParameters)
{
    /* 1. 初始化 Zigbee 平台配置 (作为休眠终端 ZED) */
    esp_zb_cfg_t zb_nwk_cfg = {
        .esp_zb_role = ESP_ZB_DEVICE_TYPE_ED,
        .install_code_policy = false,
        .nwk_cfg.zed_cfg = {
            .ed_timeout = ESP_ZB_ED_0_MINUTES,
            .keep_alive = 3000,
        },
    };
    esp_zb_init(&zb_nwk_cfg);

    /* 2. 射频发射功率优化：调低至 +10dBm (峰值发包电流从 150mA 砍至 50mA) */
    esp_zb_set_tx_power(10);

    /* 3. 创建 Cluster 列表并挂载到 Endpoint 1 */
    esp_zb_ep_list_t *ep_list = esp_zb_ep_list_create();

    /* Basic Cluster */
    esp_zb_basic_cluster_cfg_t basic_cfg = {
        .power_source = 0x03, /* Battery */
    };
    esp_zb_cluster_list_t *cluster_list = esp_zb_zcl_cluster_list_create();
    esp_zb_attribute_list_t *basic_cluster = esp_zb_basic_cluster_create(&basic_cfg);
    esp_zb_cluster_list_add_basic_cluster(cluster_list, basic_cluster, ESP_ZB_ZCL_CLUSTER_SERVER_ROLE);

    /* Identify Cluster */
    esp_zb_identify_cluster_cfg_t identify_cfg = { .identify_time = 0 };
    esp_zb_cluster_list_add_identify_cluster(cluster_list, esp_zb_identify_cluster_create(&identify_cfg), ESP_ZB_ZCL_CLUSTER_SERVER_ROLE);

    /* Door Lock Cluster (0x0101) */
    esp_zb_door_lock_cluster_cfg_t lock_cfg = {
        .lock_state = 0x01, /* Locked */
        .lock_type = 0x00,  /* Deadbolt */
        .actuator_enabled = 1,
    };
    esp_zb_attribute_list_t *lock_cluster = esp_zb_door_lock_cluster_create(&lock_cfg);
    esp_zb_cluster_list_add_door_lock_cluster(cluster_list, lock_cluster, ESP_ZB_ZCL_CLUSTER_SERVER_ROLE);

    /* On/Off Cluster (0x0006) 兼容开关模式控制 */
    esp_zb_on_off_cluster_cfg_t on_off_cfg = { .on_off = false };
    esp_zb_attribute_list_t *on_off_cluster = esp_zb_on_off_cluster_create(&on_off_cfg);
    esp_zb_cluster_list_add_on_off_cluster(cluster_list, on_off_cluster, ESP_ZB_ZCL_CLUSTER_SERVER_ROLE);

    /* 绑定 Endpoint */
    esp_zb_endpoint_config_t endpoint_config = {
        .endpoint = MYHOME_ENDPOINT,
        .app_profile_id = ESP_ZB_AF_HA_PROFILE_ID,
        .app_device_id = ESP_ZB_HA_DOOR_LOCK_DEVICE_ID,
        .app_device_version = 0,
    };
    esp_zb_ep_list_add_ep(ep_list, cluster_list, endpoint_config);
    esp_zb_device_register(ep_list);

    /* 注册全局回调 */
    esp_zb_core_action_handler_register(zb_action_handler);

    /* 启动协议栈 */
    esp_zb_set_primary_network_channel_set(ESP_ZB_TRANSCEIVER_ALL_CHANNELS_MASK);
    esp_zb_start(false);
    esp_zb_main_loop_iteration();
}

void app_main(void)
{
    /* 1. 初始化 NVS 存储 */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    /* 2. 配置电源管理与自动 Light Sleep (CPU 降频至 40MHz，空闲自动睡眠) */
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 40,
        .min_freq_mhz = 40,
        .light_sleep_enable = true
    };
    esp_pm_configure(&pm_config);

    /* 3. 配置开锁输出 GPIO (PC817 驱动脚) */
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << UNLOCK_GPIO_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(UNLOCK_GPIO_PIN, 0);

    /* 4. 配置配网按钮 GPIO (GPIO 9 内部上拉) */
    gpio_config_t btn_conf = {
        .pin_bit_mask = (1ULL << PAIR_BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn_conf);

    /* 5. 创建定时器 */
    s_pulse_timer = xTimerCreate("pulse_tmr", pdMS_TO_TICKS(UNLOCK_PULSE_MS), pdFALSE, NULL, pulse_timer_callback);
    s_cooldown_timer = xTimerCreate("cd_tmr", pdMS_TO_TICKS(UNLOCK_COOLDOWN_MS), pdFALSE, NULL, cooldown_timer_callback);
    s_fast_poll_timer = xTimerCreate("fast_tmr", pdMS_TO_TICKS(15000), pdFALSE, NULL, fast_poll_timeout_callback);

    /* 6. 创建后台任务 */
    xTaskCreate(button_task, "btn_task", 2048, NULL, 5, NULL);
    xTaskCreate(esp_zb_task, "zb_task", 4096, NULL, 5, NULL);
}

