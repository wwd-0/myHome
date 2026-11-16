/*
 * ==============================================================================
 * MyHome Zigbee 门锁光耦控制器主程序 (Pro Micro nRF52840 + PC817)
 * 核心特性：
 *   1. Zigbee 3.0 Sleepy End Device (ZED) 深度睡眠
 *   2. 1.5 秒定时轮询唤醒 (1500ms Data Request Poll)，2 节 5 号电池续航 >= 2 年
 *   3. 收到开锁指令后，PC817 导通 300ms 模拟按下室内开门按键，自动复位断开
 *   4. 支持短接 P0.20 到 GND 3 秒清除网络凭证重新配网
 * ==============================================================================
 */

#include <zephyr.h>
#include <device.h>
#include <drivers/gpio.h>
#include <settings/settings.h>

#include <zboss_api.h>
#include <zboss_api_addons.h>
#include <zigbee/zigbee_app_utils.h>
#include <zigbee/zigbee_error_handler.h>
#include <zb_nrf_platform.h>

#include "zb_door_lock.h"

#define OPT_NODE DT_ALIAS(optocoupler0)
#define LED_NODE DT_ALIAS(statusled0)
#define BTN_NODE DT_ALIAS(pairbtn0)

static const struct gpio_dt_spec optocoupler = GPIO_DT_SPEC_GET(OPT_NODE, gpios);
static const struct gpio_dt_spec status_led  = GPIO_DT_SPEC_GET_OR(LED_NODE, gpios, {0});
static const struct gpio_dt_spec pair_button = GPIO_DT_SPEC_GET_OR(BTN_NODE, gpios, {0});

/* 运行期上下文 */
static struct myhome_lock_ctx g_lock_ctx;
static struct k_work_delayable g_pulse_off_work;
static struct k_work_delayable g_cooldown_work;
static struct k_work_delayable g_pair_btn_work;
static bool g_in_cooldown = false;
static int64_t g_btn_press_time = 0;

/* ZCL 属性声明 (Basic Cluster) */
static zb_uint8_t  attr_zcl_version     = ZB_ZCL_VERSION;
static zb_uint8_t  attr_power_source     = ZB_ZCL_BASIC_POWER_SOURCE_BATTERY;
static zb_char_t   attr_mf_name[]        = {6, 'M', 'y', 'H', 'o', 'm', 'e'};
static zb_char_t   attr_model_id[]       = {12, 'Z', 'B', '-', 'D', 'o', 'o', 'r', 'L', 'o', 'c', 'k'};
static zb_char_t   attr_date_code[]      = {8, '2', '0', '2', '6', '1', '1', '1', '3'};
static zb_uint8_t  attr_app_version      = 1;

ZB_ZCL_DECLARE_BASIC_ATTRIB_LIST_EXT(
	basic_attr_list,
	&attr_zcl_version,
	&attr_app_version,
	&attr_power_source,
	attr_mf_name,
	attr_model_id,
	attr_date_code
);

/* ZCL 属性声明 (Identify Cluster) */
ZB_ZCL_DECLARE_IDENTIFY_ATTRIB_LIST(
	identify_attr_list,
	&g_lock_ctx.identify_time
);

/* ZCL 属性声明 (On/Off Cluster: 收到 ON 触发开锁) */
ZB_ZCL_DECLARE_ON_OFF_ATTRIB_LIST(
	on_off_attr_list,
	&g_lock_ctx.on_off_state
);

/* ZCL 属性声明 (Door Lock Cluster: 收到 Unlock 触发开锁) */
ZB_ZCL_START_DECLARE_ATTRIB_LIST(door_lock_attr_list)
	ZB_ZCL_SET_ATTR_DESC(ZB_ZCL_ATTR_DOOR_LOCK_LOCK_STATE_ID, (&g_lock_ctx.lock_state))
	ZB_ZCL_SET_ATTR_DESC(ZB_ZCL_ATTR_DOOR_LOCK_LOCK_TYPE_ID,  (&g_lock_ctx.lock_type))
ZB_ZCL_FINISH_DECLARE_ATTRIB_LIST;

/* Cluster 列表声明 */
ZB_HA_DECLARE_DOOR_LOCK_CLUSTER_LIST(
	door_lock_clusters,
	basic_attr_list,
	identify_attr_list,
	door_lock_attr_list
);

/* Endpoint 声明 */
ZB_HA_DECLARE_DOOR_LOCK_EP(
	door_lock_ep,
	MYHOME_LOCK_ENDPOINT,
	door_lock_clusters
);

/* Device Context 声明 */
ZB_HA_DECLARE_DOOR_LOCK_CTX(
	door_lock_ctx,
	door_lock_ep
);

/* ============================================================================
 * PC817 点动脉冲控制逻辑
 * ============================================================================ */

/* 脉冲到期回调：拉低引脚断开光耦，门锁按钮完成一次按压闭环 */
static void pulse_off_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);
	gpio_pin_set_dt(&optocoupler, 0);
	if (status_led.port) {
		gpio_pin_set_dt(&status_led, 0);
	}

	/* 门锁状态自动回到上锁闭合态，保持状态干净 */
	g_lock_ctx.lock_state = ZCL_DOOR_LOCK_STATE_LOCKED;
	g_lock_ctx.on_off_state = ZB_ZCL_ON_OFF_IS_OFF;

	/* 开启 2 秒防重触发保护期 */
	g_in_cooldown = true;
	k_work_schedule(&g_cooldown_work, K_MSEC(CONFIG_MYHOME_UNLOCK_COOLDOWN_MS));
}

/* 冷却到期回调 */
static void cooldown_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);
	g_in_cooldown = false;
}

/* 触发开锁物理动作 */
void myhome_trigger_unlock_pulse(void)
{
	if (g_in_cooldown) {
		return;
	}

	/* 1. 导通 PC817 光耦 (高电平) */
	gpio_pin_set_dt(&optocoupler, 1);
	if (status_led.port) {
		gpio_pin_set_dt(&status_led, 1);
	}

	g_lock_ctx.lock_state = ZCL_DOOR_LOCK_STATE_UNLOCKED;
	g_lock_ctx.on_off_state = ZB_ZCL_ON_OFF_IS_ON;

	/* 2. 启动 300ms 单次定时器，到期自动断开 */
	k_work_schedule(&g_pulse_off_work, K_MSEC(CONFIG_MYHOME_UNLOCK_PULSE_MS));
}

/* ============================================================================
 * Zigbee ZCL 指令回调拦截处理
 * ============================================================================ */

static zb_uint8_t zcl_specific_cluster_handler(zb_uint8_t param)
{
	zb_zcl_parsed_hdr_t *cmd_info = ZB_BUF_GET_PARAM(param, zb_zcl_parsed_hdr_t);

	if (cmd_info->cluster_id == ZB_ZCL_CLUSTER_ID_DOOR_LOCK) {
		/* Door Lock Cluster: 0x01 = Unlock Door 命令 */
		if (cmd_info->cmd_id == 0x01) {
			myhome_trigger_unlock_pulse();
			return ZB_TRUE;
		}
	} else if (cmd_info->cluster_id == ZB_ZCL_CLUSTER_ID_ON_OFF) {
		/* On/Off Cluster: 0x01 = On, 0x02 = Toggle 命令均触发开锁 */
		if (cmd_info->cmd_id == ZB_ZCL_CMD_ON_OFF_ON_ID ||
		    cmd_info->cmd_id == ZB_ZCL_CMD_ON_OFF_TOGGLE_ID) {
			myhome_trigger_unlock_pulse();
			return ZB_TRUE;
		}
	}
	return ZB_FALSE;
}

/* ZBOSS 协议栈信号处理 (网络入网、离开、认证状态机) */
void zboss_signal_handler(zb_uint8_t param)
{
	zb_zdo_app_signal_hdr_t *sg_p = NULL;
	zb_zdo_app_signal_type_t sig = zb_get_app_signal(param, &sg_p);
	zb_ret_t status = ZB_GET_APP_SIGNAL_STATUS(param);

	switch (sig) {
	case ZB_BDB_SIGNAL_DEVICE_FIRST_START:
	case ZB_BDB_SIGNAL_DEVICE_REBOOT:
		if (status == RET_OK) {
			/* 设备启动/重启成功，进入休眠轮询模式 */
			zb_set_rx_on_when_idle(ZB_FALSE);
			zb_zdo_pim_set_long_poll_interval(CONFIG_MYHOME_ZED_SLEEP_POLL_MS);
		} else {
			/* 尚未入网，启动 BDB 自动搜网入网 */
			bdb_start_top_level_commissioning(ZB_BDB_NETWORK_STEERING);
		}
		break;

	case ZB_BDB_SIGNAL_STEERING:
		if (status == RET_OK) {
			/* 成功加入 Zigbee 网络！设置 1.5 秒长轮询睡眠 */
			zb_set_rx_on_when_idle(ZB_FALSE);
			zb_zdo_pim_set_long_poll_interval(CONFIG_MYHOME_ZED_SLEEP_POLL_MS);
			if (status_led.port) {
				/* 入网成功：长亮 2 秒提示后熄灭进入省电 */
				gpio_pin_set_dt(&status_led, 1);
				k_msleep(2000);
				gpio_pin_set_dt(&status_led, 0);
			}
		} else {
			/* 搜网失败，10 秒后重试 */
			k_sleep(K_SECONDS(10));
			bdb_start_top_level_commissioning(ZB_BDB_NETWORK_STEERING);
		}
		break;

	default:
		break;
	}

	if (param) {
		zb_buf_free(param);
	}
}

/* ============================================================================
 * 硬件配网按键中断处理 (短接 P0.20 到 GND 3 秒重置配网)
 * ============================================================================ */

static struct gpio_callback pair_btn_cb_data;

static void pair_btn_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);
	if (pair_button.port && gpio_pin_get_dt(&pair_button) > 0) {
		/* 持续按下达到 3 秒：擦除 Zigbee NVRAM 凭证并重新配网 */
		if (status_led.port) {
			for (int i = 0; i < 6; i++) {
				gpio_pin_toggle_dt(&status_led);
				k_msleep(150);
			}
		}
		zb_bdb_reset_to_factory_new();
	}
}

static void pair_button_callback(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);

	if (gpio_pin_get_dt(&pair_button) > 0) {
		/* 检测到按下，计划 3 秒后核验是否为长按 */
		k_work_schedule(&g_pair_btn_work, K_MSEC(3000));
	} else {
		/* 短按松开：取消长按重置，并在本地立即触发一次点动测试！ */
		k_work_cancel_delayable(&g_pair_btn_work);
		myhome_trigger_unlock_pulse();
	}
}

/* ============================================================================
 * 主函数与外设初始化
 * ============================================================================ */

void main(void)
{
	/* 1. 初始化 GPIO 引脚 */
	if (!device_is_ready(optocoupler.port)) {
		return;
	}
	gpio_pin_configure_dt(&optocoupler, GPIO_OUTPUT_INACTIVE);

	if (status_led.port && device_is_ready(status_led.port)) {
		gpio_pin_configure_dt(&status_led, GPIO_OUTPUT_INACTIVE);
	}

	/* 2. 初始化工作队列与定时器 */
	k_work_init_delayable(&g_pulse_off_work, pulse_off_work_handler);
	k_work_init_delayable(&g_cooldown_work, cooldown_work_handler);
	k_work_init_delayable(&g_pair_btn_work, pair_btn_work_handler);

	/* 3. 配置配网/手动测试引脚中断 (P0.20) */
	if (pair_button.port && device_is_ready(pair_button.port)) {
		gpio_pin_configure_dt(&pair_button, GPIO_INPUT);
		gpio_pin_interrupt_configure_dt(&pair_button, GPIO_INT_EDGE_BOTH);
		gpio_init_callback(&pair_btn_cb_data, pair_button_callback, BIT(pair_button.pin));
		gpio_add_callback(pair_button.port, &pair_btn_cb_data);
	}

	/* 4. 初始化门锁数据结构默认值 */
	g_lock_ctx.on_off_state = ZB_ZCL_ON_OFF_IS_OFF;
	g_lock_ctx.lock_state = ZCL_DOOR_LOCK_STATE_LOCKED;
	g_lock_ctx.lock_type = ZCL_DOOR_LOCK_TYPE_DEAD_BOLT;
	g_lock_ctx.identify_time = 0;

	/* 5. 注册 Zigbee Device Context 与 ZCL 钩子 */
	ZB_AF_REGISTER_DEVICE_CTX(&door_lock_ctx);
	zb_zcl_add_cluster_handlers(
		MYHOME_LOCK_ENDPOINT,
		ZB_ZCL_CLUSTER_ID_DOOR_LOCK,
		zcl_specific_cluster_handler,
		NULL,
		NULL
	);
	zb_zcl_add_cluster_handlers(
		MYHOME_LOCK_ENDPOINT,
		ZB_ZCL_CLUSTER_ID_ON_OFF,
		zcl_specific_cluster_handler,
		NULL,
		NULL
	);

	/* 6. 注册短地址能力 (休眠终端节点 ZED, 不当路由器，关闭空闲接收) */
	zb_set_ed_node_descriptor(0, 0, 0);

	/* 7. 启动 ZBOSS Zigbee 协议栈主循环 (空闲时自动进入 Zephyr Tickless 低功耗休眠) */
	zigbee_enable();
}

