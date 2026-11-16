/*
 * ==============================================================================
 * MyHome Zigbee 门锁定义头文件 (zb_door_lock.h)
 * 采用 Zigbee 3.0 Home Automation Profile，双兼容 Door Lock & On/Off Cluster
 * ==============================================================================
 */

#ifndef ZB_DOOR_LOCK_H__
#define ZB_DOOR_LOCK_H__

#include <zboss_api.h>
#include <zboss_api_addons.h>

#define MYHOME_LOCK_ENDPOINT           1
#define MYHOME_ZIGBEE_PROFILE_ID       ZB_AF_HA_PROFILE_ID
#define MYHOME_LOCK_DEVICE_ID          ZB_HA_DOOR_LOCK_DEVICE_ID
#define MYHOME_LOCK_DEVICE_VER         1

/* 属性数量定义 */
#define MYHOME_BASIC_ATTR_COUNT        6
#define MYHOME_IDENTIFY_ATTR_COUNT     1
#define MYHOME_ON_OFF_ATTR_COUNT       1
#define MYHOME_DOOR_LOCK_ATTR_COUNT    2

/* 门锁状态枚举 (ZCL 规范) */
#define ZCL_DOOR_LOCK_STATE_NOT_FULLY_LOCKED  0x00
#define ZCL_DOOR_LOCK_STATE_LOCKED            0x01
#define ZCL_DOOR_LOCK_STATE_UNLOCKED          0x02

/* 门锁类型枚举 */
#define ZCL_DOOR_LOCK_TYPE_DEAD_BOLT          0x00
#define ZCL_DOOR_LOCK_TYPE_MAGNETIC           0x01
#define ZCL_DOOR_LOCK_TYPE_OTHER              0x02

struct myhome_lock_ctx {
	zb_uint8_t  on_off_state;
	zb_uint8_t  lock_state;
	zb_uint8_t  lock_type;
	zb_uint16_t identify_time;
};

void myhome_lock_clusters_init(void);
void myhome_trigger_unlock_pulse(void);

#endif /* ZB_DOOR_LOCK_H__ */

