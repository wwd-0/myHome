import json
import urllib.request
from typing import Callable, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class HomeAssistantConnector:
    """
    负责 Mac AI 守护服务与 UTM 虚拟机中 Home Assistant + SLZB-06 (Zigbee2MQTT) 的双向通信：
    1. 下行：监听涂鸦 Zigbee 人体存在传感器 (presence & target_distance) 唤醒识别；
    2. 上行：通过 HA Webhook 触发【开门指令】或【可疑人员抓拍告警】。
    """

    def __init__(self, ha_cfg: dict, on_tuya_presence_callback: Callable[[bool, Optional[float]], None]):
        self.base_url = ha_cfg.get("base_url", "http://homeassistant.local:8123").rstrip("/")
        self.unlock_webhook_id = ha_cfg.get("unlock_webhook_id", "ai_door_unlock_event")
        self.alert_webhook_id = ha_cfg.get("alert_webhook_id", "ai_suspicious_alert_event")
        self.on_tuya_presence_callback = on_tuya_presence_callback

        self.mqtt_cfg = ha_cfg.get("mqtt", {})
        self.mqtt_enabled = bool(self.mqtt_cfg.get("enabled", False))
        self.trigger_distance_max_m = float(self.mqtt_cfg.get("trigger_distance_max_m", 1.8))
        self.mqtt_client = None

        if self.mqtt_enabled:
            self._start_mqtt_listener()

    def _start_mqtt_listener(self):
        if mqtt is None:
            print("⚠️ [HA-MQTT] 未安装 paho-mqtt，请运行: pip install paho-mqtt")
            return

        broker_host = self.mqtt_cfg.get("broker_host", "homeassistant.local")
        broker_port = int(self.mqtt_cfg.get("broker_port", 1883))
        topic = self.mqtt_cfg.get("tuya_sensor_topic", "zigbee2mqtt/foyer_tuya_presence")

        self.mqtt_client = mqtt.Client()
        username = self.mqtt_cfg.get("username", "")
        password = self.mqtt_cfg.get("password", "")
        if username:
            self.mqtt_client.username_pw_set(username, password)

        def on_connect(client, userdata, flags, rc):
            print(f"📡 [HA-MQTT] 已连接 SLZB-06 / Zigbee2MQTT ({broker_host}:{broker_port})，监听主题: {topic}")
            client.subscribe(topic)

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                # 兼容不同型号涂鸦 Zigbee 传感器的字段名 (presence 或 occupancy)
                presence = bool(payload.get("presence", payload.get("occupancy", False)))
                distance = payload.get("target_distance", payload.get("distance", None))
                if distance is not None:
                    distance = float(distance)

                if presence and (distance is None or distance <= self.trigger_distance_max_m):
                    self.on_tuya_presence_callback(True, distance)
                elif not presence:
                    self.on_tuya_presence_callback(False, None)
            except Exception as e:
                print(f"⚠️ [HA-MQTT] 解析涂鸦传感器 JSON 失败: {e}")

        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        try:
            self.mqtt_client.connect_async(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"⚠️ [HA-MQTT] 连接 MQTT 失败: {e}")

    def _post_json(self, url: str, payload: dict):
        if requests is not None:
            requests.post(url, json=payload, timeout=2.5)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2.5)

    def trigger_door_unlock(self, person_name: str, auth_mode: str, face_score: float, extra_detail: str = ""):
        """向 UTM 中的 Home Assistant 发送开门 Webhook"""
        url = f"{self.base_url}/api/webhook/{self.unlock_webhook_id}"
        payload = {
            "event": "unlock_door",
            "person": person_name,
            "auth_mode": auth_mode,  # "人脸 + 手势" 或 "人脸 + 声纹"
            "face_score": round(face_score, 3),
            "detail": extra_detail,
        }
        print(f"🔓 [HA-Webhook] 验证通过！正在通知 Home Assistant 开门 -> {payload}")
        try:
            self._post_json(url, payload)
        except Exception as e:
            print(f"ℹ️ [HA-Webhook] (本地演示/未接通HA) 发送请求至 {url} 提示: {e}")

    def trigger_suspicious_alert(self, reason: str, snapshot_path: Optional[str] = None):
        """向 UTM 中的 Home Assistant 发送可疑人员告警 Webhook"""
        url = f"{self.base_url}/api/webhook/{self.alert_webhook_id}"
        payload = {
            "event": "suspicious_person_alert",
            "reason": reason,
            "snapshot_path": snapshot_path or "",
        }
        print(f"🚨 [HA-Webhook] 触发可疑人员高危告警！ -> {payload}")
        try:
            self._post_json(url, payload)
        except Exception as e:
            print(f"ℹ️ [HA-Webhook] (本地演示/未接通HA) 告警请求至 {url} 提示: {e}")
