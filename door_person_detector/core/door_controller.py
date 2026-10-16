import time
import urllib.request
import json
import threading
from typing import Callable, List, Optional, Tuple
from .recognizer import RecognitionResult


class DoorUnlockController:
    """
    生产级自动开门安全控制器 (支持单人与多人同框核验)
    - 多人同框精准辨别:
      1. 全自家人 (如 爸爸 + 妈妈): 同时识别并通过，协同欢迎开门
      2. 自家人 + 陌生人同行 (疑似尾随/挟持安全防线): 可配置是否阻断自动开门，并向 HA 触发尾随预警
      3. 全陌生人: 严密拒识，大门保持锁闭
    - 时序多帧验证: 连续 N 帧确认，杜绝单帧抖动
    - 冷却自锁机制: 开门成功后自动进入冷却倒计时 (如 10 秒)
    - 异步非阻塞 Webhook: 不卡顿视觉主循环
    """
    def __init__(self,
                 enabled: bool = True,
                 confirm_consecutive_frames: int = 3,
                 cooldown_seconds: float = 10.0,
                 allow_unlock_with_stranger: bool = False,
                 ha_webhook_url: str = "",
                 ha_stranger_alert_webhook_url: str = "",
                 on_unlock_callback: Optional[Callable[[List[str], float], None]] = None):
        self.enabled = enabled
        self.confirm_consecutive_frames = confirm_consecutive_frames
        self.cooldown_seconds = cooldown_seconds
        self.allow_unlock_with_stranger = allow_unlock_with_stranger
        self.ha_webhook_url = ha_webhook_url
        self.ha_stranger_alert_webhook_url = ha_stranger_alert_webhook_url
        self.on_unlock_callback = on_unlock_callback

        self.current_candidate_names: List[str] = []
        self.consecutive_count: int = 0
        self.last_unlock_time: float = 0.0
        self.last_alert_time: float = 0.0

    @property
    def is_in_cooldown(self) -> bool:
        """是否处于开门后的冷却保护期中"""
        return (time.time() - self.last_unlock_time) < self.cooldown_seconds

    @property
    def remaining_cooldown_sec(self) -> float:
        rem = self.cooldown_seconds - (time.time() - self.last_unlock_time)
        return max(0.0, rem)

    def process_result(self, result: Optional[RecognitionResult]) -> bool:
        """单人兼容接口"""
        results = [result] if result is not None else []
        unlocked, _ = self.process_frame_results(results)
        return unlocked

    def process_frame_results(self, results: List[RecognitionResult]) -> Tuple[bool, str]:
        """
        核心仲裁入口：处理单帧中同时检测到的全部主体
        返回: (是否触发开门, 当前状态中文描述文本)
        """
        if not self.enabled:
            return False, "系统未启用自动开门"

        # 冷却保护期
        if self.is_in_cooldown:
            self.consecutive_count = 0
            self.current_candidate_names = []
            return False, f"🎉 [已开门] 冷却中: {self.remaining_cooldown_sec:.1f}s"

        # 画面中无人
        if not results:
            self.consecutive_count = 0
            self.current_candidate_names = []
            return False, "🔒 [门已上锁] 待机守候"

        # 分类统计画面中的自家人与陌生人
        eligible_family = [r for r in results if (not r.is_stranger) and r.can_unlock]
        strangers = [r for r in results if r.is_stranger]

        # 场景 1: 画面中全部是陌生人
        if not eligible_family:
            self.consecutive_count = 0
            self.current_candidate_names = []
            return False, f"⚠️ 检测到陌生人 ({len(strangers)}人)，保持锁闭"

        # 场景 2: 自家人 + 陌生人同框 (疑似尾随 / 访客随行)
        family_names = sorted(list(set(r.person_name for r in eligible_family)))
        names_str = "、".join(family_names)

        if len(strangers) > 0 and not self.allow_unlock_with_stranger:
            # 严格防尾随模式：发现自家人身边有陌生人，立即拦截开门并清零防抖
            self.consecutive_count = 0
            self.current_candidate_names = []
            self._trigger_tailgating_alert(names_str, len(strangers))
            return False, f"🚨 疑似尾随风险！检测到【{names_str}】但有 {len(strangers)} 位陌生人同行，保持锁闭"

        # 场景 3: 允许通过（全自家人 或 允许陌生人随行模式）
        # 对比当前帧自家人与上一帧是否一致
        if family_names == self.current_candidate_names:
            self.consecutive_count += 1
        else:
            self.current_candidate_names = family_names
            self.consecutive_count = 1

        top_score = max(r.similarity_score for r in eligible_family)

        # 检查是否达成连续 N 帧门限
        if self.consecutive_count >= self.confirm_consecutive_frames:
            self._trigger_unlock(self.current_candidate_names, top_score, has_accompanying_stranger=(len(strangers) > 0))
            self.last_unlock_time = time.time()
            self.consecutive_count = 0
            self.current_candidate_names = []
            return True, f"🎉 开门成功！欢迎回家: {names_str}"

        return False, f"⏳ [核验中] {names_str} ({self.consecutive_count}/{self.confirm_consecutive_frames} 帧)"

    def _trigger_unlock(self, family_names: List[str], score: float, has_accompanying_stranger: bool = False):
        """触发开门动作 (异步执行)"""
        names_str = "、".join(family_names)
        extra_note = " (注意: 伴随陌生人通行)" if has_accompanying_stranger else ""
        print(f"\n=======================================================")
        print(f"🎉【安全开门认证通过】欢迎回家: {names_str}{extra_note} (最高置信度: {score:.4f})")
        print(f"=======================================================\n")

        if self.on_unlock_callback:
            try:
                self.on_unlock_callback(family_names, score)
            except Exception as e:
                print(f"[DoorController] 自定义开门回调异常: {e}")

        if self.ha_webhook_url:
            threading.Thread(
                target=self._send_ha_webhook,
                args=(family_names, score, has_accompanying_stranger),
                daemon=True
            ).start()

    def _send_ha_webhook(self, family_names: List[str], score: float, has_stranger: bool):
        payload = {
            "source": "door_person_detector",
            "persons": family_names,
            "primary_person": family_names[0] if family_names else "自家人",
            "score": round(float(score), 4),
            "multiple_persons": len(family_names) > 1,
            "has_stranger_accompanying": has_stranger,
            "timestamp": int(time.time()),
            "action": "door_unlock"
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=self.ha_webhook_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                print(f"[DoorController] 已成功发送 HA 开门 Webhook，状态码: {resp.status}")
        except Exception as e:
            print(f"[DoorController] HA Webhook 发送失败: {e}")

    def _trigger_tailgating_alert(self, family_names_str: str, stranger_count: int):
        """尾随/可疑警报触发 (带冷却去重，防止频繁骚扰)"""
        now = time.time()
        if (now - self.last_alert_time) < 15.0:
            return
        self.last_alert_time = now

        print(f"🚨【安全告警】检测到尾随嫌疑: 自家人【{family_names_str}】身边有 {stranger_count} 位未授权陌生人同行！")
        if self.ha_stranger_alert_webhook_url:
            threading.Thread(
                target=self._send_tailgating_webhook,
                args=(family_names_str, stranger_count),
                daemon=True
            ).start()

    def _send_tailgating_webhook(self, family_names_str: str, stranger_count: int):
        payload = {
            "source": "door_person_detector",
            "alert_type": "tailgating_risk",
            "family_member": family_names_str,
            "stranger_count": stranger_count,
            "timestamp": int(time.time())
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=self.ha_stranger_alert_webhook_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                print(f"[DoorController] 已成功上报 HA 尾随防范告警: {resp.status}")
        except Exception as e:
            print(f"[DoorController] HA 尾随告警发送跳过: {e}")
