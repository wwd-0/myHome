#include "policy.hpp"
#include <algorithm>
#include <cmath>

namespace myhome {

// ============================================================================
// SwitchWindowPolicy 实现
// ============================================================================

SwitchWindowPolicy::SwitchWindowPolicy(double window_sec)
    : window_sec_(window_sec) {}

std::optional<UnlockDecision> SwitchWindowPolicy::evaluate(const AuthEvent& event) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto now = event.timestamp;

    if (event.type == AuthEventType::SwitchPressed) {
        // 情况 A: 开关按下前 window_sec_ 秒内是否已核验出自家人
        double sec_since_family = std::chrono::duration<double>(now - last_family_verified_time_).count();
        bool family_in_prev = (last_family_verified_time_.time_since_epoch().count() > 0) &&
                              (sec_since_family <= window_sec_) &&
                              (last_verified_family_ != "None");

        if (family_in_prev) {
            UnlockDecision decision;
            decision.passed = true;
            decision.person = last_verified_family_;
            decision.score = last_verified_score_;
            decision.auth_mode = name();
            decision.action = "SWITCH_CLICKED";
            decision.elapsed_ms = event.elapsed_ms;
            decision.relay_pulse_ms = 500;

            // 消费状态，防止单次识别被重复利用
            last_family_verified_time_ = std::chrono::steady_clock::time_point{};
            switch_pressed_until_ = std::chrono::steady_clock::time_point{};
            return decision;
        } else {
            // 开关按下前无有效人脸：开启等待人脸窗口
            switch_pressed_until_ = now + std::chrono::milliseconds(static_cast<long long>(window_sec_ * 1000.0));
            return std::nullopt;
        }
    } else if (event.type == AuthEventType::FrameRecognized) {
        if (event.frame_result.family_count > 0 && !event.frame_result.family_members.empty()) {
            last_family_verified_time_ = now;
            last_verified_family_ = event.frame_result.family_members[0];
            last_verified_score_ = 0.90f;
            for (const auto& p : event.frame_result.persons) {
                if (p.is_family && p.name == last_verified_family_) {
                    last_verified_score_ = p.score;
                    break;
                }
            }

            // 情况 B: 检查此前 window_sec_ 秒内开关是否已被按下
            if (now < switch_pressed_until_) {
                UnlockDecision decision;
                decision.passed = true;
                decision.person = last_verified_family_;
                decision.score = last_verified_score_;
                decision.auth_mode = name();
                decision.action = "SWITCH_PREV_CLICKED";
                decision.elapsed_ms = event.elapsed_ms;
                decision.relay_pulse_ms = 500;

                // 消费状态
                switch_pressed_until_ = std::chrono::steady_clock::time_point{};
                last_family_verified_time_ = std::chrono::steady_clock::time_point{};
                return decision;
            }
        }
    }
    return std::nullopt;
}

void SwitchWindowPolicy::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    last_family_verified_time_ = std::chrono::steady_clock::time_point{};
    last_verified_family_ = "None";
    last_verified_score_ = 0.0f;
    switch_pressed_until_ = std::chrono::steady_clock::time_point{};
}

void SwitchWindowPolicy::setWindowSec(double window_sec) {
    std::lock_guard<std::mutex> lk(mtx_);
    window_sec_ = window_sec;
}

double SwitchWindowPolicy::getWindowSec() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return window_sec_;
}

bool SwitchWindowPolicy::isFamilyPresenceActive(std::chrono::steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mtx_);
    double sec_since_family = std::chrono::duration<double>(now - last_family_verified_time_).count();
    return (last_family_verified_time_.time_since_epoch().count() > 0) &&
           (sec_since_family <= window_sec_) &&
           (last_verified_family_ != "None");
}

double SwitchWindowPolicy::getPresenceRemainingSec(std::chrono::steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mtx_);
    double sec_since_family = std::chrono::duration<double>(now - last_family_verified_time_).count();
    bool active = (last_family_verified_time_.time_since_epoch().count() > 0) &&
                  (sec_since_family <= window_sec_) &&
                  (last_verified_family_ != "None");
    return active ? std::max(0.0, std::round((window_sec_ - sec_since_family) * 10.0) / 10.0) : 0.0;
}

std::string SwitchWindowPolicy::getLastVerifiedFamily() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return last_verified_family_;
}

bool SwitchWindowPolicy::isWaitingForPerson(std::chrono::steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mtx_);
    return now < switch_pressed_until_;
}

double SwitchWindowPolicy::getWaitingRemainingSec(std::chrono::steady_clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (now >= switch_pressed_until_) return 0.0;
    double rem = std::chrono::duration<double>(switch_pressed_until_ - now).count();
    return std::max(0.0, std::round(rem * 10.0) / 10.0);
}

void SwitchWindowPolicy::clearSwitchWait() {
    std::lock_guard<std::mutex> lk(mtx_);
    switch_pressed_until_ = std::chrono::steady_clock::time_point{};
}

// ============================================================================
// GesturePolicy 实现
// ============================================================================

GesturePolicy::GesturePolicy(const std::vector<std::string>& allowed_gestures)
    : allowed_gestures_(allowed_gestures) {}

std::optional<UnlockDecision> GesturePolicy::evaluate(const AuthEvent& event) {
    if (event.type != AuthEventType::FrameRecognized) return std::nullopt;
    if (event.frame_result.family_count <= 0 || event.frame_result.family_members.empty()) {
        return std::nullopt;
    }
    if (event.gesture_name == "NONE" || event.gesture_name.empty()) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    bool is_allowed = std::find(allowed_gestures_.begin(), allowed_gestures_.end(), event.gesture_name) != allowed_gestures_.end();
    if (!is_allowed) return std::nullopt;

    UnlockDecision decision;
    decision.passed = true;
    decision.person = event.frame_result.family_members[0];
    decision.score = 0.90f;
    for (const auto& p : event.frame_result.persons) {
        if (p.is_family && p.name == decision.person) {
            decision.score = p.score;
            break;
        }
    }
    decision.auth_mode = name();
    decision.action = event.gesture_name;
    decision.elapsed_ms = event.elapsed_ms;
    decision.relay_pulse_ms = 500;
    return decision;
}

void GesturePolicy::reset() {
    // 无额外跨帧状态需要清理
}

void GesturePolicy::setAllowedGestures(const std::vector<std::string>& gestures) {
    std::lock_guard<std::mutex> lk(mtx_);
    allowed_gestures_ = gestures;
}

std::vector<std::string> GesturePolicy::getAllowedGestures() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return allowed_gestures_;
}

// ============================================================================
// VoicePolicy 实现 (独立语音口令 + 声纹核验开门策略，无需按键/人脸配合)
// ============================================================================

VoicePolicy::VoicePolicy(const std::vector<std::string>& allowed_commands)
    : allowed_commands_(allowed_commands) {}

std::optional<UnlockDecision> VoicePolicy::evaluate(const AuthEvent& event) {
    if (event.type != AuthEventType::VoiceCommand) return std::nullopt;

    const auto& vr = event.voice_result;
    if (!vr.vad_passed || !vr.command_matched || !vr.is_family) {
        return std::nullopt;
    }
    if (vr.command.empty() || vr.command == "NONE" || vr.person.empty() || vr.person == "None") {
        return std::nullopt;
    }

    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!allowed_commands_.empty()) {
            bool allowed = std::find(allowed_commands_.begin(), allowed_commands_.end(), vr.command) != allowed_commands_.end();
            if (!allowed) return std::nullopt;
        }
    }

    UnlockDecision decision;
    decision.passed = true;
    decision.person = vr.person;
    decision.score = vr.speaker_score;
    decision.auth_mode = name();
    decision.action = "VOICE:" + vr.command;
    decision.detail = "语音口令「" + vr.command + "」声纹核验通过，独立开门";
    decision.elapsed_ms = (event.elapsed_ms > 0)
                              ? event.elapsed_ms
                              : std::max(1, static_cast<int>(std::round(vr.latency_ms)));
    decision.relay_pulse_ms = 500;
    return decision;
}

void VoicePolicy::reset() {}

void VoicePolicy::setAllowedCommands(const std::vector<std::string>& commands) {
    std::lock_guard<std::mutex> lk(mtx_);
    allowed_commands_ = commands;
}

std::vector<std::string> VoicePolicy::getAllowedCommands() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return allowed_commands_;
}

// ============================================================================
// CompositePolicy 实现
// ============================================================================

void CompositePolicy::addPolicy(std::shared_ptr<IUnlockPolicy> policy) {
    std::lock_guard<std::mutex> lk(mtx_);
    policies_.push_back(std::move(policy));
}

std::optional<UnlockDecision> CompositePolicy::evaluateAll(const AuthEvent& event) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto& policy : policies_) {
        auto decision = policy->evaluate(event);
        if (decision.has_value()) {
            return decision;
        }
    }
    return std::nullopt;
}

void CompositePolicy::resetAll() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto& policy : policies_) {
        policy->reset();
    }
}

std::vector<std::string> CompositePolicy::getSupportedModeNames() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<std::string> names;
    names.reserve(policies_.size());
    for (const auto& p : policies_) {
        names.push_back(p->name());
    }
    return names;
}

} // namespace myhome
