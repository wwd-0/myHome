#pragma once

#include "domain_types.hpp"
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <mutex>
#include <chrono>

namespace myhome {

// 策略接口 (Strategy Pattern - 遵循 OCP & LSP)
class IUnlockPolicy {
public:
    virtual ~IUnlockPolicy() = default;
    virtual std::string name() const = 0;
    virtual std::optional<UnlockDecision> evaluate(const AuthEvent& event) = 0;
    virtual void reset() = 0;
};

// 策略 1: 人体 + 智能开关互锁策略 (前后 N 秒窗口期)
class SwitchWindowPolicy : public IUnlockPolicy {
public:
    explicit SwitchWindowPolicy(double window_sec = 5.0);
    std::string name() const override { return "人体 + 智能开关"; }
    std::optional<UnlockDecision> evaluate(const AuthEvent& event) override;
    void reset() override;

    void setWindowSec(double window_sec);
    double getWindowSec() const;

    // 自家人在场状态查询
    bool isFamilyPresenceActive(std::chrono::steady_clock::time_point now) const;
    double getPresenceRemainingSec(std::chrono::steady_clock::time_point now) const;
    std::string getLastVerifiedFamily() const;

    // 开关等待自家人状态查询
    bool isWaitingForPerson(std::chrono::steady_clock::time_point now) const;
    double getWaitingRemainingSec(std::chrono::steady_clock::time_point now) const;
    void clearSwitchWait();

private:
    double window_sec_;
    mutable std::mutex mtx_;
    std::chrono::steady_clock::time_point last_family_verified_time_{};
    std::string last_verified_family_ = "None";
    float last_verified_score_ = 0.0f;
    std::chrono::steady_clock::time_point switch_pressed_until_{};
};

// 策略 2: 人体 + 手势核验策略
class GesturePolicy : public IUnlockPolicy {
public:
    explicit GesturePolicy(const std::vector<std::string>& allowed_gestures = {"OK", "VICTORY"});
    std::string name() const override { return "人体 + 手势"; }
    std::optional<UnlockDecision> evaluate(const AuthEvent& event) override;
    void reset() override;

    void setAllowedGestures(const std::vector<std::string>& gestures);
    std::vector<std::string> getAllowedGestures() const;

private:
    std::vector<std::string> allowed_gestures_;
    mutable std::mutex mtx_;
};

// 策略 3: 声音核验策略 (预留)
class VoicePolicy : public IUnlockPolicy {
public:
    std::string name() const override { return "声音核验"; }
    std::optional<UnlockDecision> evaluate(const AuthEvent& event) override;
    void reset() override;
};

// 策略组合器 (Composite Pattern - 遵循 OCP & DIP)
class CompositePolicy {
public:
    void addPolicy(std::shared_ptr<IUnlockPolicy> policy);
    std::optional<UnlockDecision> evaluateAll(const AuthEvent& event);
    void resetAll();
    std::vector<std::string> getSupportedModeNames() const;

    template <typename T>
    std::shared_ptr<T> getPolicy() const {
        std::lock_guard<std::mutex> lk(mtx_);
        for (const auto& p : policies_) {
            auto casted = std::dynamic_pointer_cast<T>(p);
            if (casted) return casted;
        }
        return nullptr;
    }

private:
    std::vector<std::shared_ptr<IUnlockPolicy>> policies_;
    mutable std::mutex mtx_;
};

} // namespace myhome
