#pragma once

#include "domain_types.hpp"
#include <memory>
#include <string>
#include <chrono>
#include <mutex>

namespace myhome {

// 执行器接口 (遵循 DIP & LSP)
class IUnlockActuator {
public:
    virtual ~IUnlockActuator() = default;
    virtual bool executeUnlock(const UnlockDecision& decision) = 0;
    virtual bool isRelayActive() const = 0;
};

// 实际执行器: 触发 ESP32-C6 光耦 500ms 脉冲与 HA Webhook (遵循 SRP)
class HaWebhookAndRelayActuator : public IUnlockActuator {
public:
    HaWebhookAndRelayActuator(const std::string& ha_base_url,
                              const std::string& unlock_webhook_id,
                              int relay_pulse_ms = 500);

    bool executeUnlock(const UnlockDecision& decision) override;
    bool isRelayActive() const override;

    void setRelayPulseMs(int ms) { relay_pulse_ms_ = ms; }

private:
    std::string ha_base_url_;
    std::string unlock_webhook_id_;
    int relay_pulse_ms_;
    mutable std::mutex mtx_;
    std::chrono::steady_clock::time_point relay_active_until_{};
};

// 执行器装饰器: 冷却时间控制 (Decorator Pattern - 遵循 OCP & SRP)
class CooldownActuatorDecorator : public IUnlockActuator {
public:
    CooldownActuatorDecorator(std::shared_ptr<IUnlockActuator> inner, double cooldown_sec = 10.0);

    bool executeUnlock(const UnlockDecision& decision) override;
    bool isRelayActive() const override;

    void setCooldownSec(double sec);
    double getCooldownSec() const;
    bool isInCooldown() const;

private:
    std::shared_ptr<IUnlockActuator> inner_;
    double cooldown_sec_;
    mutable std::mutex mtx_;
    std::chrono::steady_clock::time_point last_unlock_time_{};
};

} // namespace myhome
