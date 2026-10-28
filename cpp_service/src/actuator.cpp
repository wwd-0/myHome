#include "actuator.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <cmath>

namespace myhome {

namespace {

size_t curlWriteIgnore(void*, size_t size, size_t nmemb, void*) {
    return size * nmemb;
}

} // namespace

HaWebhookAndRelayActuator::HaWebhookAndRelayActuator(const std::string& ha_base_url,
                                                     const std::string& unlock_webhook_id,
                                                     int relay_pulse_ms)
    : ha_base_url_(ha_base_url),
      unlock_webhook_id_(unlock_webhook_id),
      relay_pulse_ms_(relay_pulse_ms) {}

bool HaWebhookAndRelayActuator::executeUnlock(const UnlockDecision& decision) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        relay_active_until_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(relay_pulse_ms_ + 100);
    }

    std::string ha_url = ha_base_url_;
    std::string webhook = unlock_webhook_id_;
    int pulse_ms = relay_pulse_ms_;
    UnlockDecision d = decision;

    // 异步执行硬件光耦脉冲与 HA Webhook 推送
    std::thread([ha_url, webhook, pulse_ms, d]() {
        std::cout << "\n====================================================================" << std::endl;
        std::cout << "⚡ [ESP32-C6 + 光耦继电器] 触发开门闭环!" << std::endl;
        std::cout << "   • 认证模式: " << d.auth_mode << std::endl;
        std::cout << "   • 核验人物: " << d.person << " (置信度: " << d.score << ")" << std::endl;
        std::cout << "   • 动作触发: " << d.action << std::endl;
        std::cout << "   -> [0 ms]   GPIO 拉高：光耦导通 (ON)" << std::endl;

        std::string url = ha_url + "/api/webhook/" + webhook;
        nlohmann::json payload = {
            {"event", "unlock_door"},
            {"person", d.person},
            {"auth_mode", d.auth_mode},
            {"score", std::round(d.score * 1000.0f) / 1000.0f},
            {"action", d.action},
            {"relay_pulse_ms", pulse_ms},
            {"elapsed_ms", d.elapsed_ms}
        };
        std::string body = payload.dump();

        CURL* curl = curl_easy_init();
        if (curl) {
            struct curl_slist* headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteIgnore);
            CURLcode res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
                std::cerr << "⚠️ [HA Webhook] 推送失败: " << curl_easy_strerror(res) << std::endl;
            }
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(pulse_ms));
        std::cout << "   -> [" << pulse_ms << " ms] GPIO 拉低：光耦自动恢复断开 (OFF) —— 硬件闭环完成！" << std::endl;
        std::cout << "====================================================================\n" << std::endl;
    }).detach();

    return true;
}

bool HaWebhookAndRelayActuator::isRelayActive() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return std::chrono::steady_clock::now() < relay_active_until_;
}

// ============================================================================
// CooldownActuatorDecorator 实现
// ============================================================================

CooldownActuatorDecorator::CooldownActuatorDecorator(std::shared_ptr<IUnlockActuator> inner,
                                                     double cooldown_sec)
    : inner_(std::move(inner)),
      cooldown_sec_(cooldown_sec) {}

bool CooldownActuatorDecorator::executeUnlock(const UnlockDecision& decision) {
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (last_unlock_time_.time_since_epoch().count() > 0) {
            double elapsed = std::chrono::duration<double>(now - last_unlock_time_).count();
            if (elapsed < cooldown_sec_) {
                std::cout << "⏳ [开门冷却中] 距离上次开门仅 " << std::round(elapsed * 10.0) / 10.0
                          << "s，冷却窗口 " << cooldown_sec_ << "s，忽略重复开门指令。" << std::endl;
                return false;
            }
        }
        last_unlock_time_ = now;
    }
    return inner_->executeUnlock(decision);
}

bool CooldownActuatorDecorator::isRelayActive() const {
    return inner_->isRelayActive();
}

void CooldownActuatorDecorator::setCooldownSec(double sec) {
    std::lock_guard<std::mutex> lk(mtx_);
    cooldown_sec_ = sec;
}

double CooldownActuatorDecorator::getCooldownSec() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return cooldown_sec_;
}

bool CooldownActuatorDecorator::isInCooldown() const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (last_unlock_time_.time_since_epoch().count() == 0) return false;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_unlock_time_).count();
    return elapsed < cooldown_sec_;
}

double CooldownActuatorDecorator::getCooldownRemainingSec() const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (last_unlock_time_.time_since_epoch().count() == 0) return 0.0;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_unlock_time_).count();
    if (elapsed >= cooldown_sec_) return 0.0;
    return std::max(0.0, std::round((cooldown_sec_ - elapsed) * 10.0) / 10.0);
}

} // namespace myhome
