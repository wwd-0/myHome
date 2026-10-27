#include "coordinator.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <thread>

namespace myhome {

namespace {

std::string currentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S");
    return oss.str();
}

} // namespace

VerifyServiceCoordinator::VerifyServiceCoordinator(std::shared_ptr<CppPersonEngine> person_engine,
                                                   std::shared_ptr<CppGestureEngine> gesture_engine,
                                                   std::shared_ptr<CompositePolicy> composite_policy,
                                                   std::shared_ptr<IUnlockActuator> actuator,
                                                   const std::string& cam_source,
                                                   double active_window_sec)
    : person_engine_(std::move(person_engine)),
      gesture_engine_(std::move(gesture_engine)),
      composite_policy_(std::move(composite_policy)),
      actuator_(std::move(actuator)),
      cam_source_(cam_source),
      active_window_sec_(active_window_sec) {
    last_decision_.timestamp = currentTimeStr();
}

VerifyServiceCoordinator::~VerifyServiceCoordinator() {
    stop();
}

void VerifyServiceCoordinator::stop() {
    std::unique_lock<std::mutex> lock(mtx_);
    running_ = false;
    stop_requested_ = true;
    cv_wakeup_.notify_all();
    cv_done_.notify_all();
}

void VerifyServiceCoordinator::extendVerifyTimeout(double window_sec) {
    auto now = std::chrono::steady_clock::now();
    auto new_deadline = now + std::chrono::milliseconds(static_cast<long long>(window_sec * 1000.0));
    if (new_deadline > verify_deadline_) {
        verify_deadline_ = new_deadline;
    }
    if (!verifying_) {
        verifying_ = true;
        stop_requested_ = false;
        verify_done_ = false;
        cv_wakeup_.notify_all();
    }
}

void VerifyServiceCoordinator::recordUnlockEvent(const UnlockDecision& decision) {
    std::lock_guard<std::mutex> lk(mtx_);
    unlock_count_++;
    last_decision_ = decision;
    last_decision_.timestamp = currentTimeStr();

    nlohmann::json event_item = {
        {"time", last_decision_.timestamp},
        {"person", decision.person},
        {"auth_mode", decision.auth_mode},
        {"score", std::round(decision.score * 1000.0f) / 1000.0f},
        {"action", decision.action},
        {"elapsed_ms", decision.elapsed_ms},
        {"relay_pulse_ms", decision.relay_pulse_ms}
    };
    history_.insert(history_.begin(), event_item);
    if (history_.size() > 20) history_.pop_back();
}

nlohmann::json VerifyServiceCoordinator::handleSmartSwitchTrigger() {
    auto now = std::chrono::steady_clock::now();
    AuthEvent event;
    event.type = AuthEventType::SwitchPressed;
    event.timestamp = now;
    event.elapsed_ms = 10;

    auto decision = composite_policy_->evaluateAll(event);
    double switch_win = 5.0;
    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) switch_win = sw_policy->getWindowSec();

    if (decision.has_value() && decision->passed) {
        // 情况 A: 开关按下前 N 秒内人脸已核验成功，直接开门
        actuator_->executeUnlock(*decision);
        recordUnlockEvent(*decision);
        return nlohmann::json{
            {"status", "unlocked"},
            {"passed", true},
            {"auth_mode", decision->auth_mode},
            {"person", decision->person},
            {"window_sec", switch_win},
            {"message", "开关按下前 " + std::to_string(static_cast<int>(switch_win)) + " 秒内人脸核验成功，判定开门！"}
        };
    } else {
        // 情况 B: 开关按下前无人脸，自适应延长/唤醒视频拉流互锁窗口期
        extendVerifyTimeout(switch_win + 1.5);
        return nlohmann::json{
            {"status", "waiting_person_match"},
            {"passed", false},
            {"auth_mode", "人体 + 智能开关"},
            {"window_sec", switch_win},
            {"message", "开关已按下，等待 " + std::to_string(static_cast<int>(switch_win)) + " 秒内人脸核验自家人..."}
        };
    }
}

nlohmann::json VerifyServiceCoordinator::handleSmartSwitchVerifyQuery() {
    auto now = std::chrono::steady_clock::now();
    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) {
        bool valid = sw_policy->isFamilyPresenceActive(now);
        return nlohmann::json{
            {"can_unlock", valid},
            {"person", sw_policy->getLastVerifiedFamily()},
            {"window_sec", sw_policy->getWindowSec()},
            {"remaining_sec", sw_policy->getPresenceRemainingSec(now)}
        };
    }
    return nlohmann::json{{"can_unlock", false}, {"person", "None"}, {"remaining_sec", 0.0}};
}

nlohmann::json VerifyServiceCoordinator::handleVoiceVerify(const nlohmann::json& req) {
    AuthEvent event;
    event.type = AuthEventType::VoiceCommand;
    event.timestamp = std::chrono::steady_clock::now();
    event.voice_text = req.value("text", "");

    auto decision = composite_policy_->evaluateAll(event);
    if (decision.has_value() && decision->passed) {
        actuator_->executeUnlock(*decision);
        recordUnlockEvent(*decision);
        return nlohmann::json{{"status", "unlocked"}, {"passed", true}, {"person", decision->person}};
    }
    return nlohmann::json{
        {"status", "reserved"},
        {"passed", false},
        {"auth_mode", "声音核验"},
        {"message", "声纹比对与 ASR 说话内容识别模块接口已就绪，等待后续音频模型接入"}
    };
}

nlohmann::json VerifyServiceCoordinator::testSingleImage(const std::string& b64_img) {
    nlohmann::json pred = person_engine_->predictBase64(b64_img);
    if (pred.contains("family_count") && pred["family_count"].get<int>() > 0) {
        auto now = std::chrono::steady_clock::now();
        AuthEvent event;
        event.type = AuthEventType::FrameRecognized;
        event.timestamp = now;
        event.elapsed_ms = 15;
        event.frame_result.family_count = pred.value("family_count", 0);
        event.frame_result.stranger_count = pred.value("stranger_count", 0);
        if (pred.contains("family_members")) {
            for (const auto& m : pred["family_members"]) {
                event.frame_result.family_members.push_back(m.get<std::string>());
            }
        }

        auto decision = composite_policy_->evaluateAll(event);
        if (decision.has_value() && decision->passed) {
            actuator_->executeUnlock(*decision);
            recordUnlockEvent(*decision);
        }
    }
    return pred;
}

nlohmann::json VerifyServiceCoordinator::triggerManualRelayTest() {
    UnlockDecision decision;
    decision.passed = true;
    decision.person = "Web手动测试";
    decision.auth_mode = "手动模拟";
    decision.score = 1.0f;
    decision.action = "MANUAL_BUTTON";
    decision.elapsed_ms = 12;
    decision.relay_pulse_ms = 500;

    actuator_->executeUnlock(decision);
    recordUnlockEvent(decision);
    return nlohmann::json{{"status", "ok"}, {"message", "已触发 ESP32-C6 光耦 500ms 点动脉冲"}};
}

nlohmann::json VerifyServiceCoordinator::requestVerifySync(double custom_timeout_sec) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!verifying_) {
        double timeout = (custom_timeout_sec > 0.0) ? custom_timeout_sec : active_window_sec_;
        extendVerifyTimeout(timeout);
    }
    cv_done_.wait(lock, [this]() { return verify_done_ || !running_; });
    return nlohmann::json{
        {"passed", last_decision_.passed},
        {"person", last_decision_.person},
        {"auth_mode", last_decision_.auth_mode},
        {"score", last_decision_.score},
        {"action", last_decision_.action},
        {"elapsed_ms", last_decision_.elapsed_ms},
        {"timestamp", last_decision_.timestamp}
    };
}

nlohmann::json VerifyServiceCoordinator::requestVerifyAsync() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (verifying_) {
        return nlohmann::json{{"status", "already_verifying"}, {"message", "服务已在拉流核验中"}};
    }
    extendVerifyTimeout(active_window_sec_);
    return nlohmann::json{{"status", "started"}, {"window_sec", active_window_sec_}};
}

nlohmann::json VerifyServiceCoordinator::requestStopVerify() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (verifying_) {
        stop_requested_ = true;
        return nlohmann::json{{"status", "stopping"}};
    }
    return nlohmann::json{{"status", "already_idle"}};
}

nlohmann::json VerifyServiceCoordinator::getStatus() {
    std::unique_lock<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now();
    bool relay_on = actuator_->isRelayActive();

    bool presence_active = false;
    double presence_rem = 0.0;
    std::string last_family = "None";
    double switch_win = 5.0;

    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) {
        presence_active = sw_policy->isFamilyPresenceActive(now);
        presence_rem = sw_policy->getPresenceRemainingSec(now);
        last_family = sw_policy->getLastVerifiedFamily();
        switch_win = sw_policy->getWindowSec();
    }

    return nlohmann::json{
        {"service", "myhome_cpp_verify_server"},
        {"state", verifying_ ? "VERIFYING_STREAM" : "IDLE_SLEEP"},
        {"model_engine", "LightPersonIDNet (MobileNetV3 + GeM)"},
        {"supported_modes", composite_policy_->getSupportedModeNames()},
        {"remaining_sec", verifying_ ? std::round(remaining_sec_ * 10.0) / 10.0 : 0.0},
        {"window_sec", active_window_sec_},
        {"switch_face_window_sec", switch_win},
        {"enrolled_members", person_engine_->getEnrolledNames()},
        {"live_person", live_person_},
        {"live_score", std::round(live_score_ * 1000.0f) / 1000.0f},
        {"live_has_stranger", live_has_stranger_},
        {"family_presence_active", presence_active},
        {"presence_remaining_sec", presence_rem},
        {"last_verified_family", last_family},
        {"relay_active", relay_on},
        {"unlock_count", unlock_count_.load()},
        {"last_result", {
            {"passed", last_decision_.passed},
            {"person", last_decision_.person},
            {"auth_mode", last_decision_.auth_mode},
            {"score", last_decision_.score},
            {"action", last_decision_.action},
            {"elapsed_ms", last_decision_.elapsed_ms},
            {"timestamp", last_decision_.timestamp}
        }},
        {"history", history_}
    };
}

std::vector<uchar> VerifyServiceCoordinator::getLatestJpeg() {
    std::lock_guard<std::mutex> lock(frame_mtx_);
    return latest_jpeg_;
}

nlohmann::json VerifyServiceCoordinator::getMembersSummary() {
    return person_engine_->getMembersSummary();
}

std::vector<uchar> VerifyServiceCoordinator::getPhotoBinary(const std::string& filename) {
    return person_engine_->getPhotoBinary(filename);
}

nlohmann::json VerifyServiceCoordinator::uploadPhoto(const std::string& name, const std::string& base64_image) {
    return person_engine_->uploadPhoto(name, base64_image);
}

nlohmann::json VerifyServiceCoordinator::deletePhoto(const std::string& filename) {
    return person_engine_->deletePhoto(filename);
}

nlohmann::json VerifyServiceCoordinator::reloadGallery() {
    return person_engine_->reloadGallery();
}

void VerifyServiceCoordinator::runMainThreadLoop() {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay", 1);

    while (running_) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            live_person_ = "None";
            live_score_ = 0.0f;
            live_has_stranger_ = false;
            remaining_sec_ = 0.0;
            cv_wakeup_.wait(lock, [this]() { return verifying_ || !running_; });
            if (!running_) break;
        }

        auto t_start = std::chrono::steady_clock::now();
        std::cout << "\n📡 [C++ 核验服务] 收到唤醒信号！正在打开摄像头视频流..." << std::endl;

        cv::VideoCapture cap;
        if (cam_source_ == "0" || cam_source_.empty()) {
            cap.open(0);
        } else {
            cap.open(cam_source_, cv::CAP_FFMPEG);
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        bool passed = false;
        UnlockDecision final_decision;

        if (!cap.isOpened()) {
            std::cerr << "❌ [C++ 核验服务] 无法打开视频源: " << cam_source_ << std::endl;
        } else {
            gesture_engine_->resetStreak();
            cv::Mat frame;

            while (running_) {
                auto now = std::chrono::steady_clock::now();
                double remaining_sec = 0.0;
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (stop_requested_) {
                        std::cout << "⏹️ [C++ 核验服务] 收到手动停止指令，释放摄像头。" << std::endl;
                        break;
                    }
                    if (now >= verify_deadline_) {
                        std::cout << "⏱️ [C++ 核验服务] 窗口期结束，关闭视频流回到 0% CPU 休眠态。" << std::endl;
                        break;
                    }
                    remaining_sec = std::chrono::duration<double>(verify_deadline_ - now).count();
                    remaining_sec_ = remaining_sec;
                }

                if (!cap.read(frame) || frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                    continue;
                }

                // 1. 高性能纯净视觉感知 (无无用 JPEG 渲染，节约 90% CPU)
                FrameRecognitionResult frame_res = person_engine_->detectAndRecognize(frame);
                int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());

                // 2. 检测手势
                auto gesture_res = gesture_engine_->detectGesture(frame);
                std::string gesture_name = gesture_res.value_or("NONE");

                // 更新状态缓存
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    live_has_stranger_ = (frame_res.stranger_count > 0);
                    if (frame_res.family_count > 0 && !frame_res.family_members.empty()) {
                        live_person_ = frame_res.family_members[0];
                        live_score_ = 0.90f;
                    } else {
                        live_person_ = live_has_stranger_ ? "Stranger" : "None";
                        live_score_ = 0.0f;
                    }
                }

                // 3. 构建领域事件并由策略组合器统一评估 (遵循 Strategy + OCP)
                AuthEvent event;
                event.type = AuthEventType::FrameRecognized;
                event.timestamp = now;
                event.frame_result = frame_res;
                event.gesture_name = gesture_name;
                event.elapsed_ms = elapsed_ms;

                auto decision = composite_policy_->evaluateAll(event);
                if (decision.has_value() && decision->passed) {
                    passed = true;
                    final_decision = *decision;

                    // 在最终抓拍帧绘制绿色开门指示
                    cv::rectangle(frame, cv::Point(15, 15), cv::Point(780, 72), cv::Scalar(0, 180, 60), -1);
                    std::string banner = "UNLOCKED! [" + final_decision.person + " + " + final_decision.action + "] (" +
                                         std::to_string(elapsed_ms) + "ms)";
                    cv::putText(frame, banner, cv::Point(28, 54), cv::FONT_HERSHEY_SIMPLEX,
                                0.85, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

                    {
                        std::vector<uchar> buf;
                        cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
                        std::lock_guard<std::mutex> flk(frame_mtx_);
                        latest_jpeg_ = std::move(buf);
                    }

                    // 执行硬件开门与 HA Webhook 推送
                    actuator_->executeUnlock(final_decision);
                    recordUnlockEvent(final_decision);
                    break; // 认出即关流，释放视频硬件
                }

                // 编码最新帧供前端实时预览 (/api/frame.jpg)
                {
                    std::vector<uchar> buf;
                    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 75});
                    std::lock_guard<std::mutex> flk(frame_mtx_);
                    latest_jpeg_ = std::move(buf);
                }
            }
            cap.release();
        }

        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (!passed) {
                last_decision_ = UnlockDecision{};
                last_decision_.timestamp = currentTimeStr();
            }
            verifying_ = false;
            stop_requested_ = false;
            remaining_sec_ = 0.0;
            verify_done_ = true;
            cv_done_.notify_all();
        }
    }
}

} // namespace myhome
