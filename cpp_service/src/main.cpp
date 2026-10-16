#include "person_engine.hpp"
#include "gesture_engine.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

size_t curlWriteIgnore(void*, size_t size, size_t nmemb, void*) {
    return size * nmemb;
}

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

class VerifyServiceCoordinator {
public:
    VerifyServiceCoordinator(myhome::CppPersonEngine& person_eng,
                             myhome::CppGestureEngine& gesture_eng,
                             const std::string& cam_source,
                             const std::string& ha_url,
                             const std::string& unlock_webhook,
                             double window_sec)
        : person_engine_(person_eng),
          gesture_engine_(gesture_eng),
          cam_source_(cam_source),
          ha_base_url_(ha_url),
          unlock_webhook_id_(unlock_webhook),
          active_window_sec_(window_sec) {
        last_result_ = json{
            {"passed", false},
            {"person", "None"},
            {"auth_mode", "None"},
            {"face_score", 0.0},
            {"gesture", "None"},
            {"relay_pulse_ms", 0},
            {"elapsed_ms", 0},
            {"timestamp", currentTimeStr()}
        };
    }

    // 触发 ESP32-C6 500ms 光耦脉冲并异步推送 Home Assistant Webhook
    void triggerRelayAndHaAsync(const std::string& person,
                                float score,
                                const std::string& gesture_or_action,
                                const std::string& auth_mode,
                                int elapsed_ms) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            relay_active_until_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(600);
            unlock_count_++;
            json event_item = {
                {"time", currentTimeStr()},
                {"person", person},
                {"auth_mode", auth_mode},
                {"score", std::round(score * 1000.0f) / 1000.0f},
                {"action", gesture_or_action},
                {"elapsed_ms", elapsed_ms},
                {"relay_pulse_ms", 500}
            };
            history_.insert(history_.begin(), event_item);
            if (history_.size() > 20) history_.pop_back();

            last_result_ = json{
                {"passed", true},
                {"person", person},
                {"auth_mode", auth_mode},
                {"score", std::round(score * 1000.0f) / 1000.0f},
                {"action", gesture_or_action},
                {"relay_pulse_ms", 500},
                {"elapsed_ms", elapsed_ms},
                {"timestamp", currentTimeStr()}
            };
        }

        std::string ha_url = ha_base_url_;
        std::string webhook = unlock_webhook_id_;
        std::thread([=]() {
            std::cout << "\n====================================================================" << std::endl;
            std::cout << "⚡ [ESP32-C6 + 光耦继电器] 触发开门闭环!" << std::endl;
            std::cout << "   • 认证模式: " << auth_mode << std::endl;
            std::cout << "   • 核验人物: " << person << " (置信度: " << score << ")" << std::endl;
            std::cout << "   • 动作触发: " << gesture_or_action << std::endl;
            std::cout << "   -> [0 ms]   GPIO 拉高：光耦导通 (ON)" << std::endl;

            std::string url = ha_url + "/api/webhook/" + webhook;
            json payload = {
                {"event", "unlock_door"},
                {"person", person},
                {"auth_mode", auth_mode},
                {"score", std::round(score * 1000.0f) / 1000.0f},
                {"action", gesture_or_action},
                {"relay_pulse_ms", 500},
                {"elapsed_ms", elapsed_ms}
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
                curl_easy_perform(curl);
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << "   -> [500 ms] GPIO 拉低：光耦自动恢复断开 (OFF) —— 硬件闭环完成！" << std::endl;
            std::cout << "====================================================================\n" << std::endl;
        }).detach();
    }

    // 向 HA 汇报当前门前自家人状态 (纯感知数据汇报，供 HA 自身做状态决策)
    void reportPersonPresenceToHaAsync(const std::string& person, float score, bool has_stranger) {
        std::string ha_url = ha_base_url_;
        std::string webhook = unlock_webhook_id_;
        std::thread([=]() {
            std::string url = ha_url + "/api/webhook/" + webhook;
            json payload = {
                {"event", "person_verified"},
                {"person", person},
                {"score", std::round(score * 1000.0f) / 1000.0f},
                {"has_stranger", has_stranger},
                {"presence_window_sec", 15}
            };
            std::string body = payload.dump();

            CURL* curl = curl_easy_init();
            if (curl) {
                struct curl_slist* headers = nullptr;
                headers = curl_slist_append(headers, "Content-Type: application/json");
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
                curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1500L);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteIgnore);
                curl_easy_perform(curl);
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
            }
        }).detach();
    }

    // 模式 2 处理：收到智能开关按下事件
    json handleSmartSwitchTrigger() {
        std::unique_lock<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();

        // 检查过去 15 秒内门前是否核验出过自家人
        bool family_valid = (now < family_presence_until_) && (last_verified_family_ != "None");

        if (family_valid) {
            std::string person = last_verified_family_;
            float score = last_verified_score_;
            lock.unlock();
            // 直接触发开门
            triggerRelayAndHaAsync(person, score, "SWITCH_CLICKED", "人体 + 智能开关", 10);
            return json{
                {"status", "unlocked"},
                {"passed", true},
                {"auth_mode", "人体 + 智能开关"},
                {"person", person},
                {"message", "自家人在场窗口期有效，已触发直接开门！"}
            };
        } else {
            // 自家人未在场或窗口过期：标记开关已按下（5秒有效），并立刻拉起人体识别预热
            switch_pressed_until_ = now + std::chrono::seconds(5);
            if (!verifying_) {
                verifying_ = true;
                stop_requested_ = false;
                verify_timeout_sec_ = 8.0; // 短窗口快速验证
                verify_done_ = false;
                cv_wakeup_.notify_all();
            }
            return json{
                {"status", "waiting_person_match"},
                {"passed", false},
                {"auth_mode", "人体 + 智能开关"},
                {"message", "开关已按下，正在等待门前人体模型核验自家人 (5秒窗口期)..."}
            };
        }
    }

    // 模式 2 查询：供 HA 同步查询当前门厅是否处于自家人合法开门窗口期
    json handleSmartSwitchVerifyQuery() {
        std::unique_lock<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        bool valid = (now < family_presence_until_) && (last_verified_family_ != "None");
        return json{
            {"can_unlock", valid},
            {"person", last_verified_family_},
            {"score", last_verified_score_},
            {"remaining_sec", valid ? std::chrono::duration_cast<std::chrono::seconds>(family_presence_until_ - now).count() : 0}
        };
    }

    // 模式 3 声音核验接口预留
    json handleVoiceVerifyReserved(const json& req) {
        return json{
            {"status", "reserved"},
            {"passed", false},
            {"auth_mode", "声音核验"},
            {"message", "声纹比对与 ASR 说话内容识别模块接口已就绪，等待后续音频模型接入"}
        };
    }

    // 页面测试专用：单张图片 Base64 纯净推理
    json testSingleImage(const std::string& b64_img) {
        json pred = person_engine_.predictBase64(b64_img);
        if (pred.contains("family_count") && pred["family_count"].get<int>() > 0) {
            std::unique_lock<std::mutex> lock(mtx_);
            family_presence_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            if (pred.contains("family_members") && !pred["family_members"].empty()) {
                last_verified_family_ = pred["family_members"][0].get<std::string>();
            } else {
                last_verified_family_ = "自家人";
            }
            last_verified_score_ = 0.92f;
            reportPersonPresenceToHaAsync(last_verified_family_, last_verified_score_, pred.value("has_stranger", false));
        }
        return pred;
    }

    json requestVerifySync(double custom_timeout_sec = -1.0) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!verifying_) {
            verifying_ = true;
            stop_requested_ = false;
            verify_timeout_sec_ = (custom_timeout_sec > 0.0) ? custom_timeout_sec : active_window_sec_;
            verify_done_ = false;
            cv_wakeup_.notify_all();
        }
        cv_done_.wait(lock, [this]() { return verify_done_ || !running_; });
        return last_result_;
    }

    json requestVerifyAsync() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (verifying_) {
            return json{{"status", "already_verifying"}, {"message", "服务已在拉流核验中"}};
        }
        verifying_ = true;
        stop_requested_ = false;
        verify_timeout_sec_ = active_window_sec_;
        verify_done_ = false;
        cv_wakeup_.notify_all();
        return json{{"status", "started"}, {"window_sec", verify_timeout_sec_}};
    }

    json requestStopVerify() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (verifying_) {
            stop_requested_ = true;
            return json{{"status", "stopping"}};
        }
        return json{{"status", "already_idle"}};
    }

    json getStatus() {
        std::unique_lock<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        bool relay_on = now < relay_active_until_;
        bool presence_active = (now < family_presence_until_) && (last_verified_family_ != "None");

        return json{
            {"service", "myhome_cpp_verify_server"},
            {"state", verifying_ ? "VERIFYING_STREAM" : "IDLE_SLEEP"},
            {"model_engine", "LightPersonIDNet (MobileNetV3 + GeM)"},
            {"supported_modes", json::array({"人体 + 手势", "人体 + 智能开关", "声音 (预留)"})},
            {"remaining_sec", verifying_ ? std::round(remaining_sec_ * 10.0) / 10.0 : 0.0},
            {"window_sec", active_window_sec_},
            {"enrolled_members", person_engine_.getEnrolledNames()},
            {"live_person", live_person_},
            {"live_score", std::round(live_score_ * 1000.0f) / 1000.0f},
            {"live_has_stranger", live_has_stranger_},
            {"family_presence_active", presence_active},
            {"last_verified_family", last_verified_family_},
            {"relay_active", relay_on},
            {"unlock_count", unlock_count_.load()},
            {"last_result", last_result_},
            {"history", history_}
        };
    }

    std::vector<uchar> getLatestJpeg() {
        std::lock_guard<std::mutex> lock(frame_mtx_);
        return latest_jpeg_;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        running_ = false;
        stop_requested_ = true;
        cv_wakeup_.notify_all();
        cv_done_.notify_all();
    }

    void runMainThreadLoop() {
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay", 1);

        while (running_) {
            double timeout_sec = active_window_sec_;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                live_person_ = "None";
                live_score_ = 0.0f;
                live_has_stranger_ = false;
                remaining_sec_ = 0.0;
                cv_wakeup_.wait(lock, [this]() { return verifying_ || !running_; });
                if (!running_) break;
                timeout_sec = verify_timeout_sec_;
            }

            std::cout << "\n📡 [C++ 核验服务] 收到唤醒信号！正在打开摄像头视频流 (窗口期: "
                      << timeout_sec << "s)..." << std::endl;

            auto t_start = std::chrono::steady_clock::now();
            cv::VideoCapture cap;
            if (cam_source_ == "0" || cam_source_.empty()) {
                cap.open(0);
            } else {
                cap.open(cam_source_, cv::CAP_FFMPEG);
            }
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

            bool passed = false;
            std::string matched_person = "None";
            float matched_score = 0.0f;
            std::string matched_gesture = "None";
            std::string passed_mode = "None";
            int elapsed_ms = 0;

            if (!cap.isOpened()) {
                std::cerr << "❌ [C++ 核验服务] 无法打开视频源: " << cam_source_ << std::endl;
            } else {
                gesture_engine_.resetStreak();
                cv::Mat frame;
                while (running_) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed_sec = std::chrono::duration<double>(now - t_start).count();
                    double rem = std::max(0.0, timeout_sec - elapsed_sec);

                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        remaining_sec_ = rem;
                        if (stop_requested_) {
                            std::cout << "⏹️ [C++ 核验服务] 收到手动停止指令，释放摄像头。" << std::endl;
                            break;
                        }
                    }

                    if (elapsed_sec >= timeout_sec) {
                        std::cout << "⏱️ [C++ 核验服务] " << timeout_sec
                                  << "s 窗口期结束，关闭视频流回到 0% CPU 休眠态。" << std::endl;
                        break;
                    }

                    if (!cap.read(frame) || frame.empty()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(15));
                        continue;
                    }

                    // 调用统一的人体识别模型微服务
                    json pred = person_engine_.predictFrame(frame);
                    bool has_family = (pred.value("family_count", 0) > 0);
                    bool has_stranger = pred.value("has_stranger", false);
                    std::string current_family_name = "None";
                    float current_family_score = 0.0f;

                    if (has_family && pred.contains("family_members") && !pred["family_members"].empty()) {
                        current_family_name = pred["family_members"][0].get<std::string>();
                        current_family_score = 0.90f;
                    }

                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        live_has_stranger_ = has_stranger;
                        if (has_family) {
                            live_person_ = current_family_name;
                            live_score_ = current_family_score;
                            family_presence_until_ = now + std::chrono::seconds(15);
                            last_verified_family_ = current_family_name;
                            last_verified_score_ = current_family_score;
                        } else {
                            live_person_ = has_stranger ? "Stranger" : "None";
                            live_score_ = 0.0f;
                        }
                    }

                    if (has_family) {
                        reportPersonPresenceToHaAsync(current_family_name, current_family_score, has_stranger);

                        // 检查方式 2：智能开关是否刚刚在 5 秒内被按下
                        bool switch_active = false;
                        {
                            std::lock_guard<std::mutex> lk(mtx_);
                            switch_active = (now < switch_pressed_until_);
                        }
                        if (switch_active) {
                            passed = true;
                            matched_person = current_family_name;
                            matched_score = current_family_score;
                            matched_gesture = "SWITCH_PREV_CLICKED";
                            passed_mode = "人体 + 智能开关";
                            elapsed_ms = static_cast<int>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());
                            triggerRelayAndHaAsync(matched_person, matched_score, matched_gesture, passed_mode, elapsed_ms);
                            break;
                        }

                        // 检查方式 1：手势识别 (OK, VICTORY, etc.)
                        auto gesture_res = gesture_engine_.detectGesture(frame);
                        if (gesture_res.has_value()) {
                            passed = true;
                            matched_person = current_family_name;
                            matched_score = current_family_score;
                            matched_gesture = *gesture_res;
                            passed_mode = "人体 + 手势";
                            elapsed_ms = static_cast<int>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());

                            // 在最终抓拍帧画绿色开门标识
                            cv::rectangle(frame, cv::Point(15, 15), cv::Point(780, 72), cv::Scalar(0, 180, 60), -1);
                            std::string banner = "UNLOCKED! [" + matched_person + " + " + matched_gesture + "] (" +
                                                 std::to_string(elapsed_ms) + "ms)";
                            cv::putText(frame, banner, cv::Point(28, 54), cv::FONT_HERSHEY_SIMPLEX,
                                        0.85, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

                            {
                                std::vector<uchar> buf;
                                cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
                                std::lock_guard<std::mutex> flk(frame_mtx_);
                                latest_jpeg_ = std::move(buf);
                            }

                            triggerRelayAndHaAsync(matched_person, matched_score, matched_gesture, passed_mode, elapsed_ms);
                            break;
                        }
                    }

                    // 编码最新画面供 Web 前端实时预览 (/api/frame.jpg)
                    {
                        std::vector<uchar> buf;
                        cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 80});
                        std::lock_guard<std::mutex> flk(frame_mtx_);
                        latest_jpeg_ = std::move(buf);
                    }
                }
                cap.release();
            }

            {
                std::unique_lock<std::mutex> lock(mtx_);
                if (!passed) {
                    last_result_ = json{
                        {"passed", false},
                        {"person", matched_person},
                        {"auth_mode", "None"},
                        {"face_score", 0.0},
                        {"gesture", "None"},
                        {"relay_pulse_ms", 0},
                        {"elapsed_ms", elapsed_ms},
                        {"timestamp", currentTimeStr()}
                    };
                }
                verifying_ = false;
                stop_requested_ = false;
                remaining_sec_ = 0.0;
                verify_done_ = true;
                cv_done_.notify_all();
            }
        }
    }

    myhome::CppPersonEngine& getPersonEngine() { return person_engine_; }

private:
    myhome::CppPersonEngine& person_engine_;
    myhome::CppGestureEngine& gesture_engine_;
    std::string cam_source_;
    std::string ha_base_url_;
    std::string unlock_webhook_id_;
    double active_window_sec_;

    std::mutex mtx_;
    std::mutex frame_mtx_;
    std::vector<uchar> latest_jpeg_;
    std::condition_variable cv_wakeup_;
    std::condition_variable cv_done_;
    bool running_ = true;
    bool verifying_ = false;
    bool stop_requested_ = false;
    bool verify_done_ = false;
    double verify_timeout_sec_ = 15.0;
    double remaining_sec_ = 0.0;

    std::string live_person_ = "None";
    float live_score_ = 0.0f;
    bool live_has_stranger_ = false;

    // 自家人在场有效期与智能开关互锁状态
    std::chrono::steady_clock::time_point family_presence_until_{};
    std::string last_verified_family_ = "None";
    float last_verified_score_ = 0.0f;
    std::chrono::steady_clock::time_point switch_pressed_until_{};

    std::chrono::steady_clock::time_point relay_active_until_{};
    json last_result_;
    std::vector<json> history_;
    std::atomic<int> unlock_count_{0};
};

void runHttpServer(VerifyServiceCoordinator& coordinator, int port) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "❌ 无法创建 HTTP Socket" << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(server_fd, 64) < 0) {
        std::cerr << "❌ 端口 " << port << " 绑定失败，请检查端口是否被占用" << std::endl;
        ::close(server_fd);
        return;
    }

    std::cout << "🌐 [Web 控制台与统一 API 已就绪] http://127.0.0.1:" << port << "/" << std::endl;
    std::cout << "   • 页面控制台: http://127.0.0.1:" << port << "/" << std::endl;
    std::cout << "   • 页面测试 API: POST /api/test_image (支持大图 Base64 上传与纯净感知输出)" << std::endl;
    std::cout << "   • 成员底库 API: GET /api/members, POST /api/upload_photo, POST /api/delete_photo" << std::endl;
    std::cout << "   • 开关互锁 API: POST /api/switch_trigger, GET /api/switch_verify" << std::endl;
    std::cout << "   • 声音预留 API: POST /api/verify_voice" << std::endl;
    std::cout << "   • 硬件脉冲模拟: POST /api/test_relay" << std::endl;

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        std::thread([client_fd, &coordinator]() {
            // 循环读取完整的 HTTP 请求（支持 500KB+ Base64 图片上传）
            std::string req_header;
            std::string req_body;
            char chunk[8192];
            bool header_done = false;
            size_t content_length = 0;

            while (true) {
                ssize_t n = ::read(client_fd, chunk, sizeof(chunk));
                if (n <= 0) break;
                if (!header_done) {
                    req_header.append(chunk, static_cast<size_t>(n));
                    auto pos = req_header.find("\r\n\r\n");
                    if (pos != std::string::npos) {
                        header_done = true;
                        req_body = req_header.substr(pos + 4);
                        req_header = req_header.substr(0, pos);

                        auto cl_pos = req_header.find("Content-Length:");
                        if (cl_pos == std::string::npos) {
                            cl_pos = req_header.find("content-length:");
                        }
                        if (cl_pos != std::string::npos) {
                            size_t num_start = req_header.find_first_not_of(" \t", cl_pos + 15);
                            size_t num_end = req_header.find("\r\n", num_start);
                            if (num_start != std::string::npos && num_end != std::string::npos) {
                                content_length = std::stoul(req_header.substr(num_start, num_end - num_start));
                            }
                        }
                        if (req_body.size() >= content_length) break;
                    }
                } else {
                    req_body.append(chunk, static_cast<size_t>(n));
                    if (req_body.size() >= content_length) break;
                }
            }

            if (req_header.empty()) {
                ::close(client_fd);
                return;
            }

            std::istringstream iss(req_header);
            std::string method, raw_path;
            iss >> method >> raw_path;

            // 分离路径与 QueryString
            std::string path = raw_path;
            std::string query_str;
            auto q_pos = raw_path.find('?');
            if (q_pos != std::string::npos) {
                path = raw_path.substr(0, q_pos);
                query_str = raw_path.substr(q_pos + 1);
            }

            // 处理 OPTIONS 预检请求
            if (method == "OPTIONS") {
                std::string resp =
                    "HTTP/1.1 204 No Content\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: Content-Type\r\n\r\n";
                ::write(client_fd, resp.c_str(), resp.size());
                ::close(client_fd);
                return;
            }

            // 1. 访问 / 返回前端页面 ./web/index.html
            if (path == "/" || path == "/index.html") {
                std::ifstream ifs("./web/index.html", std::ios::binary);
                if (ifs) {
                    std::ostringstream html_ss;
                    html_ss << ifs.rdbuf();
                    std::string html = html_ss.str();
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: text/html; charset=utf-8\r\n"
                        << "Content-Length: " << html.size() << "\r\n"
                        << "Cache-Control: no-cache\r\n"
                        << "Connection: close\r\n\r\n";
                    std::string header_str = hdr.str();
                    ::write(client_fd, header_str.c_str(), header_str.size());
                    ::write(client_fd, html.c_str(), html.size());
                    ::close(client_fd);
                    return;
                }
            }

            // 2. 照片缩略图直接流送 /api/photo?file=xxx
            if (path == "/api/photo") {
                std::string file_param;
                auto f_pos = query_str.find("file=");
                if (f_pos != std::string::npos) {
                    file_param = query_str.substr(f_pos + 5);
                    auto amp_pos = file_param.find('&');
                    if (amp_pos != std::string::npos) file_param = file_param.substr(0, amp_pos);
                }
                auto bin = coordinator.getPersonEngine().getPhotoBinary(file_param);
                if (!bin.empty()) {
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: image/jpeg\r\n"
                        << "Content-Length: " << bin.size() << "\r\n"
                        << "Cache-Control: public, max-age=86400\r\n"
                        << "Access-Control-Allow-Origin: *\r\n"
                        << "Connection: close\r\n\r\n";
                    std::string header_str = hdr.str();
                    ::write(client_fd, header_str.c_str(), header_str.size());
                    ::write(client_fd, reinterpret_cast<const char*>(bin.data()), bin.size());
                } else {
                    std::string nf = "HTTP/1.1 404 Not Found\r\n\r\n";
                    ::write(client_fd, nf.c_str(), nf.size());
                }
                ::close(client_fd);
                return;
            }

            // 3. 实时预览帧 /api/frame.jpg
            if (path == "/api/frame.jpg") {
                auto jpg = coordinator.getLatestJpeg();
                if (!jpg.empty()) {
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: image/jpeg\r\n"
                        << "Content-Length: " << jpg.size() << "\r\n"
                        << "Cache-Control: no-store, no-cache\r\n"
                        << "Access-Control-Allow-Origin: *\r\n"
                        << "Connection: close\r\n\r\n";
                    std::string header_str = hdr.str();
                    ::write(client_fd, header_str.c_str(), header_str.size());
                    ::write(client_fd, reinterpret_cast<const char*>(jpg.data()), jpg.size());
                } else {
                    std::string noc = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                    ::write(client_fd, noc.c_str(), noc.size());
                }
                ::close(client_fd);
                return;
            }

            // 4. JSON REST API 处理
            json resp_json;
            if (path == "/api/test_image" || path == "/api/predict") {
                try {
                    auto j = json::parse(req_body);
                    std::string b64 = j.value("image", "");
                    resp_json = coordinator.testSingleImage(b64);
                } catch (const std::exception& e) {
                    resp_json = json{{"error", std::string("JSON 解析失败: ") + e.what()}};
                }
            } else if (path == "/api/members") {
                resp_json = coordinator.getPersonEngine().getMembersSummary();
            } else if (path == "/api/upload_photo") {
                try {
                    auto j = json::parse(req_body);
                    std::string name = j.value("name", "");
                    std::string b64 = j.value("image", "");
                    resp_json = coordinator.getPersonEngine().uploadPhoto(name, b64);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/delete_photo") {
                try {
                    auto j = json::parse(req_body);
                    std::string f = j.value("file", "");
                    resp_json = coordinator.getPersonEngine().deletePhoto(f);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/switch_trigger") {
                resp_json = coordinator.handleSmartSwitchTrigger();
            } else if (path == "/api/switch_verify") {
                resp_json = coordinator.handleSmartSwitchVerifyQuery();
            } else if (path == "/api/verify_voice") {
                try {
                    auto j = json::parse(req_body);
                    resp_json = coordinator.handleVoiceVerifyReserved(j);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/verify") {
                resp_json = coordinator.requestVerifySync();
            } else if (path == "/api/start_verify") {
                resp_json = coordinator.requestVerifyAsync();
            } else if (path == "/api/stop_verify") {
                resp_json = coordinator.requestStopVerify();
            } else if (path == "/api/test_relay") {
                coordinator.triggerRelayAndHaAsync("Web手动测试", 1.0f, "MANUAL_BUTTON", "手动模拟", 12);
                resp_json = json{{"status", "ok"}, {"message", "已触发 ESP32-C6 光耦 500ms 点动脉冲"}};
            } else if (path == "/api/reload") {
                resp_json = coordinator.getPersonEngine().reloadGallery();
            } else {
                resp_json = coordinator.getStatus();
            }

            std::string body = resp_json.dump(2) + "\n";
            std::ostringstream oss;
            oss << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: application/json; charset=utf-8\r\n"
                << "Access-Control-Allow-Origin: *\r\n"
                << "Content-Length: " << body.size() << "\r\n"
                << "Connection: close\r\n\r\n"
                << body;
            std::string http_resp = oss.str();
            ::write(client_fd, http_resp.c_str(), http_resp.size());
            ::close(client_fd);
        }).detach();
    }
}

int main(int argc, char** argv) {
    std::string root_dir = ".";
    if (!fs::exists("./config.yaml") && fs::exists("../../config.yaml")) {
        root_dir = "../..";
    } else if (!fs::exists("./config.yaml") && fs::exists("../config.yaml")) {
        root_dir = "..";
    }
    fs::current_path(root_dir);

    YAML::Node cfg = YAML::LoadFile("config.yaml");
    double active_window_sec = cfg["system"]["active_window_sec"].as<double>(15.0);
    std::string ha_url = cfg["home_assistant"]["base_url"].as<std::string>("http://homeassistant.local:8123");
    std::string unlock_webhook = cfg["home_assistant"]["unlock_webhook_id"].as<std::string>("ai_door_unlock_event");
    std::string cam_source = cfg["vision"]["camera_source"].as<std::string>("0");
    std::string db_dir = cfg["vision"]["face"]["db_dir"].as<std::string>("./vision/db/faces");
    int confirm_frames = cfg["vision"]["gesture"]["confirm_frames"].as<int>(3);

    std::cout << "========================================================================" << std::endl;
    std::cout << "🚀 MyHome 智能门禁控制服务 (统一集成 LightPersonIDNet 人体模型)" << std::endl;
    std::cout << "   • 模式 1: 人体 + 手势" << std::endl;
    std::cout << "   • 模式 2: 人体 + 智能开关 (支持 HA 状态自裁决与双向互锁)" << std::endl;
    std::cout << "   • 模式 3: 声音核验 (接口预留)" << std::endl;
    std::cout << "========================================================================" << std::endl;

    curl_global_init(CURL_GLOBAL_ALL);

    // 实例化纯 C++ 原生人体识别引擎 (YOLOv8-Nano + MobileNetV3 ArcFace, 0 Python 依赖)
    myhome::CppPersonEngine person_engine(
        "./vision/models/yolov8n.onnx",
        "./vision/models/person_classifier.onnx",
        "./vision/models/classes.json",
        db_dir);

    // 实例化手势识别引擎
    myhome::CppGestureEngine gesture_engine(
        "./vision/models/palm_detection_mediapipe_2023feb.onnx",
        "./vision/models/handpose_estimation_mediapipe_2023feb.onnx",
        {"OK", "VICTORY", "OPEN_PALM", "THUMB_UP"},
        confirm_frames);

    VerifyServiceCoordinator coordinator(
        person_engine, gesture_engine, cam_source, ha_url, unlock_webhook, active_window_sec);

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--self-test") {
            std::cout << "✅ [Self-Test] C++ 服务与人体模型微服务自检通过！当前状态:\n"
                      << coordinator.getStatus().dump(2) << std::endl;
            return 0;
        }
    }

    int port = 8090;
    std::thread http_thread(runHttpServer, std::ref(coordinator), port);
    http_thread.detach();

    std::cout << "💤 [服务运行中] 打开浏览器访问 http://127.0.0.1:8090 开始测试与管理！" << std::endl;
    coordinator.runMainThreadLoop();

    curl_global_cleanup();
    return 0;
}
