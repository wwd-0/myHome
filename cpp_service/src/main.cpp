#include "face_engine.hpp"
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
    VerifyServiceCoordinator(myhome::CppFaceEngine& face_eng,
                             myhome::CppGestureEngine& gesture_eng,
                             const std::string& cam_source,
                             const std::string& ha_url,
                             const std::string& unlock_webhook,
                             double window_sec)
        : face_engine_(face_eng),
          gesture_engine_(gesture_eng),
          cam_source_(cam_source),
          ha_base_url_(ha_url),
          unlock_webhook_id_(unlock_webhook),
          active_window_sec_(window_sec) {
        last_result_ = json{
            {"passed", false},
            {"person", "None"},
            {"face_score", 0.0},
            {"gesture", "None"},
            {"relay_pulse_ms", 0},
            {"elapsed_ms", 0},
            {"timestamp", currentTimeStr()}
        };
    }

    // 模拟或真实触发 ESP32-C6 + 光耦 500ms 脉冲与 Home Assistant Webhook
    void triggerRelayAndHaAsync(const std::string& person,
                                float face_score,
                                const std::string& gesture,
                                int elapsed_ms) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            relay_active_until_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(600);
            json event_item = {
                {"time", currentTimeStr()},
                {"person", person},
                {"face_score", std::round(face_score * 1000.0f) / 1000.0f},
                {"gesture", gesture},
                {"elapsed_ms", elapsed_ms},
                {"relay_pulse_ms", 500}
            };
            history_.insert(history_.begin(), event_item);
            if (history_.size() > 15) history_.pop_back();
        }

        std::string ha_url = ha_base_url_;
        std::string webhook = unlock_webhook_id_;
        std::thread([=]() {
            std::cout << "\n====================================================================" << std::endl;
            std::cout << "⚡ [ESP32-C6 + 光耦继电器] 收到开门指令 (人物: " << person << ", 动作: " << gesture << ")" << std::endl;
            std::cout << "   -> [0 ms]   GPIO 拉高：光耦导通 (ON)" << std::endl;

            std::string url = ha_url + "/api/webhook/" + webhook;
            json payload = {
                {"event", "unlock_door"},
                {"person", person},
                {"auth_mode", "人脸 + 手势"},
                {"face_score", std::round(face_score * 1000.0f) / 1000.0f},
                {"gesture", gesture},
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
            return json{{"status", "already_verifying"}, {"message", "视频流已在预热/核验中"}};
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

    json reloadDb() {
        std::unique_lock<std::mutex> lock(mtx_);
        int count = face_engine_.reloadFamilyDb();
        return json{
            {"status", "ok"},
            {"enrolled_count", count},
            {"members", face_engine_.getEnrolledNames()}
        };
    }

    json getStatus() {
        std::unique_lock<std::mutex> lock(mtx_);
        bool relay_on = std::chrono::steady_clock::now() < relay_active_until_;
        return json{
            {"service", "myhome_cpp_verify_server"},
            {"state", verifying_ ? "VERIFYING_STREAM" : "IDLE_SLEEP"},
            {"remaining_sec", verifying_ ? std::round(remaining_sec_ * 10.0) / 10.0 : 0.0},
            {"window_sec", active_window_sec_},
            {"enrolled_members", face_engine_.getEnrolledNames()},
            {"live_person", live_person_},
            {"live_face_score", std::round(live_face_score_ * 1000.0f) / 1000.0f},
            {"live_has_stranger", live_has_stranger_},
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
                live_face_score_ = 0.0f;
                live_has_stranger_ = false;
                remaining_sec_ = 0.0;
                cv_wakeup_.wait(lock, [this]() { return verifying_ || !running_; });
                if (!running_) break;
                timeout_sec = verify_timeout_sec_;
            }

            std::cout << "\n📡 [C++ 核验服务] 收到唤醒请求！正在打开摄像头拉流 (窗口期: "
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
                            std::cout << "⏹️ [C++ 核验服务] 收到手动停止指令，释放摄像头回到休眠态。" << std::endl;
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

                    bool has_stranger = false;
                    auto face_res = face_engine_.analyzeFrame(frame, has_stranger);

                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        live_has_stranger_ = has_stranger;
                        if (face_res.has_value()) {
                            live_person_ = face_res->person_name;
                            live_face_score_ = face_res->score;
                        } else {
                            live_person_ = has_stranger ? "Stranger" : "None";
                            live_face_score_ = 0.0f;
                        }
                    }

                    if (face_res.has_value()) {
                        auto gesture_res = gesture_engine_.detectGesture(frame);
                        if (gesture_res.has_value()) {
                            passed = true;
                            matched_person = face_res->person_name;
                            matched_score = face_res->score;
                            matched_gesture = *gesture_res;
                            elapsed_ms = static_cast<int>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - t_start)
                                    .count());

                            // 在最终抓拍帧上画上绿色开门横幅，方便前端页面定格展示最后开门画面
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

                            unlock_count_++;
                            triggerRelayAndHaAsync(matched_person, matched_score, matched_gesture, elapsed_ms);
                            break;
                        }
                    }

                    std::string hud = "ACTIVE STREAM | Window: " + cv::format("%.1fs", rem);
                    cv::putText(frame, hud, cv::Point(20, 38), cv::FONT_HERSHEY_SIMPLEX,
                                0.75, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

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
                last_result_ = json{
                    {"passed", passed},
                    {"person", matched_person},
                    {"face_score", std::round(matched_score * 1000.0f) / 1000.0f},
                    {"gesture", matched_gesture},
                    {"relay_pulse_ms", passed ? 500 : 0},
                    {"elapsed_ms", elapsed_ms},
                    {"timestamp", currentTimeStr()}
                };
                verifying_ = false;
                stop_requested_ = false;
                remaining_sec_ = 0.0;
                verify_done_ = true;
                cv_done_.notify_all();
            }
        }
    }

private:
    myhome::CppFaceEngine& face_engine_;
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
    float live_face_score_ = 0.0f;
    bool live_has_stranger_ = false;
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
        ::listen(server_fd, 32) < 0) {
        std::cerr << "❌ 端口 " << port << " 绑定失败，请检查端口是否被占用" << std::endl;
        ::close(server_fd);
        return;
    }

    std::cout << "🌐 [Web 测试控制台已就绪] 浏览器打开: http://127.0.0.1:" << port << "/" << std::endl;
    std::cout << "   - [同步核验] POST http://127.0.0.1:" << port << "/api/verify" << std::endl;
    std::cout << "   - [异步唤醒] POST http://127.0.0.1:" << port << "/api/start_verify" << std::endl;
    std::cout << "   - [停止拉流] POST http://127.0.0.1:" << port << "/api/stop_verify" << std::endl;
    std::cout << "   - [模拟光耦] POST http://127.0.0.1:" << port << "/api/test_relay" << std::endl;
    std::cout << "   - [重载底库] POST http://127.0.0.1:" << port << "/api/reload\n" << std::endl;

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        std::thread([client_fd, &coordinator]() {
            char buffer[4096] = {0};
            ssize_t n = ::read(client_fd, buffer, sizeof(buffer) - 1);
            if (n <= 0) {
                ::close(client_fd);
                return;
            }

            std::string req(buffer, static_cast<size_t>(n));
            std::istringstream iss(req);
            std::string method, path;
            iss >> method >> path;

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

            // 1. 访问根路径 / 时直接返回前端测试页面 ./web/index.html
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

            // 2. 获取实时编码画面 /api/frame.jpg
            if (path.rfind("/api/frame.jpg", 0) == 0) {
                auto jpg = coordinator.getLatestJpeg();
                if (!jpg.empty()) {
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: image/jpeg\r\n"
                        << "Content-Length: " << jpg.size() << "\r\n"
                        << "Cache-Control: no-store, no-cache, must-revalidate\r\n"
                        << "Access-Control-Allow-Origin: *\r\n"
                        << "Connection: close\r\n\r\n";
                    std::string header_str = hdr.str();
                    ::write(client_fd, header_str.c_str(), header_str.size());
                    ::write(client_fd, reinterpret_cast<const char*>(jpg.data()), jpg.size());
                } else {
                    std::string not_found = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                    ::write(client_fd, not_found.c_str(), not_found.size());
                }
                ::close(client_fd);
                return;
            }

            // 3. JSON API 路由
            json resp_json;
            if (path.rfind("/api/verify", 0) == 0) {
                resp_json = coordinator.requestVerifySync();
            } else if (path.rfind("/api/start_verify", 0) == 0) {
                resp_json = coordinator.requestVerifyAsync();
            } else if (path.rfind("/api/stop_verify", 0) == 0) {
                resp_json = coordinator.requestStopVerify();
            } else if (path.rfind("/api/test_relay", 0) == 0) {
                coordinator.triggerRelayAndHaAsync("手动测试(Web)", 1.0f, "MANUAL_BUTTON", 12);
                resp_json = json{{"status", "ok"}, {"message", "已触发 ESP32-C6 光耦 500ms 点动脉冲模拟"}};
            } else if (path.rfind("/api/reload", 0) == 0) {
                resp_json = coordinator.reloadDb();
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
    std::string det_model = cfg["vision"]["face"]["det_model_path"].as<std::string>(
        "./vision/models/face_detection_yunet_2023mar.onnx");
    std::string rec_model = cfg["vision"]["face"]["rec_model_path"].as<std::string>(
        "./vision/models/face_recognition_sface_2021dec.onnx");
    float face_thresh = cfg["vision"]["face"]["similarity_threshold"].as<float>(0.38f);
    int min_face_w = cfg["vision"]["face"]["min_face_width_px"].as<int>(90);
    int confirm_frames = cfg["vision"]["gesture"]["confirm_frames"].as<int>(3);

    std::cout << "========================================================================" << std::endl;
    std::cout << "🚀 MyHome C++17 高性能按需拉流与【人脸+手势】核验服务启动中..." << std::endl;
    std::cout << "========================================================================" << std::endl;

    curl_global_init(CURL_GLOBAL_ALL);

    myhome::CppFaceEngine face_engine(db_dir, det_model, rec_model, face_thresh, min_face_w);
    myhome::CppGestureEngine gesture_engine(
        "./vision/models/palm_detection_mediapipe_2023feb.onnx",
        "./vision/models/handpose_estimation_mediapipe_2023feb.onnx",
        {"OK", "VICTORY", "OPEN_PALM", "THUMB_UP"},
        confirm_frames);

    VerifyServiceCoordinator coordinator(
        face_engine, gesture_engine, cam_source, ha_url, unlock_webhook, active_window_sec);

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--self-test") {
            std::cout << "✅ [Self-Test] C++17 服务初始化自检通过！当前状态:\n"
                      << coordinator.getStatus().dump(2) << std::endl;
            return 0;
        }
    }

    int port = 8090;
    std::thread http_thread(runHttpServer, std::ref(coordinator), port);
    http_thread.detach();

    std::cout << "💤 [进入 0% CPU 深度待机] 浏览器打开 http://127.0.0.1:8090 即可可视化操控测试！" << std::endl;
    coordinator.runMainThreadLoop();

    curl_global_cleanup();
    return 0;
}
