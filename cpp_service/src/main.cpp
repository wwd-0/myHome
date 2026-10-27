#include "coordinator.hpp"
#include "policy.hpp"
#include "actuator.hpp"
#include "person_engine.hpp"
#include "gesture_engine.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <csignal>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void safeSocketWrite(int fd, const std::string& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        ssize_t sent = ::write(fd, data.data() + total_sent, data.size() - total_sent);
        if (sent <= 0) break;
        total_sent += static_cast<size_t>(sent);
    }
}

void safeSocketWriteBinary(int fd, const std::vector<uchar>& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        ssize_t sent = ::write(fd, data.data() + total_sent, data.size() - total_sent);
        if (sent <= 0) break;
        total_sent += static_cast<size_t>(sent);
    }
}

} // namespace

void runHttpServer(std::shared_ptr<myhome::VerifyServiceCoordinator> coordinator, int port) {
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

        std::thread([client_fd, coordinator]() {
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
                                try {
                                    content_length = std::stoul(req_header.substr(num_start, num_end - num_start));
                                } catch (...) {
                                    content_length = 0;
                                }
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
                safeSocketWrite(client_fd, resp);
                ::close(client_fd);
                return;
            }

            // 1. 静态主页 /
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
                    safeSocketWrite(client_fd, hdr.str());
                    safeSocketWrite(client_fd, html);
                    ::close(client_fd);
                    return;
                }
            }

            // 2. 照片缩略图流送 /api/photo?file=xxx (遵循 LoD)
            if (path == "/api/photo") {
                std::string file_param;
                auto f_pos = query_str.find("file=");
                if (f_pos != std::string::npos) {
                    file_param = query_str.substr(f_pos + 5);
                    auto amp_pos = file_param.find('&');
                    if (amp_pos != std::string::npos) file_param = file_param.substr(0, amp_pos);
                }
                auto bin = coordinator->getPhotoBinary(file_param);
                if (!bin.empty()) {
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: image/jpeg\r\n"
                        << "Content-Length: " << bin.size() << "\r\n"
                        << "Cache-Control: public, max-age=86400\r\n"
                        << "Access-Control-Allow-Origin: *\r\n"
                        << "Connection: close\r\n\r\n";
                    safeSocketWrite(client_fd, hdr.str());
                    safeSocketWriteBinary(client_fd, bin);
                } else {
                    std::string nf = "HTTP/1.1 404 Not Found\r\n\r\n";
                    safeSocketWrite(client_fd, nf);
                }
                ::close(client_fd);
                return;
            }

            // 3. 实时预览帧 /api/frame.jpg
            if (path == "/api/frame.jpg") {
                auto jpg = coordinator->getLatestJpeg();
                if (!jpg.empty()) {
                    std::ostringstream hdr;
                    hdr << "HTTP/1.1 200 OK\r\n"
                        << "Content-Type: image/jpeg\r\n"
                        << "Content-Length: " << jpg.size() << "\r\n"
                        << "Cache-Control: no-store, no-cache\r\n"
                        << "Access-Control-Allow-Origin: *\r\n"
                        << "Connection: close\r\n\r\n";
                    safeSocketWrite(client_fd, hdr.str());
                    safeSocketWriteBinary(client_fd, jpg);
                } else {
                    std::string noc = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                    safeSocketWrite(client_fd, noc);
                }
                ::close(client_fd);
                return;
            }

            // 4. REST JSON API 路由 (通过 Coordinator 门面进行业务交互 - 遵循 LoD)
            json resp_json;
            if (path == "/api/test_image" || path == "/api/predict") {
                try {
                    auto j = json::parse(req_body);
                    std::string b64 = j.value("image", "");
                    resp_json = coordinator->testSingleImage(b64);
                } catch (const std::exception& e) {
                    resp_json = json{{"error", std::string("JSON 解析失败: ") + e.what()}};
                }
            } else if (path == "/api/members") {
                resp_json = coordinator->getMembersSummary();
            } else if (path == "/api/upload_photo") {
                try {
                    auto j = json::parse(req_body);
                    std::string name = j.value("name", "");
                    std::string b64 = j.value("image", "");
                    resp_json = coordinator->uploadPhoto(name, b64);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/delete_photo") {
                try {
                    auto j = json::parse(req_body);
                    std::string f = j.value("file", "");
                    resp_json = coordinator->deletePhoto(f);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/switch_trigger") {
                resp_json = coordinator->handleSmartSwitchTrigger();
            } else if (path == "/api/switch_verify") {
                resp_json = coordinator->handleSmartSwitchVerifyQuery();
            } else if (path == "/api/verify_voice") {
                try {
                    auto j = json::parse(req_body);
                    resp_json = coordinator->handleVoiceVerify(j);
                } catch (...) {
                    resp_json = json{{"error", "Invalid payload"}};
                }
            } else if (path == "/api/verify") {
                resp_json = coordinator->requestVerifySync();
            } else if (path == "/api/start_verify") {
                resp_json = coordinator->requestVerifyAsync();
            } else if (path == "/api/stop_verify") {
                resp_json = coordinator->requestStopVerify();
            } else if (path == "/api/test_relay") {
                resp_json = coordinator->triggerManualRelayTest();
            } else if (path == "/api/reload") {
                resp_json = coordinator->reloadGallery();
            } else {
                resp_json = coordinator->getStatus();
            }

            std::string body = resp_json.dump(2) + "\n";
            std::ostringstream oss;
            oss << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: application/json; charset=utf-8\r\n"
                << "Access-Control-Allow-Origin: *\r\n"
                << "Content-Length: " << body.size() << "\r\n"
                << "Connection: close\r\n\r\n"
                << body;
            safeSocketWrite(client_fd, oss.str());
            ::close(client_fd);
        }).detach();
    }
}

int main(int argc, char** argv) {
    // 忽略 SIGPIPE，防止客户端异常断开导致整个服务崩溃
    signal(SIGPIPE, SIG_IGN);

    std::string root_dir = ".";
    if (!fs::exists("./config.yaml") && fs::exists("../../config.yaml")) {
        root_dir = "../..";
    } else if (!fs::exists("./config.yaml") && fs::exists("../config.yaml")) {
        root_dir = "..";
    }
    fs::current_path(root_dir);

    // 1. 读取全局配置
    YAML::Node cfg = YAML::LoadFile("config.yaml");
    double active_window_sec = cfg["system"]["active_window_sec"].as<double>(15.0);
    double switch_face_window_sec = 5.0;
    if (cfg["system"]["switch_face_window_sec"]) {
        switch_face_window_sec = cfg["system"]["switch_face_window_sec"].as<double>(5.0);
    }
    double unlock_cooldown_sec = cfg["system"]["unlock_cooldown_sec"].as<double>(10.0);

    std::string ha_url = cfg["home_assistant"]["base_url"].as<std::string>("http://homeassistant.local:8123");
    std::string unlock_webhook = cfg["home_assistant"]["unlock_webhook_id"].as<std::string>("ai_door_unlock_event");
    std::string cam_source = cfg["vision"]["camera_source"].as<std::string>("0");
    std::string db_dir = cfg["vision"]["face"]["db_dir"].as<std::string>("./vision/db/faces");
    int confirm_frames = cfg["vision"]["gesture"]["confirm_frames"].as<int>(3);

    std::vector<std::string> allowed_gestures = {"OK", "VICTORY", "OPEN_PALM", "THUMB_UP"};
    if (cfg["vision"]["gesture"]["unlock_gestures"]) {
        allowed_gestures.clear();
        for (const auto& g : cfg["vision"]["gesture"]["unlock_gestures"]) {
            allowed_gestures.push_back(g.as<std::string>());
        }
    }

    std::cout << "========================================================================" << std::endl;
    std::cout << "🚀 MyHome 智能门禁控制服务 (遵循六大设计原则架构重构版)" << std::endl;
    std::cout << "   • 模式 1: 人体 + 手势 (允许手势数: " << allowed_gestures.size() << ")" << std::endl;
    std::cout << "   • 模式 2: 人体 + 智能开关 (C++ 统一控制开门，前后 " << switch_face_window_sec << "s 互锁判定)" << std::endl;
    std::cout << "   • 模式 3: 声音核验 (接口预留)" << std::endl;
    std::cout << "   • 防抖/冷却: 确认帧数=" << confirm_frames << ", 开门冷却=" << unlock_cooldown_sec << "s" << std::endl;
    std::cout << "========================================================================" << std::endl;

    curl_global_init(CURL_GLOBAL_ALL);

    // 2. 依赖注入与组件装配 (遵循 DIP & OCP)
    auto person_engine = std::make_shared<myhome::CppPersonEngine>(
        "./vision/models/yolov8n.onnx",
        "./vision/models/person_classifier.onnx",
        "./vision/models/classes.json",
        db_dir);

    auto gesture_engine = std::make_shared<myhome::CppGestureEngine>(
        "./vision/models/palm_detection_mediapipe_2023feb.onnx",
        "./vision/models/handpose_estimation_mediapipe_2023feb.onnx",
        allowed_gestures,
        confirm_frames);

    // 装配开门判定策略组合器 (Strategy Pattern & Composite Pattern - 遵循 OCP)
    auto composite_policy = std::make_shared<myhome::CompositePolicy>();
    composite_policy->addPolicy(std::make_shared<myhome::SwitchWindowPolicy>(switch_face_window_sec));
    composite_policy->addPolicy(std::make_shared<myhome::GesturePolicy>(allowed_gestures));
    composite_policy->addPolicy(std::make_shared<myhome::VoicePolicy>());

    // 装配执行器与冷却装饰器 (Decorator Pattern - 遵循 OCP & SRP)
    auto raw_actuator = std::make_shared<myhome::HaWebhookAndRelayActuator>(ha_url, unlock_webhook, 500);
    auto actuator = std::make_shared<myhome::CooldownActuatorDecorator>(raw_actuator, unlock_cooldown_sec);

    // 装配核心协调器 (Facade Pattern - 遵循 LoD, SRP, DIP)
    auto coordinator = std::make_shared<myhome::VerifyServiceCoordinator>(
        person_engine, gesture_engine, composite_policy, actuator, cam_source, active_window_sec);

    // 自检模式
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--self-test") {
            std::cout << "✅ [Self-Test] C++ 服务重构自检通过！当前状态:\n"
                      << coordinator->getStatus().dump(2) << std::endl;
            return 0;
        }
    }

    int port = 8090;
    std::thread http_thread(runHttpServer, coordinator, port);
    http_thread.detach();

    std::cout << "💤 [服务运行中] 打开浏览器访问 http://127.0.0.1:8090 开始测试与管理！" << std::endl;
    coordinator->runMainThreadLoop();

    curl_global_cleanup();
    return 0;
}
