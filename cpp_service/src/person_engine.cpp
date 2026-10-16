#include "person_engine.hpp"

#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    auto* s = static_cast<std::string*>(userp);
    s->append(static_cast<char*>(contents), total);
    return total;
}

size_t curlWriteBinaryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    auto* vec = static_cast<std::vector<uchar>*>(userp);
    const auto* p = static_cast<const uchar*>(contents);
    vec->insert(vec->end(), p, p + total);
    return total;
}

} // namespace

namespace myhome {

CppPersonEngine::CppPersonEngine(const std::string& service_url, const std::string& db_dir)
    : service_url_(service_url), db_dir_(db_dir) {
    ensureServiceRunning();
}

CppPersonEngine::~CppPersonEngine() = default;

std::string CppPersonEngine::httpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string response;
    if (!curl) return response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 3000L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code < 200 || http_code >= 300) {
        return "";
    }
    return response;
}

std::string CppPersonEngine::httpPost(const std::string& url, const std::string& body, const std::string& content_type) {
    CURL* curl = curl_easy_init();
    std::string response;
    if (!curl) return response;

    struct curl_slist* headers = nullptr;
    std::string ct_header = "Content-Type: " + content_type;
    headers = curl_slist_append(headers, ct_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 15000L); // 推理/重载允许较长超时

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return "";
    }
    return response;
}

bool CppPersonEngine::isHealthy() {
    std::string resp = httpGet(service_url_ + "/health");
    if (resp.empty()) return false;
    try {
        auto j = json::parse(resp);
        return j.value("status", "") == "healthy";
    } catch (...) {
        return false;
    }
}

bool CppPersonEngine::ensureServiceRunning() {
    if (isHealthy()) {
        std::cout << "✅ [PersonEngine] 检测到 Python 人体推理微服务在线 (" << service_url_ << ")" << std::endl;
        return true;
    }

    std::cout << "⚡ [PersonEngine] Python 人体微服务未运行，正在自动启动..." << std::endl;

    // 尝试寻找可用的 Python 解释器
    std::string python_bin = "/Users/a58/miniforge3/envs/comfyui/bin/python";
    if (!fs::exists(python_bin)) {
        python_bin = "python3";
    }

    std::string script_path = "door_person_detector/service.py";
    if (!fs::exists(script_path)) {
        script_path = "../door_person_detector/service.py";
    }

    std::string cmd = python_bin + " " + script_path + " --port 8092 > /dev/null 2>&1 &";
    std::system(cmd.c_str());

    // 等待服务就绪（最多 4 秒）
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (isHealthy()) {
            std::cout << "🔥 [PersonEngine] Python 人体推理微服务就绪！(端口 8092, MPS 加速)" << std::endl;
            return true;
        }
    }

    std::cerr << "⚠️ [PersonEngine] 警告: 未能在 4s 内连接到 Python 人体微服务，后续请求可能受影响" << std::endl;
    return false;
}

json CppPersonEngine::getMembersSummary() {
    std::string resp = httpGet(service_url_ + "/members");
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"total_members", 0}, {"max_members", 15}, {"members", json::array()}};
}

std::vector<uchar> CppPersonEngine::getPhotoBinary(const std::string& filename) {
    std::vector<uchar> data;
    CURL* curl = curl_easy_init();
    if (!curl) return data;

    std::string url = service_url_ + "/photo?file=" + filename;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteBinaryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 3000L);

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return data;
}

json CppPersonEngine::uploadPhoto(const std::string& name, const std::string& base64_image) {
    json req = {{"name", name}, {"image", base64_image}};
    std::string resp = httpPost(service_url_ + "/upload_photo", req.dump());
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"error", "Failed to upload photo"}};
}

json CppPersonEngine::deletePhoto(const std::string& filename) {
    json req = {{"file", filename}};
    std::string resp = httpPost(service_url_ + "/delete_photo", req.dump());
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"error", "Failed to delete photo"}};
}

json CppPersonEngine::reloadGallery() {
    std::string resp = httpPost(service_url_ + "/reload_gallery", "{}");
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"status", "error"}};
}

json CppPersonEngine::predictFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return json{{"error", "Empty frame"}};
    }
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
    std::string raw(reinterpret_cast<const char*>(buf.data()), buf.size());
    std::string resp = httpPost(service_url_ + "/predict", raw, "image/jpeg");
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"total_persons", 0}, {"family_count", 0}, {"stranger_count", 0}, {"family_members", json::array()}};
}

json CppPersonEngine::predictBase64(const std::string& base64_image) {
    json req = {{"image", base64_image}};
    std::string resp = httpPost(service_url_ + "/predict", req.dump());
    if (!resp.empty()) {
        try {
            return json::parse(resp);
        } catch (...) {}
    }
    return json{{"error", "Inference failed"}};
}

std::vector<std::string> CppPersonEngine::getEnrolledNames() {
    auto summary = getMembersSummary();
    std::vector<std::string> names;
    if (summary.contains("members") && summary["members"].is_array()) {
        for (const auto& m : summary["members"]) {
            names.push_back(m.value("name", ""));
        }
    }
    return names;
}

} // namespace myhome
