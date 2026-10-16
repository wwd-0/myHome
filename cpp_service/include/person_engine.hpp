#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace myhome {

struct PersonMatchResult {
    std::string person_name;
    float score;
    bool is_family;
    bool is_stranger;
    cv::Rect bbox;
    std::string status; // "正面", "侧脸", "无脸主体"
};

class CppPersonEngine {
public:
    CppPersonEngine(const std::string& service_url = "http://127.0.0.1:8092",
                    const std::string& db_dir = "./vision/db/faces");
    ~CppPersonEngine();

    // 检查并自动拉起 Python 人体模型微服务
    bool ensureServiceRunning();

    // 快速检测 Python 服务健康度
    bool isHealthy();

    // 获取所有注册家庭成员信息 (最大 15 位)
    nlohmann::json getMembersSummary();

    // 获取照片二进制 (供 Web 预览)
    std::vector<uchar> getPhotoBinary(const std::string& filename);

    // 上传新照片并自动重建特征底库
    nlohmann::json uploadPhoto(const std::string& name, const std::string& base64_image);

    // 删除单张照片并自动重建特征底库
    nlohmann::json deletePhoto(const std::string& filename);

    // 一键热重载底库特征
    nlohmann::json reloadGallery();

    // 对单张图片/视频帧执行人体检测与身份识别 (返回包含纯净感知 FrameSummary 与标注图的 JSON)
    nlohmann::json predictFrame(const cv::Mat& frame);

    // 对 Base64 编码的测试图片执行人体检测与识别
    nlohmann::json predictBase64(const std::string& base64_image);

    // 获取已注册家庭成员姓名列表
    std::vector<std::string> getEnrolledNames();

    const std::string& getDbDir() const { return db_dir_; }
    const std::string& getServiceUrl() const { return service_url_; }

private:
    std::string httpGet(const std::string& url);
    std::string httpPost(const std::string& url, const std::string& body, const std::string& content_type = "application/json");

    std::string service_url_;
    std::string db_dir_;
    std::vector<std::string> cached_members_;
};

} // namespace myhome
