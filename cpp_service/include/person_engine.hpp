#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>

namespace myhome {

struct PersonDetail {
    std::string name;
    float score;
    bool is_family;
    bool is_stranger;
    cv::Rect bbox;
    std::string status; // "正面", "侧脸", "无脸主体"
    double latency_ms;
};

class CppPersonEngine {
public:
    CppPersonEngine(const std::string& yolo_or_url = "./vision/models/yolov8n.onnx",
                    const std::string& classifier_or_db = "./vision/models/person_classifier.onnx",
                    const std::string& classes_json_path = "./vision/models/classes.json",
                    const std::string& db_dir = "./vision/db/faces");
    ~CppPersonEngine();

    // 检查并确保模型已就绪 (纯 C++ 原生，无需 Python 服务)
    bool ensureServiceRunning();

    // 快速检测 C++ 引擎健康度
    bool isHealthy();

    // 获取所有注册家庭成员信息 (最大 15 位)
    nlohmann::json getMembersSummary();

    // 获取照片二进制 (供 Web 预览)
    std::vector<uchar> getPhotoBinary(const std::string& filename);

    // 上传新照片 (保存至训练集目录)
    nlohmann::json uploadPhoto(const std::string& name, const std::string& base64_image);

    // 删除单张照片
    nlohmann::json deletePhoto(const std::string& filename);

    // 一键热重载模型与类别
    nlohmann::json reloadGallery();

    // 对单张图片/视频帧执行人体检测与身份识别 (纯 C++ 原生推理，返回包含纯净感知 FrameSummary 与标注图的 JSON)
    nlohmann::json predictFrame(const cv::Mat& frame);

    // 对 Base64 编码的测试图片执行人体检测与识别
    nlohmann::json predictBase64(const std::string& base64_image);

    // 获取已注册家庭成员姓名列表
    std::vector<std::string> getEnrolledNames();

    const std::string& getDbDir() const { return db_dir_; }
    std::string getServiceUrl() const { return "native://opencv_dnn"; }
    const std::vector<std::string>& getClasses() const { return classes_; }

private:
    bool loadModels();
    cv::Mat cropMaximalAspectRatio(const cv::Mat& img, cv::Size target_size, cv::Rect* out_crop_rect = nullptr);

    std::string yolo_model_path_;
    std::string classifier_model_path_;
    std::string classes_json_path_;
    std::string db_dir_;

    std::vector<std::string> classes_;
    cv::dnn::Net yolo_net_;
    cv::dnn::Net classifier_net_;
    std::mutex net_mtx_;
};

} // namespace myhome
