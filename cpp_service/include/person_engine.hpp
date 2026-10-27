#pragma once

#include "domain_types.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>

namespace myhome {

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

    // 纯净推理接口 (返回强类型结构体，0 渲染开销，供高性能视频流循环使用 - 遵循 ISP)
    FrameRecognitionResult detectAndRecognize(const cv::Mat& frame);

    // 可视化渲染独立接口 (将识别结果绘制到图像上，供 Web 界面测试按需调用 - 遵循 SRP)
    static cv::Mat renderAnnotation(const cv::Mat& frame, const FrameRecognitionResult& result);

    // 工具函数: 将 cv::Mat 编码为 Base64 JPEG 格式
    static std::string toBase64Jpeg(const cv::Mat& img, int quality = 85);

    // 对单张图片/视频帧执行人体检测与身份识别 (返回包含纯净感知 FrameSummary 与标注图的 JSON，兼容旧 API)
    nlohmann::json predictFrame(const cv::Mat& frame);

    // 对 Base64 编码的测试图片执行人体检测与识别
    nlohmann::json predictBase64(const std::string& base64_image);

    // 获取所有注册家庭成员信息 (最大 15 位)
    nlohmann::json getMembersSummary();

    // 获取照片二进制 (供 Web 预览)
    std::vector<uchar> getPhotoBinary(const std::string& filename);

    // 上传新照片 (保存至训练集目录，防御目录穿越)
    nlohmann::json uploadPhoto(const std::string& name, const std::string& base64_image);

    // 删除单张照片
    nlohmann::json deletePhoto(const std::string& filename);

    // 一键热重载模型与类别 (线程安全)
    nlohmann::json reloadGallery();

    // 获取已注册家庭成员姓名列表 (线程安全)
    std::vector<std::string> getEnrolledNames();

    const std::string& getDbDir() const { return db_dir_; }
    std::string getServiceUrl() const { return "native://opencv_dnn"; }
    std::vector<std::string> getClasses();

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
    mutable std::mutex net_mtx_;
};

} // namespace myhome
