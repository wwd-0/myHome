#include "person_engine.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

static const std::string kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64Encode(const unsigned char* data, size_t len) {
    std::string ret;
    int i = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; i < 4; i++) ret += kBase64Chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for (int j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        for (int j = 0; j < i + 1; j++) ret += kBase64Chars[char_array_4[j]];
        while (i++ < 3) ret += '=';
    }
    return ret;
}

std::vector<uchar> base64Decode(const std::string& encoded_string) {
    std::string clean = encoded_string;
    auto comma_pos = clean.find(',');
    if (comma_pos != std::string::npos) {
        clean = clean.substr(comma_pos + 1);
    }
    std::vector<uchar> ret;
    size_t in_len = clean.size();
    size_t i = 0;
    size_t in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (clean[in_] != '=') && (isalnum(static_cast<unsigned char>(clean[in_])) || (clean[in_] == '+') || (clean[in_] == '/'))) {
        char_array_4[i++] = clean[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) char_array_4[i] = static_cast<unsigned char>(kBase64Chars.find(char_array_4[i]));
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; i < 3; i++) ret.push_back(char_array_3[i]);
            i = 0;
        }
    }
    if (i) {
        for (size_t j = 0; j < i; j++) char_array_4[j] = static_cast<unsigned char>(kBase64Chars.find(char_array_4[j]));
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        for (size_t j = 0; j < i - 1; j++) ret.push_back(char_array_3[j]);
    }
    return ret;
}

} // namespace

namespace myhome {

CppPersonEngine::CppPersonEngine(const std::string& yolo_or_url,
                                 const std::string& classifier_or_db,
                                 const std::string& classes_json_path,
                                 const std::string& db_dir)
    : classes_json_path_(classes_json_path),
      db_dir_(db_dir) {
    if (yolo_or_url.rfind("http", 0) == 0) {
        yolo_model_path_ = "./vision/models/yolov8n.onnx";
        classifier_model_path_ = "./vision/models/person_classifier.onnx";
    } else {
        yolo_model_path_ = yolo_or_url;
        classifier_model_path_ = classifier_or_db;
    }
    loadModels();
}

CppPersonEngine::~CppPersonEngine() = default;

bool CppPersonEngine::loadModels() {
    std::lock_guard<std::mutex> lk(net_mtx_);

    std::vector<std::string> yolo_candidates = {
        yolo_model_path_,
        "./vision/models/yolov8n.onnx",
        "../vision/models/yolov8n.onnx"
    };
    std::string resolved_yolo;
    for (const auto& p : yolo_candidates) {
        if (!p.empty() && fs::exists(p)) {
            resolved_yolo = p;
            break;
        }
    }

    std::vector<std::string> cls_candidates = {
        classifier_model_path_,
        "./vision/models/person_classifier.onnx",
        "../vision/models/person_classifier.onnx"
    };
    std::string resolved_cls;
    for (const auto& p : cls_candidates) {
        if (!p.empty() && fs::exists(p)) {
            resolved_cls = p;
            break;
        }
    }

    std::vector<std::string> json_candidates = {
        classes_json_path_,
        "./vision/models/classes.json",
        "../vision/models/classes.json"
    };
    std::string resolved_json;
    for (const auto& p : json_candidates) {
        if (!p.empty() && fs::exists(p)) {
            resolved_json = p;
            break;
        }
    }

    if (resolved_yolo.empty() || resolved_cls.empty()) {
        std::cerr << "❌ [CppPersonEngine] 无法定位 ONNX 模型文件: yolo=" << resolved_yolo
                  << ", classifier=" << resolved_cls << std::endl;
        return false;
    }

    try {
        yolo_net_ = cv::dnn::readNetFromONNX(resolved_yolo);
        classifier_net_ = cv::dnn::readNetFromONNX(resolved_cls);
    } catch (const cv::Exception& e) {
        std::cerr << "❌ [CppPersonEngine] OpenCV DNN 加载模型失败: " << e.what() << std::endl;
        return false;
    }

    classes_.clear();
    if (!resolved_json.empty()) {
        try {
            std::ifstream f(resolved_json);
            json j;
            f >> j;
            if (j.is_array()) {
                for (const auto& item : j) {
                    classes_.push_back(item.get<std::string>());
                }
            }
        } catch (...) {
            classes_ = {"淘淘", "爸爸"};
        }
    }
    if (classes_.empty()) {
        classes_ = {"淘淘", "爸爸"};
    }

    std::cout << "🚀 [CppPersonEngine] 纯 C++ 原生 OpenCV DNN 人体感知引擎就绪！" << std::endl;
    std::cout << "   • YOLOv8 检测模型: " << resolved_yolo << std::endl;
    std::cout << "   • 人体分类模型:   " << resolved_cls << std::endl;
    std::cout << "   • 已内化自家人数: " << classes_.size() << " 位 -> ";
    for (const auto& c : classes_) std::cout << "[" << c << "] ";
    std::cout << std::endl;

    return true;
}

bool CppPersonEngine::isHealthy() {
    std::lock_guard<std::mutex> lk(net_mtx_);
    return !yolo_net_.empty() && !classifier_net_.empty();
}

bool CppPersonEngine::ensureServiceRunning() {
    if (isHealthy()) return true;
    return loadModels();
}

cv::Mat CppPersonEngine::cropMaximalAspectRatio(const cv::Mat& img, cv::Size target_size, cv::Rect* out_crop_rect) {
    int th = target_size.height;
    int tw = target_size.width;
    float target_ratio = static_cast<float>(tw) / static_cast<float>(th);
    int h = img.rows;
    int w = img.cols;
    float current_ratio = static_cast<float>(w) / static_cast<float>(h);

    int crop_x = 0, crop_y = 0, crop_w = w, crop_h = h;
    if (current_ratio > target_ratio) {
        crop_h = h;
        crop_w = std::max(1, static_cast<int>(std::round(h * target_ratio)));
        crop_x = std::max(0, (w - crop_w) / 2);
        crop_y = 0;
    } else {
        crop_w = w;
        crop_h = std::max(1, static_cast<int>(std::round(w / target_ratio)));
        crop_x = 0;
        crop_y = std::max(0, (h - crop_h) / 2);
    }
    crop_w = std::min(w - crop_x, crop_w);
    crop_h = std::min(h - crop_y, crop_h);

    cv::Rect crop_rect(crop_x, crop_y, crop_w, crop_h);
    if (out_crop_rect) *out_crop_rect = crop_rect;

    cv::Mat cropped = img(crop_rect);
    cv::Mat resized;
    cv::resize(cropped, resized, target_size, 0, 0, cv::INTER_AREA);
    return resized;
}

FrameRecognitionResult CppPersonEngine::detectAndRecognize(const cv::Mat& frame) {
    FrameRecognitionResult result;
    if (frame.empty()) return result;

    auto t_start = std::chrono::steady_clock::now();
    int frame_w = frame.cols;
    int frame_h = frame.rows;
    float aspect_ratio = static_cast<float>(frame_w) / static_cast<float>(frame_h);

    std::lock_guard<std::mutex> lk(net_mtx_);
    if (yolo_net_.empty() || classifier_net_.empty()) {
        return result;
    }

    int num_classes = static_cast<int>(classes_.size());

    // 策略 1: 若输入图像本身就是约 1:2 的单人人像切片 (如 256x512 测试图/头像)
    bool is_direct_portrait = (aspect_ratio >= 0.40f && aspect_ratio <= 0.65f);

    if (is_direct_portrait) {
        cv::Rect crop_rect;
        cv::Mat patch = cropMaximalAspectRatio(frame, cv::Size(256, 512), &crop_rect);

        cv::Mat cls_blob = cv::dnn::blobFromImage(
            patch, 1.0 / 128.0, cv::Size(256, 512), cv::Scalar(127.5, 127.5, 127.5), true, false);
        classifier_net_.setInput(cls_blob);
        cv::Mat cls_out = classifier_net_.forward();

        int best_idx = 0;
        float best_score = -1.0f;
        if (cls_out.total() > 0) {
            const float* scores = cls_out.ptr<float>(0);
            for (int c = 0; c < std::min(static_cast<int>(cls_out.total()), num_classes); ++c) {
                if (scores[c] > best_score) {
                    best_score = scores[c];
                    best_idx = c;
                }
            }
        }

        float threshold = 0.20f;
        bool is_family = (best_score >= threshold && best_idx < num_classes);
        bool is_stranger = !is_family;
        std::string name = is_family ? classes_[best_idx] : "陌生人";
        float norm_score = 0.0f;
        if (is_family) {
            norm_score = 0.80f + 0.19f * std::min(1.0f, std::max(0.0f, (best_score - threshold) / 0.25f));
        } else {
            norm_score = std::max(0.0f, std::min(0.30f, (best_score - 0.05f) / 0.15f * 0.30f));
        }

        PersonDetail pd;
        pd.name = name;
        pd.score = norm_score;
        pd.is_family = is_family;
        pd.is_stranger = is_stranger;
        pd.bbox = cv::Rect(0, 0, frame_w, frame_h);
        pd.status = "正面";
        pd.latency_ms = 0.0;
        result.persons.push_back(pd);
    } else {
        // 策略 2: 完整门禁画面，先运行 YOLOv8-Nano 检测所有人像目标
        cv::Mat yolo_blob = cv::dnn::blobFromImage(
            frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        yolo_net_.setInput(yolo_blob);
        cv::Mat yolo_out = yolo_net_.forward();

        cv::Mat det_mat(84, 8400, CV_32F, yolo_out.ptr<float>());
        cv::Mat det_t;
        cv::transpose(det_mat, det_t);

        std::vector<cv::Rect> candidate_boxes;
        std::vector<float> candidate_confs;
        float x_factor = static_cast<float>(frame_w) / 640.0f;
        float y_factor = static_cast<float>(frame_h) / 640.0f;

        for (int i = 0; i < det_t.rows; ++i) {
            const float* row = det_t.ptr<float>(i);
            float person_conf = row[4];
            if (person_conf >= 0.40f) {
                float cx = row[0];
                float cy = row[1];
                float w = row[2];
                float h = row[3];
                int left = static_cast<int>((cx - 0.5f * w) * x_factor);
                int top = static_cast<int>((cy - 0.5f * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);
                if (width >= 35 && height >= 70) {
                    candidate_boxes.push_back(cv::Rect(left, top, width, height));
                    candidate_confs.push_back(person_conf);
                }
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(candidate_boxes, candidate_confs, 0.40f, 0.45f, nms_indices);

        // 注意: 若检测不到人像目标，直接返回 0 人，坚决不做全画幅盲切兜底以杜绝误报
        for (int idx : nms_indices) {
            cv::Rect b = candidate_boxes[idx];
            int pad_x = static_cast<int>(b.width * 0.08f);
            int pad_y = static_cast<int>(b.height * 0.08f);
            int x1 = std::max(0, b.x - pad_x);
            int y1 = std::max(0, b.y - pad_y);
            int x2 = std::min(frame_w, b.x + b.width + pad_x);
            int y2 = std::min(frame_h, b.y + b.height + pad_y);
            int crop_w = std::max(1, x2 - x1);
            int crop_h = std::max(1, y2 - y1);
            cv::Rect padded_rect(x1, y1, crop_w, crop_h);

            cv::Mat person_crop = frame(padded_rect);
            cv::Mat patch = cropMaximalAspectRatio(person_crop, cv::Size(256, 512));

            cv::Mat cls_blob = cv::dnn::blobFromImage(
                patch, 1.0 / 128.0, cv::Size(256, 512), cv::Scalar(127.5, 127.5, 127.5), true, false);
            classifier_net_.setInput(cls_blob);
            cv::Mat cls_out = classifier_net_.forward();

            int best_idx = 0;
            float best_score = -1.0f;
            if (cls_out.total() > 0) {
                const float* scores = cls_out.ptr<float>(0);
                for (int c = 0; c < std::min(static_cast<int>(cls_out.total()), num_classes); ++c) {
                    if (scores[c] > best_score) {
                        best_score = scores[c];
                        best_idx = c;
                    }
                }
            }

            float threshold = 0.20f;
            bool is_family = (best_score >= threshold && best_idx < num_classes);
            bool is_stranger = !is_family;
            std::string name = is_family ? classes_[best_idx] : "陌生人";
            float norm_score = 0.0f;
            if (is_family) {
                norm_score = 0.80f + 0.19f * std::min(1.0f, std::max(0.0f, (best_score - threshold) / 0.25f));
            } else {
                norm_score = std::max(0.0f, std::min(0.30f, (best_score - 0.05f) / 0.15f * 0.30f));
            }

            PersonDetail pd;
            pd.name = name;
            pd.score = norm_score;
            pd.is_family = is_family;
            pd.is_stranger = is_stranger;
            pd.bbox = b;
            pd.status = "正面";
            pd.latency_ms = 0.0;
            result.persons.push_back(pd);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.total_persons = static_cast<int>(result.persons.size());

    for (const auto& p : result.persons) {
        if (p.is_family) {
            result.family_count++;
            if (std::find(result.family_members.begin(), result.family_members.end(), p.name) == result.family_members.end()) {
                result.family_members.push_back(p.name);
            }
        } else {
            result.stranger_count++;
        }
    }

    return result;
}

cv::Mat CppPersonEngine::renderAnnotation(const cv::Mat& frame, const FrameRecognitionResult& result) {
    if (frame.empty()) return cv::Mat();
    cv::Mat annotated = frame.clone();
    int frame_w = annotated.cols;

    for (const auto& p : result.persons) {
        cv::Scalar color = p.is_family ? cv::Scalar(0, 220, 100) : cv::Scalar(60, 60, 240);
        cv::rectangle(annotated, p.bbox, color, 2, cv::LINE_AA);

        int bx = p.bbox.x, by = p.bbox.y, bw = p.bbox.width, bh = p.bbox.height;
        int c_len = std::min(20, std::min(std::max(1, bw / 4), std::max(1, bh / 4)));
        int t_thick = 3;

        cv::line(annotated, cv::Point(bx, by), cv::Point(bx + c_len, by), color, t_thick);
        cv::line(annotated, cv::Point(bx, by), cv::Point(bx, by + c_len), color, t_thick);
        cv::line(annotated, cv::Point(bx + bw, by), cv::Point(bx + bw - c_len, by), color, t_thick);
        cv::line(annotated, cv::Point(bx + bw, by), cv::Point(bx + bw, by + c_len), color, t_thick);
        cv::line(annotated, cv::Point(bx, by + bh), cv::Point(bx + c_len, by + bh), color, t_thick);
        cv::line(annotated, cv::Point(bx, by + bh), cv::Point(bx, by + bh - c_len), color, t_thick);
        cv::line(annotated, cv::Point(bx + bw, by + bh), cv::Point(bx + bw - c_len, by + bh), color, t_thick);
        cv::line(annotated, cv::Point(bx + bw, by + bh), cv::Point(bx + bw, by + bh - c_len), color, t_thick);

        std::string tag_text = p.name + " (" + cv::format("%.2f", p.score) + ")";
        int tag_y = std::max(24, by - 8);
        int tag_w = std::min(bw, 180);
        cv::rectangle(annotated, cv::Point(bx, tag_y - 20), cv::Point(bx + tag_w, tag_y + 4), color, -1);
        cv::putText(annotated, tag_text, cv::Point(bx + 6, tag_y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    int hud_h = 44;
    cv::Mat overlay = annotated.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(frame_w, hud_h), cv::Scalar(20, 24, 30), -1);
    cv::addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated);

    std::string hud_info = cv::format("Total: %d | Family: %d | Stranger: %d | Latency: %.1fms (Pure C++)",
                                      result.total_persons, result.family_count, result.stranger_count, result.latency_ms);
    cv::putText(annotated, hud_info, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 230, 255), 2, cv::LINE_AA);

    return annotated;
}

std::string CppPersonEngine::toBase64Jpeg(const cv::Mat& img, int quality) {
    if (img.empty()) return "";
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, quality});
    return "data:image/jpeg;base64," + base64Encode(buf.data(), buf.size());
}

nlohmann::json CppPersonEngine::predictFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return json{{"error", "Empty frame"}};
    }

    FrameRecognitionResult result = detectAndRecognize(frame);
    cv::Mat annotated = renderAnnotation(frame, result);
    std::string b64_img = toBase64Jpeg(annotated, 85);

    json persons_json = json::array();
    for (const auto& p : result.persons) {
        persons_json.push_back({
            {"name", p.name},
            {"score", std::round(p.score * 1000.0f) / 1000.0f},
            {"is_family", p.is_family},
            {"is_stranger", p.is_stranger},
            {"status", p.status},
            {"bbox", {{"x", p.bbox.x}, {"y", p.bbox.y}, {"w", p.bbox.width}, {"h", p.bbox.height}}},
            {"latency_ms", std::round(result.latency_ms * 10.0) / 10.0}
        });
    }

    return json{
        {"total_persons", result.total_persons},
        {"family_count", result.family_count},
        {"stranger_count", result.stranger_count},
        {"family_members", result.family_members},
        {"has_stranger", result.stranger_count > 0},
        {"persons", persons_json},
        {"latency_ms", std::round(result.latency_ms * 10.0) / 10.0},
        {"annotated_image", b64_img}
    };
}

nlohmann::json CppPersonEngine::predictBase64(const std::string& base64_image) {
    if (base64_image.empty()) {
        return json{{"error", "Empty base64 image"}};
    }
    auto raw_bytes = base64Decode(base64_image);
    if (raw_bytes.empty()) {
        return json{{"error", "Failed to decode base64 data"}};
    }
    cv::Mat frame = cv::imdecode(raw_bytes, cv::IMREAD_COLOR);
    if (frame.empty()) {
        return json{{"error", "Failed to decode image buffer to cv::Mat"}};
    }
    return predictFrame(frame);
}

nlohmann::json CppPersonEngine::getMembersSummary() {
    std::vector<std::string> current_classes;
    {
        std::lock_guard<std::mutex> lk(net_mtx_);
        current_classes = classes_;
    }

    json members_list = json::array();
    for (const auto& c : current_classes) {
        int count = 0;
        std::string c_dir = "door_person_detector/data/dataset/train/" + c;
        if (!fs::exists(c_dir)) {
            c_dir = "../door_person_detector/data/dataset/train/" + c;
        }
        if (fs::exists(c_dir)) {
            for (const auto& entry : fs::directory_iterator(c_dir)) {
                if (entry.is_regular_file() && entry.path().filename().string()[0] != '.') {
                    count++;
                }
            }
        }
        members_list.push_back({
            {"name", c},
            {"sample_count", count},
            {"is_model_class", true},
            {"status", "已内化至 C++ 模型权重 (0 照片存储)"}
        });
    }

    return json{
        {"mode", "方案 A (端到端分类模型，0 照片存储)"},
        {"model_type", "YOLOv8-Nano + LightPersonIDNet (C++ Native OpenCV DNN)"},
        {"device", "OpenCV DNN (Pure C++ Native)"},
        {"total_members", current_classes.size()},
        {"max_members", 15},
        {"members", members_list},
        {"classes", current_classes}
    };
}

std::vector<uchar> CppPersonEngine::getPhotoBinary(const std::string& filename) {
    std::vector<uchar> data;
    if (filename.empty() || filename.find("..") != std::string::npos || filename.find('/') != std::string::npos) {
        return data;
    }
    std::vector<std::string> search_dirs = {
        db_dir_,
        "vision/db/faces",
        "../vision/db/faces",
        "door_person_detector/data/dataset/train",
        "../door_person_detector/data/dataset/train"
    };
    for (const auto& d : search_dirs) {
        if (!fs::exists(d)) continue;
        for (const auto& entry : fs::recursive_directory_iterator(d)) {
            if (entry.is_regular_file() && entry.path().filename().string() == filename) {
                std::ifstream file(entry.path(), std::ios::binary);
                if (file) {
                    return std::vector<uchar>((std::istreambuf_iterator<char>(file)),
                                              std::istreambuf_iterator<char>());
                }
            }
        }
    }
    return data;
}

nlohmann::json CppPersonEngine::uploadPhoto(const std::string& name, const std::string& base64_image) {
    if (name.empty() || base64_image.empty()) {
        return {{"error", "Invalid photo name or image"}};
    }

    // 防御目录穿越
    std::string safe_name;
    for (char ch : name) {
        if (ch != '/' && ch != '\\' && ch != '.' && ch != ':') {
            safe_name += ch;
        }
    }
    if (safe_name.empty()) {
        return {{"error", "Invalid photo name (illegal characters)"}};
    }

    auto raw_bytes = base64Decode(base64_image);
    if (raw_bytes.empty()) {
        return {{"error", "Failed to decode base64 image"}};
    }
    std::string target_dir = "door_person_detector/data/dataset/train/" + safe_name;
    if (!fs::exists("door_person_detector") && fs::exists("../door_person_detector")) {
        target_dir = "../" + target_dir;
    }
    fs::create_directories(target_dir);
    std::string fname = safe_name + "_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
    std::ofstream out(target_dir + "/" + fname, std::ios::binary);
    out.write(reinterpret_cast<const char*>(raw_bytes.data()), raw_bytes.size());
    return {{"status", "success"}, {"filename", fname}, {"message", "照片已保存至训练集目录"}};
}

nlohmann::json CppPersonEngine::deletePhoto(const std::string& filename) {
    if (filename.empty() || filename.find("..") != std::string::npos || filename.find('/') != std::string::npos) {
        return {{"error", "Invalid filename"}};
    }
    std::vector<std::string> search_dirs = {
        "door_person_detector/data/dataset/train",
        "../door_person_detector/data/dataset/train",
        db_dir_,
        "vision/db/faces"
    };
    for (const auto& d : search_dirs) {
        if (!fs::exists(d)) continue;
        for (const auto& entry : fs::recursive_directory_iterator(d)) {
            if (entry.is_regular_file() && entry.path().filename().string() == filename) {
                fs::remove(entry.path());
                return {{"status", "success"}, {"deleted", filename}};
            }
        }
    }
    return {{"error", "File not found"}};
}

nlohmann::json CppPersonEngine::reloadGallery() {
    bool ok = loadModels();
    std::vector<std::string> current_classes;
    {
        std::lock_guard<std::mutex> lk(net_mtx_);
        current_classes = classes_;
    }
    return {
        {"status", ok ? "success" : "error"},
        {"message", ok ? "C++ 原生模型与家庭成员映射已刷新" : "模型重载失败"},
        {"classes", current_classes}
    };
}

std::vector<std::string> CppPersonEngine::getEnrolledNames() {
    std::lock_guard<std::mutex> lk(net_mtx_);
    return classes_;
}

std::vector<std::string> CppPersonEngine::getClasses() {
    std::lock_guard<std::mutex> lk(net_mtx_);
    return classes_;
}

} // namespace myhome
