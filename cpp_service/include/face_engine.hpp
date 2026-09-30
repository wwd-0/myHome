#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace myhome {

struct FaceMatchResult {
    std::string person_name;
    float score;
    cv::Rect bbox;
};

class CppFaceEngine {
public:
    CppFaceEngine(const std::string& db_dir,
                  const std::string& det_model_path,
                  const std::string& rec_model_path,
                  float sim_threshold = 0.38f,
                  int min_face_width_px = 90);

    // 扫描并热重载家人照片底库 (支持同一人多张照片自动求均值融合)
    int reloadFamilyDb();

    // 分析单帧画面：返回门前置信度最高的自家人，以及画面中是否存在陌生人
    std::optional<FaceMatchResult> analyzeFrame(cv::Mat& frame, bool& has_stranger_out);

    std::vector<std::string> getEnrolledNames() const;

private:
    static std::string parsePersonName(const std::string& filename);
    std::pair<std::string, float> matchFeature(const cv::Mat& norm_feat) const;

    std::string db_dir_;
    std::string det_model_path_;
    std::string rec_model_path_;
    float sim_threshold_;
    int min_face_width_px_;

    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;
    std::map<std::string, cv::Mat> family_db_;
};

} // namespace myhome
