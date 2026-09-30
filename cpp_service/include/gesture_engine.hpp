#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <optional>

namespace myhome {

class CppGestureEngine {
public:
    CppGestureEngine(const std::string& palm_model_path,
                     const std::string& hand_model_path,
                     const std::vector<std::string>& unlock_gestures = {"OK", "VICTORY", "OPEN_PALM", "THUMB_UP"},
                     int confirm_frames = 3);

    // 检测画面中的 21 点手部骨骼并识别手势（连续 confirm_frames 帧确认后返回手势名称）
    std::optional<std::string> detectGesture(cv::Mat& frame);

    void resetStreak();

private:
    struct Anchor {
        float cx;
        float cy;
    };

    void generatePalmAnchors();
    std::optional<std::vector<cv::Point3f>> detectHandLandmarks(const cv::Mat& frame);
    static void drawHandSkeleton(cv::Mat& frame, const std::vector<cv::Point3f>& lm);
    static std::string classify21Landmarks(const std::vector<cv::Point3f>& lm);

    cv::dnn::Net palm_net_;
    cv::dnn::Net hand_net_;
    std::vector<Anchor> anchors_;
    std::vector<std::string> unlock_gestures_;
    int confirm_frames_;

    std::string streak_gesture_;
    int streak_count_ = 0;
};

} // namespace myhome
