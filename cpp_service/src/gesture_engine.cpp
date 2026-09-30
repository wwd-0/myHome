#include "gesture_engine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace myhome {

static const std::vector<std::pair<int, int>> kHandConnections = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {5, 9}, {9, 10}, {10, 11}, {11, 12},
    {9, 13}, {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}
};

CppGestureEngine::CppGestureEngine(const std::string& palm_model_path,
                                   const std::string& hand_model_path,
                                   const std::vector<std::string>& unlock_gestures,
                                   int confirm_frames)
    : unlock_gestures_(unlock_gestures),
      confirm_frames_(confirm_frames) {
    generatePalmAnchors();
    palm_net_ = cv::dnn::readNetFromONNX(palm_model_path);
    hand_net_ = cv::dnn::readNetFromONNX(hand_model_path);
    std::cout << "🖐️ [CppGestureEngine] 已加载手势 ONNX 权重 (防抖确认帧数: " << confirm_frames_ << ")" << std::endl;
}

void CppGestureEngine::generatePalmAnchors() {
    anchors_.clear();
    anchors_.reserve(2016);
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) {
            float cx = (x + 0.5f) / 24.0f;
            float cy = (y + 0.5f) / 24.0f;
            anchors_.push_back({cx, cy});
            anchors_.push_back({cx, cy});
        }
    }
    for (int y = 0; y < 12; ++y) {
        for (int x = 0; x < 12; ++x) {
            float cx = (x + 0.5f) / 12.0f;
            float cy = (y + 0.5f) / 12.0f;
            for (int k = 0; k < 6; ++k) {
                anchors_.push_back({cx, cy});
            }
        }
    }
}

std::optional<std::vector<cv::Point3f>> CppGestureEngine::detectHandLandmarks(const cv::Mat& frame) {
    int h = frame.rows;
    int w = frame.cols;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    float scale = std::min(192.0f / w, 192.0f / h);
    int nw = static_cast<int>(std::round(w * scale));
    int nh = static_cast<int>(std::round(h * scale));
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(nw, nh));

    int pad_top = (192 - nh) / 2;
    int pad_left = (192 - nw) / 2;
    cv::Mat padded = cv::Mat::zeros(192, 192, CV_8UC3);
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, nw, nh)));

    int sz_p[] = {1, 192, 192, 3};
    cv::Mat blob_p(4, sz_p, CV_32F);
    padded.convertTo(blob_p.reshape(3, 192 * 192), CV_32F, 1.0 / 255.0);

    palm_net_.setInput(blob_p);
    std::vector<cv::Mat> out_p;
    palm_net_.forward(out_p, std::vector<std::string>{"Identity", "Identity_1"});
    const cv::Mat& reg_out = out_p[0];   // (1, 2016, 18)
    const cv::Mat& score_out = out_p[1]; // (1, 2016, 1)

    const float* score_ptr = score_out.ptr<float>();
    int best_idx = 0;
    float best_prob = -1.0f;
    for (int i = 0; i < 2016; ++i) {
        float logit = std::clamp(score_ptr[i], -50.0f, 50.0f);
        float prob = 1.0f / (1.0f + std::exp(-logit));
        if (prob > best_prob) {
            best_prob = prob;
            best_idx = i;
        }
    }

    if (best_prob < 0.60f) return std::nullopt;

    const float* row = reg_out.ptr<float>() + best_idx * 18;
    const Anchor& anc = anchors_[best_idx];

    float cx = anc.cx + row[0] / 192.0f;
    float cy = anc.cy + row[1] / 192.0f;
    float pw = row[2] / 192.0f;
    float ph = row[3] / 192.0f;

    float kp0_x = anc.cx + row[4] / 192.0f;
    float kp0_y = anc.cy + row[5] / 192.0f;
    float kp2_x = anc.cx + row[8] / 192.0f;
    float kp2_y = anc.cy + row[9] / 192.0f;

    auto toOrig = [&](float nx, float ny) -> cv::Point2f {
        return cv::Point2f((nx * 192.0f - pad_left) / scale,
                           (ny * 192.0f - pad_top) / scale);
    };

    cv::Point2f oc = toOrig(cx, cy);
    cv::Point2f okp0 = toOrig(kp0_x, kp0_y);
    cv::Point2f okp2 = toOrig(kp2_x, kp2_y);
    float box_size = std::max(pw * 192.0f / scale, ph * 192.0f / scale) * 2.6f;

    float vx = okp2.x - okp0.x;
    float vy = okp2.y - okp0.y;
    float v_len = std::hypot(vx, vy) + 1e-6f;
    float hand_cx = oc.x + (vx / v_len) * (box_size * 0.18f);
    float hand_cy = oc.y + (vy / v_len) * (box_size * 0.18f);

    float angle_rad = static_cast<float>(M_PI * 0.5) - std::atan2(-vy, vx);
    float angle_deg = angle_rad * 180.0f / static_cast<float>(M_PI);

    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(hand_cx, hand_cy), angle_deg, 224.0 / (box_size + 1e-6f));
    rot_mat.at<double>(0, 2) += 112.0 - hand_cx;
    rot_mat.at<double>(1, 2) += 112.0 - hand_cy;

    cv::Mat hand_crop;
    cv::warpAffine(rgb, hand_crop, rot_mat, cv::Size(224, 224), cv::INTER_LINEAR);

    int sz_h[] = {1, 224, 224, 3};
    cv::Mat blob_h(4, sz_h, CV_32F);
    hand_crop.convertTo(blob_h.reshape(3, 224 * 224), CV_32F, 1.0 / 255.0);

    hand_net_.setInput(blob_h);
    std::vector<cv::Mat> out_h;
    hand_net_.forward(out_h, std::vector<std::string>{"Identity", "Identity_1"});
    const cv::Mat& lms_out = out_h[0];        // (1, 63)
    const cv::Mat& hand_score_out = out_h[1]; // (1, 1)

    if (hand_score_out.at<float>(0, 0) < 0.50f) return std::nullopt;

    cv::Mat inv_mat;
    cv::invertAffineTransform(rot_mat, inv_mat);
    const float* pts_ptr = lms_out.ptr<float>();

    std::vector<cv::Point3f> landmarks(21);
    for (int i = 0; i < 21; ++i) {
        float px = pts_ptr[i * 3 + 0];
        float py = pts_ptr[i * 3 + 1];
        float pz = pts_ptr[i * 3 + 2];
        float ox = static_cast<float>(inv_mat.at<double>(0, 0) * px + inv_mat.at<double>(0, 1) * py + inv_mat.at<double>(0, 2));
        float oy = static_cast<float>(inv_mat.at<double>(1, 0) * px + inv_mat.at<double>(1, 1) * py + inv_mat.at<double>(1, 2));
        landmarks[i] = cv::Point3f(ox / w, oy / h, pz);
    }

    return landmarks;
}

void CppGestureEngine::drawHandSkeleton(cv::Mat& frame, const std::vector<cv::Point3f>& lm) {
    int h = frame.rows;
    int w = frame.cols;
    std::vector<cv::Point> pts(21);
    for (int i = 0; i < 21; ++i) {
        pts[i] = cv::Point(static_cast<int>(lm[i].x * w), static_cast<int>(lm[i].y * h));
    }
    for (const auto& [i, j] : kHandConnections) {
        cv::line(frame, pts[i], pts[j], cv::Scalar(0, 255, 180), 2, cv::LINE_AA);
    }
    for (int i = 0; i < 21; ++i) {
        bool is_tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
        cv::circle(frame, pts[i], is_tip ? 5 : 3,
                   is_tip ? cv::Scalar(0, 120, 255) : cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    }
}

std::string CppGestureEngine::classify21Landmarks(const std::vector<cv::Point3f>& lm) {
    auto distToWrist = [&](int idx) {
        return std::hypot(lm[idx].x - lm[0].x, lm[idx].y - lm[0].y);
    };

    bool index_up = distToWrist(8) > distToWrist(6) * 1.13f;
    bool middle_up = distToWrist(12) > distToWrist(10) * 1.13f;
    bool ring_up = distToWrist(16) > distToWrist(14) * 1.13f;
    bool pinky_up = distToWrist(20) > distToWrist(18) * 1.13f;

    float pinch_dist = std::hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);
    float palm_scale = distToWrist(9) + 1e-6f;

    if ((pinch_dist / palm_scale) < 0.42f && middle_up && ring_up && pinky_up) {
        return "OK";
    }
    if (index_up && middle_up && !ring_up && !pinky_up) {
        return "VICTORY";
    }
    float thumb_open = std::hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) / palm_scale;
    if (index_up && middle_up && ring_up && pinky_up && thumb_open > 0.32f) {
        return "OPEN_PALM";
    }
    if (!index_up && !middle_up && !ring_up && !pinky_up && (lm[4].y < lm[5].y - 0.04f)) {
        return "THUMB_UP";
    }
    return "NONE";
}

void CppGestureEngine::resetStreak() {
    streak_gesture_.clear();
    streak_count_ = 0;
}

std::optional<std::string> CppGestureEngine::detectGesture(cv::Mat& frame) {
    auto landmarks = detectHandLandmarks(frame);
    std::string current = "NONE";
    if (landmarks.has_value()) {
        drawHandSkeleton(frame, *landmarks);
        current = classify21Landmarks(*landmarks);
    }

    if (current != "NONE") {
        if (current == streak_gesture_) {
            streak_count_++;
        } else {
            streak_gesture_ = current;
            streak_count_ = 1;
        }
        std::string txt = "Gesture: " + current + " (" + std::to_string(streak_count_) + "/" + std::to_string(confirm_frames_) + ")";
        cv::putText(frame, txt, cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 215, 255), 2);
    } else {
        resetStreak();
    }

    if (streak_count_ >= confirm_frames_) {
        bool allowed = std::find(unlock_gestures_.begin(), unlock_gestures_.end(), current) != unlock_gestures_.end();
        if (allowed) {
            resetStreak();
            return current;
        }
    }

    return std::nullopt;
}

} // namespace myhome
