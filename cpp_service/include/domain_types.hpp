#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <optional>
#include <chrono>

namespace myhome {

// 单个人体目标明细 (结构化领域数据)
struct PersonDetail {
    std::string name;
    float score = 0.0f;
    bool is_family = false;
    bool is_stranger = false;
    cv::Rect bbox;
    std::string status = "正面";
    double latency_ms = 0.0;
};

// 单帧纯净感知结果 (强类型领域对象，分离视觉特征与可视化图像)
struct FrameRecognitionResult {
    int total_persons = 0;
    int family_count = 0;
    int stranger_count = 0;
    std::vector<std::string> family_members;
    std::vector<PersonDetail> persons;
    double latency_ms = 0.0;
};

// 手势识别结果
struct GestureObservation {
    std::string gesture = "NONE";
    int streak_count = 0;
    bool is_confirmed = false;
};

// 统一开门判定决策结果
struct UnlockDecision {
    bool passed = false;
    std::string person = "None";
    std::string auth_mode = "None";
    float score = 0.0f;
    std::string action = "None";
    int elapsed_ms = 0;
    int relay_pulse_ms = 500;
    std::string timestamp;
};

// 认证事件类型与载荷
enum class AuthEventType {
    FrameRecognized,
    SwitchPressed,
    VoiceCommand
};

struct AuthEvent {
    AuthEventType type = AuthEventType::FrameRecognized;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    FrameRecognitionResult frame_result;
    std::string gesture_name = "NONE";
    std::string voice_text = "";
    int elapsed_ms = 0;
};

} // namespace myhome
