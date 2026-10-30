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

// 单个检测到的人脸明细 (YuNet 人脸检测结果)
struct FaceDetail {
    cv::Rect bbox;
    float score = 0.0f;
};

// 单帧纯净感知结果 (强类型领域对象，分离视觉特征与可视化图像)
struct FrameRecognitionResult {
    int total_persons = 0;
    int family_count = 0;
    int stranger_count = 0;
    int face_count = 0;
    std::vector<std::string> family_members;
    std::vector<PersonDetail> persons;
    std::vector<FaceDetail> faces;
    double latency_ms = 0.0;
};

// 手势识别结果
struct GestureObservation {
    std::string gesture = "NONE";
    int streak_count = 0;
    bool is_confirmed = false;
};

// 方案 A 两级轻量语音核验结果 (Stage 1 口令识别 + Stage 2 声纹比对)
struct VoiceRecognitionResult {
    bool vad_passed = false;
    bool command_matched = false;
    bool is_family = false;
    std::string stage = "IDLE";       // "VAD_SILENCE" | "STAGE1_COMMAND_REJECT" | "STAGE2_STRANGER_REJECT" | "STAGE2_VERIFIED"
    std::string command = "NONE";     // 命中的具体口令 (10 种口令之一，未命中为 "NONE")
    float command_score = 0.0f;       // 第一级口令置信度
    std::string person = "None";      // 识别出的家庭成员姓名 (陌生人为 "陌生人")
    float speaker_score = 0.0f;       // 第二级声纹余弦相似度
    double latency_ms = 0.0;          // 两级推理总耗时 (ms)
};

// 统一开门判定决策结果
struct UnlockDecision {
    bool passed = false;
    std::string status = "idle"; // "idle" | "pending" | "unlocked" | "blocked" | "cooldown" | "cancelled" | "superseded"
    std::string person = "None";
    std::string auth_mode = "None";
    float score = 0.0f;
    std::string action = "None";
    std::string detail = "";
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
    VoiceRecognitionResult voice_result;
    std::string gesture_name = "NONE";
    std::string voice_text = "";
    int elapsed_ms = 0;
};

} // namespace myhome
