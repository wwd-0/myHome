#pragma once

#include "domain_types.hpp"
#include "policy.hpp"
#include "actuator.hpp"
#include "person_engine.hpp"
#include "gesture_engine.hpp"
#include "voice_engine.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <optional>
#include <cstdint>

namespace myhome {

// 人脸抓拍快照策略配置
struct SnapshotConfig {
    bool enabled = true;
    std::string root_dir = "./output/snapshots";
    std::string stranger_subdir = "陌生人";
    double min_interval_sec = 1.0;
    size_t max_total = 500;
    size_t cleanup_batch = 100;
    std::string config_file_path = "./config.yaml";
};

// 门禁核心协调器 (Facade & Coordinator Pattern - 遵循 SRP, OCP, DIP, LoD)
class VerifyServiceCoordinator {
public:
    VerifyServiceCoordinator(std::shared_ptr<CppPersonEngine> person_engine,
                             std::shared_ptr<CppGestureEngine> gesture_engine,
                             std::shared_ptr<CppVoiceEngine> voice_engine,
                             std::shared_ptr<CompositePolicy> composite_policy,
                             std::shared_ptr<IUnlockActuator> actuator,
                             const std::string& cam_source,
                             double active_window_sec = 15.0);
    ~VerifyServiceCoordinator();

    // 核心会话与控制流
    nlohmann::json requestVerifySync(double custom_timeout_sec = -1.0);
    nlohmann::json requestVerifyAsync();
    nlohmann::json requestStopVerify();
    void stop();

    // 主拉流工作循环 (运行在专用工作线程)
    void runMainThreadLoop();

    // 模式 2 处理：收到智能开关按下事件 (统一逻辑，前后 N 秒互锁，支持拉流截止时间自适应延长)
    nlohmann::json handleSmartSwitchTrigger();

    // 模式 2 查询：供 HA 同步查询当前门厅是否处于有效互锁窗口期
    nlohmann::json handleSmartSwitchVerifyQuery();

    // 模式 3 声音核验接口：支持独立语音口令 + 声纹核验直接开门 (无需开关或人脸配合)
    nlohmann::json handleVoiceVerify(const nlohmann::json& req);

    // 家庭成员声纹与口令底库管理门面接口 (支持 10 种口令、最多 10 位家人)
    nlohmann::json getVoiceSummary();
    nlohmann::json uploadVoiceSample(const std::string& person,
                                     const std::string& command_tag,
                                     const std::string& base64_wav);
    nlohmann::json deleteVoiceSample(const std::string& filename);
    std::vector<uchar> getVoiceFileBinary(const std::string& filename);
    nlohmann::json reloadVoiceGallery();

    // 页面测试专用：单张图片 Base64 纯净推理与仿真
    nlohmann::json testSingleImage(const std::string& b64_img);

    // 硬件点动脉冲与 HA Webhook 手动模拟
    nlohmann::json triggerManualRelayTest();

    // 获取服务状态与历史日志
    nlohmann::json getStatus();

    // 实时最新画面 JPEG (供 /api/frame.jpg 流送)
    std::vector<uchar> getLatestJpeg();

    // 家庭成员底库管理门面接口 (遵循 Law of Demeter - 避免外部穿透)
    nlohmann::json getMembersSummary();
    std::vector<uchar> getPhotoBinary(const std::string& filename);
    nlohmann::json uploadPhoto(const std::string& name, const std::string& base64_image);
    nlohmann::json deletePhoto(const std::string& filename);
    nlohmann::json reloadGallery();

    // 人脸自动抓拍快照配置与管理接口
    void configureSnapshots(const SnapshotConfig& cfg);
    nlohmann::json updateSnapshotSettings(const nlohmann::json& req);
    nlohmann::json getSnapshotsSummary();
    std::vector<uchar> getSnapshotBinary(const std::string& subdir, const std::string& filename);
    nlohmann::json deleteSnapshot(const std::string& subdir,
                                  const std::string& filename,
                                  bool clear_subdir = false,
                                  bool clear_all = false);

private:
    struct PendingSwitchAudit {
        uint64_t event_id = 0;
        std::chrono::steady_clock::time_point pressed_at{};
        std::chrono::steady_clock::time_point deadline{};
        double window_sec = 5.0;
        bool saw_stranger = false;
        bool camera_unavailable = false;
    };

    uint64_t recordAuditEvent(UnlockDecision decision,
                              const std::string& status,
                              const std::string& detail,
                              bool increment_unlock);
    void resolvePendingSwitchUnlock(UnlockDecision decision, bool actuator_executed);
    void finalizeExpiredSwitchLocked(std::chrono::steady_clock::time_point now);
    void pushHistoryLocked(nlohmann::json item);
    nlohmann::json* findHistoryByIdLocked(uint64_t id);
    void extendVerifyTimeoutLocked(double window_sec);

    // 人脸快照内部辅助方法
    void maybeSaveFaceSnapshot(const cv::Mat& frame, const FrameRecognitionResult& frame_res);
    void ensureSnapshotSubdirsLocked();
    size_t scanSnapshotCountLocked() const;
    void enforceSnapshotRetentionLocked();

    std::shared_ptr<CppPersonEngine> person_engine_;
    std::shared_ptr<CppGestureEngine> gesture_engine_;
    std::shared_ptr<CppVoiceEngine> voice_engine_;
    std::shared_ptr<CompositePolicy> composite_policy_;
    std::shared_ptr<IUnlockActuator> actuator_;

    std::string cam_source_;
    double active_window_sec_;

    mutable std::mutex mtx_;
    mutable std::mutex frame_mtx_;
    std::vector<uchar> latest_jpeg_;

    std::condition_variable cv_wakeup_;
    std::condition_variable cv_done_;
    bool running_ = true;
    bool verifying_ = false;
    bool stop_requested_ = false;
    bool verify_done_ = false;

    // 自适应拉流截止时间 (解决开关按下时拉流可能提前关闭的 Bug)
    std::chrono::steady_clock::time_point verify_deadline_{};
    double remaining_sec_ = 0.0;

    // 实时感知数据缓存
    std::string live_person_ = "None";
    float live_score_ = 0.0f;
    bool live_has_stranger_ = false;
    std::string live_voice_stage_ = "IDLE";
    std::string live_voice_command_ = "NONE";
    std::string live_voice_person_ = "None";
    float live_voice_score_ = 0.0f;

    // 审计与统计
    uint64_t next_event_id_ = 1;
    std::optional<PendingSwitchAudit> pending_switch_;
    UnlockDecision last_decision_;
    std::vector<nlohmann::json> history_;
    std::atomic<int> unlock_count_{0};

    // 人脸抓拍快照状态与限流控制
    mutable std::mutex snapshot_mtx_;
    SnapshotConfig snapshot_cfg_;
    std::chrono::steady_clock::time_point last_snapshot_steady_{};
    size_t cached_snapshot_count_ = 0;
};

} // namespace myhome
