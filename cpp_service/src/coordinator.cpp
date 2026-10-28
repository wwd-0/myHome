#include "coordinator.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <thread>
#include <cmath>
#include <cctype>

namespace fs = std::filesystem;

namespace myhome {

namespace {

std::string currentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S");
    return oss.str();
}

std::string formatOneDecimal(double val) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << val;
    return oss.str();
}

std::string formatIntervalValue(double val) {
    double r1 = std::round(val * 10.0) / 10.0;
    double r2 = std::round(val * 100.0) / 100.0;
    std::ostringstream oss;
    if (std::abs(r2 - r1) < 1e-6) {
        oss << std::fixed << std::setprecision(1) << r1;
    } else {
        oss << std::fixed << std::setprecision(2) << r2;
    }
    return oss.str();
}

bool persistSnapshotIntervalToYaml(const std::string& config_path, double min_interval_sec, std::string& out_err) {
    std::ifstream ifs(config_path);
    if (!ifs) {
        out_err = "无法读取配置文件: " + config_path;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    ifs.close();

    std::string formatted = formatIntervalValue(min_interval_sec);
    bool in_vision = false;
    bool in_snapshot = false;
    int vision_idx = -1;
    int snapshot_idx = -1;
    bool updated = false;

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& raw = lines[i];
        size_t indent = 0;
        while (indent < raw.size() && raw[indent] == ' ') {
            indent++;
        }
        std::string trimmed = raw.substr(indent);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        if (indent == 0) {
            in_vision = (trimmed.rfind("vision:", 0) == 0);
            in_snapshot = false;
            if (in_vision) {
                vision_idx = static_cast<int>(i);
            }
        } else if (in_vision && indent == 2) {
            in_snapshot = (trimmed.rfind("snapshot:", 0) == 0);
            if (in_snapshot) {
                snapshot_idx = static_cast<int>(i);
            }
        } else if (in_snapshot && indent >= 4) {
            if (trimmed.rfind("min_interval_sec:", 0) == 0) {
                std::string trailing_comment;
                size_t hash_pos = trimmed.find('#');
                if (hash_pos != std::string::npos) {
                    trailing_comment = "  " + trimmed.substr(hash_pos);
                }
                lines[i] = std::string(indent, ' ') + "min_interval_sec: " + formatted + trailing_comment;
                updated = true;
                break;
            }
        }
    }

    if (!updated) {
        if (snapshot_idx >= 0) {
            lines.insert(lines.begin() + snapshot_idx + 1, "    min_interval_sec: " + formatted);
            updated = true;
        } else if (vision_idx >= 0) {
            lines.insert(lines.begin() + vision_idx + 1, "  snapshot:");
            lines.insert(lines.begin() + vision_idx + 2, "    min_interval_sec: " + formatted);
            updated = true;
        } else {
            lines.push_back("");
            lines.push_back("vision:");
            lines.push_back("  snapshot:");
            lines.push_back("    min_interval_sec: " + formatted);
            updated = true;
        }
    }

    std::string tmp_path = config_path + ".tmp";
    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            out_err = "无法写入临时配置文件: " + tmp_path;
            return false;
        }
        for (const auto& l : lines) {
            ofs << l << "\n";
        }
        ofs.flush();
        if (!ofs.good()) {
            out_err = "写入临时配置文件异常: " + tmp_path;
            return false;
        }
    }

    // 语法与数值校验，确保不会损坏原 config.yaml
    try {
        YAML::Node check = YAML::LoadFile(tmp_path);
        double parsed = check["vision"]["snapshot"]["min_interval_sec"].as<double>();
        if (std::abs(parsed - std::stod(formatted)) > 0.05) {
            out_err = "配置文件回读校验不一致";
            std::error_code rm_ec;
            fs::remove(tmp_path, rm_ec);
            return false;
        }
    } catch (const std::exception& e) {
        out_err = std::string("YAML 校验失败: ") + e.what();
        std::error_code rm_ec;
        fs::remove(tmp_path, rm_ec);
        return false;
    }

    std::error_code ren_ec;
    fs::rename(tmp_path, config_path, ren_ec);
    if (ren_ec) {
        out_err = "替换配置文件失败: " + ren_ec.message();
        std::error_code rm_ec;
        fs::remove(tmp_path, rm_ec);
        return false;
    }
    return true;
}

bool isSafePathSegment(const std::string& seg) {
    if (seg.empty() || seg == "." || seg == "..") return false;
    if (seg.find("..") != std::string::npos) return false;
    for (char c : seg) {
        if (c == '/' || c == '\\' || c == '\0' || c == ':') return false;
    }
    return true;
}

std::string sanitizeSubdirName(const std::string& raw, const std::string& fallback) {
    std::string out;
    for (char c : raw) {
        if (c != '/' && c != '\\' && c != '\0' && c != ':') {
            out.push_back(c);
        }
    }
    while (!out.empty() && (out.front() == '.' || out.front() == ' ')) out.erase(out.begin());
    while (!out.empty() && (out.back() == '.' || out.back() == ' ')) out.pop_back();
    if (out.empty() || out.find("..") != std::string::npos) {
        return fallback.empty() ? "陌生人" : fallback;
    }
    return out;
}

bool isImageFilename(const std::string& fname) {
    if (fname.empty() || fname[0] == '.') return false;
    auto dot = fname.rfind('.');
    if (dot == std::string::npos) return false;
    std::string ext = fname.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png");
}

std::string formatSnapshotFilenameNow() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S_")
        << std::setw(3) << std::setfill('0') << ms.count() << ".jpg";
    return oss.str();
}

bool parseFilenameTimestamp(const std::string& fname, int64_t& out_epoch_ms, std::string& out_display) {
    // 解析格式: YYYYMMDD_HHMMSS_mmm.jpg 或 YYYYMMDD_HHMMSS.jpg
    if (fname.size() < 15 || fname[8] != '_') return false;
    for (size_t i : {0,1,2,3,4,5,6,7, 9,10,11,12,13,14}) {
        if (!std::isdigit(static_cast<unsigned char>(fname[i]))) return false;
    }
    std::tm tm_buf{};
    tm_buf.tm_year = std::stoi(fname.substr(0, 4)) - 1900;
    tm_buf.tm_mon = std::stoi(fname.substr(4, 2)) - 1;
    tm_buf.tm_mday = std::stoi(fname.substr(6, 2));
    tm_buf.tm_hour = std::stoi(fname.substr(9, 2));
    tm_buf.tm_min = std::stoi(fname.substr(11, 2));
    tm_buf.tm_sec = std::stoi(fname.substr(13, 2));
    tm_buf.tm_isdst = -1;

    int millis = 0;
    if (fname.size() >= 19 && fname[15] == '_' &&
        std::isdigit(static_cast<unsigned char>(fname[16])) &&
        std::isdigit(static_cast<unsigned char>(fname[17])) &&
        std::isdigit(static_cast<unsigned char>(fname[18]))) {
        millis = std::stoi(fname.substr(16, 3));
    }

    std::time_t sec = std::mktime(&tm_buf);
    if (sec == static_cast<std::time_t>(-1)) return false;
    out_epoch_ms = static_cast<int64_t>(sec) * 1000LL + millis;

    std::ostringstream oss;
    oss << fname.substr(0, 4) << "-" << fname.substr(4, 2) << "-" << fname.substr(6, 2)
        << " " << fname.substr(9, 2) << ":" << fname.substr(11, 2) << ":" << fname.substr(13, 2);
    out_display = oss.str();
    return true;
}

struct SnapshotFileRecord {
    fs::path full_path;
    std::string subdir;
    std::string filename;
    std::string display_time;
    int64_t epoch_ms = 0;
};

std::vector<SnapshotFileRecord> collectAllSnapshotFiles(const std::string& root_dir) {
    std::vector<SnapshotFileRecord> records;
    std::error_code ec;
    if (root_dir.empty() || !fs::exists(root_dir, ec)) return records;

    for (const auto& sub_entry : fs::directory_iterator(root_dir, ec)) {
        if (!sub_entry.is_directory(ec)) continue;
        std::string sub_name = sub_entry.path().filename().string();
        if (sub_name.empty() || sub_name[0] == '.') continue;

        for (const auto& file_entry : fs::directory_iterator(sub_entry.path(), ec)) {
            if (!file_entry.is_regular_file(ec)) continue;
            std::string fname = file_entry.path().filename().string();
            if (!isImageFilename(fname)) continue;

            SnapshotFileRecord rec;
            rec.full_path = file_entry.path();
            rec.subdir = sub_name;
            rec.filename = fname;

            if (!parseFilenameTimestamp(fname, rec.epoch_ms, rec.display_time)) {
                auto ftime = file_entry.last_write_time(ec);
                auto sctp = std::chrono::time_point_cast<std::chrono::milliseconds>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                rec.epoch_ms = sctp.time_since_epoch().count();
                std::time_t tt = static_cast<std::time_t>(rec.epoch_ms / 1000);
                std::tm tm_buf{};
                localtime_r(&tt, &tm_buf);
                std::ostringstream oss;
                oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
                rec.display_time = oss.str();
            }
            records.push_back(std::move(rec));
        }
    }
    return records;
}

} // namespace

VerifyServiceCoordinator::VerifyServiceCoordinator(std::shared_ptr<CppPersonEngine> person_engine,
                                                   std::shared_ptr<CppGestureEngine> gesture_engine,
                                                   std::shared_ptr<CompositePolicy> composite_policy,
                                                   std::shared_ptr<IUnlockActuator> actuator,
                                                   const std::string& cam_source,
                                                   double active_window_sec)
    : person_engine_(std::move(person_engine)),
      gesture_engine_(std::move(gesture_engine)),
      composite_policy_(std::move(composite_policy)),
      actuator_(std::move(actuator)),
      cam_source_(cam_source),
      active_window_sec_(active_window_sec) {
    last_decision_.timestamp = currentTimeStr();
}

VerifyServiceCoordinator::~VerifyServiceCoordinator() {
    stop();
}

void VerifyServiceCoordinator::stop() {
    std::unique_lock<std::mutex> lock(mtx_);
    running_ = false;
    stop_requested_ = true;
    cv_wakeup_.notify_all();
    cv_done_.notify_all();
}

void VerifyServiceCoordinator::extendVerifyTimeoutLocked(double window_sec) {
    auto now = std::chrono::steady_clock::now();
    auto new_deadline = now + std::chrono::milliseconds(static_cast<long long>(window_sec * 1000.0));
    if (new_deadline > verify_deadline_) {
        verify_deadline_ = new_deadline;
    }
    if (!verifying_) {
        verifying_ = true;
        stop_requested_ = false;
        verify_done_ = false;
        cv_wakeup_.notify_all();
    }
}

void VerifyServiceCoordinator::pushHistoryLocked(nlohmann::json item) {
    history_.insert(history_.begin(), std::move(item));
    if (history_.size() > 30) {
        history_.resize(30);
    }
}

nlohmann::json* VerifyServiceCoordinator::findHistoryByIdLocked(uint64_t id) {
    for (auto& item : history_) {
        if (item.contains("id") && item["id"].get<uint64_t>() == id) {
            return &item;
        }
    }
    return nullptr;
}

uint64_t VerifyServiceCoordinator::recordAuditEvent(UnlockDecision decision,
                                                    const std::string& status,
                                                    const std::string& detail,
                                                    bool increment_unlock) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (increment_unlock) {
        unlock_count_++;
    }
    uint64_t eid = next_event_id_++;
    decision.status = status;
    decision.detail = detail;
    decision.timestamp = currentTimeStr();
    last_decision_ = decision;

    nlohmann::json event_item = {
        {"id", eid},
        {"time", decision.timestamp},
        {"person", decision.person},
        {"auth_mode", decision.auth_mode},
        {"score", std::round(decision.score * 1000.0f) / 1000.0f},
        {"action", decision.action},
        {"elapsed_ms", decision.elapsed_ms},
        {"relay_pulse_ms", decision.relay_pulse_ms},
        {"passed", decision.passed},
        {"status", status},
        {"detail", detail}
    };
    pushHistoryLocked(std::move(event_item));
    return eid;
}

void VerifyServiceCoordinator::resolvePendingSwitchUnlock(UnlockDecision decision, bool actuator_executed) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto now = std::chrono::steady_clock::now();

    if (!pending_switch_.has_value()) {
        uint64_t eid = next_event_id_++;
        if (actuator_executed) {
            unlock_count_++;
            decision.passed = true;
            decision.status = "unlocked";
            decision.detail = "先按开关后核验自家人通过，门锁已开启";
        } else {
            decision.passed = false;
            decision.status = "cooldown";
            decision.action = "SWITCH_COOLDOWN";
            decision.relay_pulse_ms = 0;
            decision.detail = "识别到自家人，但处于 10 秒防重触发冷却保护期";
        }
        decision.timestamp = currentTimeStr();
        last_decision_ = decision;

        nlohmann::json event_item = {
            {"id", eid},
            {"time", decision.timestamp},
            {"person", decision.person},
            {"auth_mode", decision.auth_mode},
            {"score", std::round(decision.score * 1000.0f) / 1000.0f},
            {"action", decision.action},
            {"elapsed_ms", decision.elapsed_ms},
            {"relay_pulse_ms", decision.relay_pulse_ms},
            {"passed", decision.passed},
            {"status", decision.status},
            {"detail", decision.detail}
        };
        pushHistoryLocked(std::move(event_item));
        return;
    }

    int total_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        now - pending_switch_->pressed_at).count());
    uint64_t eid = pending_switch_->event_id;
    pending_switch_.reset();

    decision.elapsed_ms = std::max(1, total_ms);
    decision.timestamp = currentTimeStr();
    if (actuator_executed) {
        unlock_count_++;
        decision.passed = true;
        decision.status = "unlocked";
        decision.detail = "按键后 " + formatOneDecimal(total_ms / 1000.0) + "s 识别到自家人，门锁已开启";
    } else {
        decision.passed = false;
        decision.status = "cooldown";
        decision.action = "SWITCH_COOLDOWN";
        decision.relay_pulse_ms = 0;
        decision.detail = "识别到自家人，但处于 10 秒防重触发冷却保护期";
    }
    last_decision_ = decision;

    if (auto* item = findHistoryByIdLocked(eid)) {
        (*item)["time"] = decision.timestamp;
        (*item)["person"] = decision.person;
        (*item)["auth_mode"] = decision.auth_mode;
        (*item)["score"] = std::round(decision.score * 1000.0f) / 1000.0f;
        (*item)["action"] = decision.action;
        (*item)["elapsed_ms"] = decision.elapsed_ms;
        (*item)["relay_pulse_ms"] = decision.relay_pulse_ms;
        (*item)["passed"] = decision.passed;
        (*item)["status"] = decision.status;
        (*item)["detail"] = decision.detail;
        (*item)["remaining_sec"] = 0.0;
    }
}

void VerifyServiceCoordinator::finalizeExpiredSwitchLocked(std::chrono::steady_clock::time_point now) {
    if (!pending_switch_.has_value()) return;

    int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        now - pending_switch_->pressed_at).count());

    if (now < pending_switch_->deadline) {
        double rem = std::chrono::duration<double>(pending_switch_->deadline - now).count();
        rem = std::max(0.0, std::round(rem * 10.0) / 10.0);
        if (auto* item = findHistoryByIdLocked(pending_switch_->event_id)) {
            (*item)["elapsed_ms"] = std::max(0, elapsed_ms);
            (*item)["remaining_sec"] = rem;
        }
        return;
    }

    int window_ms = static_cast<int>(std::round(pending_switch_->window_sec * 1000.0));
    std::string win_str = formatOneDecimal(pending_switch_->window_sec);
    std::string person_label = "未识别到自家人";
    std::string detail_msg = "按键后 " + win_str + "s 内未检测到自家人，已安全拦截";

    if (pending_switch_->saw_stranger) {
        person_label = "陌生人 (未授权)";
        detail_msg = "按键后 " + win_str + "s 内仅检测到陌生人，已拒绝开门";
    } else if (pending_switch_->camera_unavailable) {
        detail_msg = "按键后 " + win_str + "s 内未识别到自家人 (摄像头未就绪)，已拦截";
    }

    uint64_t eid = pending_switch_->event_id;
    double logged_win = pending_switch_->window_sec;
    pending_switch_.reset();

    UnlockDecision blocked_dec;
    blocked_dec.passed = false;
    blocked_dec.status = "blocked";
    blocked_dec.person = person_label;
    blocked_dec.auth_mode = "人体 + 智能开关";
    blocked_dec.score = 0.0f;
    blocked_dec.action = "SWITCH_TIMEOUT_BLOCKED";
    blocked_dec.detail = detail_msg;
    blocked_dec.elapsed_ms = window_ms;
    blocked_dec.relay_pulse_ms = 0;
    blocked_dec.timestamp = currentTimeStr();
    last_decision_ = blocked_dec;

    if (auto* item = findHistoryByIdLocked(eid)) {
        (*item)["person"] = person_label;
        (*item)["action"] = "SWITCH_TIMEOUT_BLOCKED";
        (*item)["elapsed_ms"] = window_ms;
        (*item)["passed"] = false;
        (*item)["status"] = "blocked";
        (*item)["detail"] = detail_msg;
        (*item)["remaining_sec"] = 0.0;
    }

    std::cout << "🛡️ [门禁安全拦截] 门厅开关按下后 " << logged_win
              << "s 内未核验到自家人，已记入审计流水 (" << detail_msg << ")。" << std::endl;
}

nlohmann::json VerifyServiceCoordinator::handleSmartSwitchTrigger() {
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        finalizeExpiredSwitchLocked(now);
    }

    AuthEvent event;
    event.type = AuthEventType::SwitchPressed;
    event.timestamp = now;
    event.elapsed_ms = 10;

    auto decision = composite_policy_->evaluateAll(event);
    double switch_win = 5.0;
    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) switch_win = sw_policy->getWindowSec();

    if (decision.has_value() && decision->passed) {
        // 情况 A: 开关按下前 N 秒内人脸已核验成功
        bool executed = actuator_->executeUnlock(*decision);
        if (executed) {
            uint64_t eid = recordAuditEvent(*decision, "unlocked", "自家人在场有效期内按键，门锁即刻开启", true);
            return nlohmann::json{
                {"status", "unlocked"},
                {"passed", true},
                {"event_id", eid},
                {"auth_mode", decision->auth_mode},
                {"person", decision->person},
                {"window_sec", switch_win},
                {"message", "开关按下前 " + formatOneDecimal(switch_win) + " 秒内已识别自家人【" + decision->person + "】，门锁已打开！"}
            };
        } else {
            double cd_rem = actuator_->getCooldownRemainingSec();
            UnlockDecision cd_dec = *decision;
            cd_dec.passed = false;
            cd_dec.action = "SWITCH_COOLDOWN";
            cd_dec.relay_pulse_ms = 0;
            uint64_t eid = recordAuditEvent(cd_dec, "cooldown", "自家人在场按键，处于 10 秒冷却防抖保护期", false);
            return nlohmann::json{
                {"status", "cooldown"},
                {"passed", false},
                {"event_id", eid},
                {"auth_mode", decision->auth_mode},
                {"person", decision->person},
                {"cooldown_remaining_sec", cd_rem},
                {"window_sec", switch_win},
                {"message", "已识别到自家人【" + decision->person + "】，但门锁处于冷却保护期（剩余 " + formatOneDecimal(cd_rem) + " 秒），已记入流水"}
            };
        }
    } else {
        // 情况 B: 开关按下前无人脸，立即写入“等待核验”流水并唤醒视频拉流互锁窗口期
        uint64_t eid = 0;
        std::string win_str = formatOneDecimal(switch_win);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (pending_switch_.has_value()) {
                int prev_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - pending_switch_->pressed_at).count());
                if (auto* prev_item = findHistoryByIdLocked(pending_switch_->event_id)) {
                    (*prev_item)["person"] = "连续按键刷新";
                    (*prev_item)["action"] = "SWITCH_SUPERSEDED";
                    (*prev_item)["elapsed_ms"] = std::max(1, prev_ms);
                    (*prev_item)["passed"] = false;
                    (*prev_item)["status"] = "superseded";
                    (*prev_item)["detail"] = "等待核验期间再次按键，已刷新 " + win_str + "s 互锁窗口";
                    (*prev_item)["remaining_sec"] = 0.0;
                }
                pending_switch_.reset();
            }

            eid = next_event_id_++;
            std::string ts = currentTimeStr();

            PendingSwitchAudit pending;
            pending.event_id = eid;
            pending.pressed_at = now;
            pending.deadline = now + std::chrono::milliseconds(static_cast<long long>(switch_win * 1000.0));
            pending.window_sec = switch_win;
            pending_switch_ = pending;

            nlohmann::json pending_item = {
                {"id", eid},
                {"time", ts},
                {"person", "等待自家人核验..."},
                {"auth_mode", "人体 + 智能开关"},
                {"score", 0.0},
                {"action", "SWITCH_WAITING"},
                {"elapsed_ms", 0},
                {"relay_pulse_ms", 0},
                {"passed", false},
                {"status", "pending"},
                {"detail", "门厅开关已按下，正在等待 " + win_str + "s 内识别自家人"},
                {"remaining_sec", switch_win}
            };
            pushHistoryLocked(std::move(pending_item));

            last_decision_.passed = false;
            last_decision_.status = "pending";
            last_decision_.person = "等待自家人核验...";
            last_decision_.auth_mode = "人体 + 智能开关";
            last_decision_.score = 0.0f;
            last_decision_.action = "SWITCH_WAITING";
            last_decision_.detail = "门厅开关已按下，正在等待 " + win_str + "s 内识别自家人";
            last_decision_.elapsed_ms = 0;
            last_decision_.relay_pulse_ms = 0;
            last_decision_.timestamp = ts;

            extendVerifyTimeoutLocked(switch_win + 0.6);
        }

        std::cout << "🔘 [门厅智能开关] 收到按键信号 (#" << eid << ")，已记入审计流水，开启 "
                  << switch_win << "s 自家人核验窗口..." << std::endl;

        return nlohmann::json{
            {"status", "waiting_person_match"},
            {"passed", false},
            {"event_id", eid},
            {"auth_mode", "人体 + 智能开关"},
            {"window_sec", switch_win},
            {"message", "门厅开关已按下（已记入审计流水），正在等待 " + win_str + " 秒内自家人核验..."}
        };
    }
}

nlohmann::json VerifyServiceCoordinator::handleSmartSwitchVerifyQuery() {
    auto now = std::chrono::steady_clock::now();
    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) {
        bool valid = sw_policy->isFamilyPresenceActive(now);
        return nlohmann::json{
            {"can_unlock", valid},
            {"person", sw_policy->getLastVerifiedFamily()},
            {"window_sec", sw_policy->getWindowSec()},
            {"remaining_sec", sw_policy->getPresenceRemainingSec(now)}
        };
    }
    return nlohmann::json{{"can_unlock", false}, {"person", "None"}, {"remaining_sec", 0.0}};
}

nlohmann::json VerifyServiceCoordinator::handleVoiceVerify(const nlohmann::json& req) {
    AuthEvent event;
    event.type = AuthEventType::VoiceCommand;
    event.timestamp = std::chrono::steady_clock::now();
    event.voice_text = req.value("text", "");

    auto decision = composite_policy_->evaluateAll(event);
    if (decision.has_value() && decision->passed) {
        bool executed = actuator_->executeUnlock(*decision);
        recordAuditEvent(*decision,
                         executed ? "unlocked" : "cooldown",
                         executed ? "声纹与通行口令核验通过" : "声纹核验通过，但处于冷却保护期",
                         executed);
        return nlohmann::json{{"status", executed ? "unlocked" : "cooldown"}, {"passed", executed}, {"person", decision->person}};
    }
    return nlohmann::json{
        {"status", "reserved"},
        {"passed", false},
        {"auth_mode", "声音核验"},
        {"message", "声纹比对与 ASR 说话内容识别模块接口已就绪，等待后续音频模型接入"}
    };
}

nlohmann::json VerifyServiceCoordinator::testSingleImage(const std::string& b64_img) {
    nlohmann::json pred = person_engine_->predictBase64(b64_img);
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        finalizeExpiredSwitchLocked(now);
        if (pending_switch_.has_value() && pred.value("stranger_count", 0) > 0) {
            pending_switch_->saw_stranger = true;
        }
    }

    if (pred.contains("family_count") && pred["family_count"].get<int>() > 0) {
        AuthEvent event;
        event.type = AuthEventType::FrameRecognized;
        event.timestamp = now;
        event.elapsed_ms = 15;
        event.frame_result.family_count = pred.value("family_count", 0);
        event.frame_result.stranger_count = pred.value("stranger_count", 0);
        if (pred.contains("family_members")) {
            for (const auto& m : pred["family_members"]) {
                event.frame_result.family_members.push_back(m.get<std::string>());
            }
        }

        auto decision = composite_policy_->evaluateAll(event);
        if (decision.has_value() && decision->passed) {
            bool executed = actuator_->executeUnlock(*decision);
            if (decision->action == "SWITCH_PREV_CLICKED") {
                resolvePendingSwitchUnlock(*decision, executed);
            } else {
                recordAuditEvent(*decision,
                                 executed ? "unlocked" : "cooldown",
                                 executed ? "图像核验联动开锁成功" : "核验通过，但处于冷却保护期",
                                 executed);
            }
            pred["switch_interlock_unlocked"] = executed;
        }
    }
    return pred;
}

nlohmann::json VerifyServiceCoordinator::triggerManualRelayTest() {
    UnlockDecision decision;
    decision.passed = true;
    decision.person = "Web手动测试";
    decision.auth_mode = "手动模拟";
    decision.score = 1.0f;
    decision.action = "MANUAL_BUTTON";
    decision.elapsed_ms = 12;
    decision.relay_pulse_ms = 500;

    bool executed = actuator_->executeUnlock(decision);
    if (executed) {
        recordAuditEvent(decision, "unlocked", "控制台手动触发应急点动开锁", true);
        return nlohmann::json{
            {"status", "ok"},
            {"passed", true},
            {"message", "已触发 ESP32-C6 光耦 500ms 应急开锁脉冲"}
        };
    } else {
        double cd_rem = actuator_->getCooldownRemainingSec();
        decision.passed = false;
        decision.action = "COOLDOWN_PROTECTED";
        decision.relay_pulse_ms = 0;
        recordAuditEvent(decision, "cooldown", "应急点动触发防重保护（10s冷却期内）", false);
        return nlohmann::json{
            {"status", "cooldown"},
            {"passed", false},
            {"cooldown_remaining_sec", cd_rem},
            {"message", "门锁处于 10 秒防重触发冷却期（剩余 " + formatOneDecimal(cd_rem) + " 秒），已记入审计流水"}
        };
    }
}

nlohmann::json VerifyServiceCoordinator::requestVerifySync(double custom_timeout_sec) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!verifying_) {
        double timeout = (custom_timeout_sec > 0.0) ? custom_timeout_sec : active_window_sec_;
        extendVerifyTimeoutLocked(timeout);
    }
    cv_done_.wait(lock, [this]() { return verify_done_ || !running_; });
    return nlohmann::json{
        {"passed", last_decision_.passed},
        {"status", last_decision_.status},
        {"person", last_decision_.person},
        {"auth_mode", last_decision_.auth_mode},
        {"score", last_decision_.score},
        {"action", last_decision_.action},
        {"detail", last_decision_.detail},
        {"elapsed_ms", last_decision_.elapsed_ms},
        {"timestamp", last_decision_.timestamp}
    };
}

nlohmann::json VerifyServiceCoordinator::requestVerifyAsync() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (verifying_) {
        return nlohmann::json{{"status", "already_verifying"}, {"message", "服务已在拉流核验中"}};
    }
    extendVerifyTimeoutLocked(active_window_sec_);
    return nlohmann::json{{"status", "started"}, {"window_sec", active_window_sec_}};
}

nlohmann::json VerifyServiceCoordinator::requestStopVerify() {
    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) {
        sw_policy->clearSwitchWait();
    }

    std::unique_lock<std::mutex> lock(mtx_);
    if (pending_switch_.has_value()) {
        auto now = std::chrono::steady_clock::now();
        int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - pending_switch_->pressed_at).count());
        if (auto* item = findHistoryByIdLocked(pending_switch_->event_id)) {
            (*item)["person"] = "手动中止";
            (*item)["action"] = "SWITCH_CANCELLED";
            (*item)["elapsed_ms"] = std::max(1, elapsed_ms);
            (*item)["passed"] = false;
            (*item)["status"] = "cancelled";
            (*item)["detail"] = "手动点击强制休眠，已中止本次按键核验";
            (*item)["remaining_sec"] = 0.0;
        }
        last_decision_.passed = false;
        last_decision_.status = "cancelled";
        last_decision_.person = "手动中止";
        last_decision_.auth_mode = "人体 + 智能开关";
        last_decision_.action = "SWITCH_CANCELLED";
        last_decision_.detail = "手动点击强制休眠，已中止本次按键核验";
        last_decision_.elapsed_ms = std::max(1, elapsed_ms);
        last_decision_.timestamp = currentTimeStr();
        pending_switch_.reset();
    }

    if (verifying_) {
        stop_requested_ = true;
        return nlohmann::json{{"status", "stopping"}};
    }
    return nlohmann::json{{"status", "already_idle"}};
}

nlohmann::json VerifyServiceCoordinator::getStatus() {
    std::unique_lock<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now();
    finalizeExpiredSwitchLocked(now);

    bool relay_on = actuator_->isRelayActive();
    double cooldown_rem = actuator_->getCooldownRemainingSec();

    bool presence_active = false;
    double presence_rem = 0.0;
    std::string last_family = "None";
    double switch_win = 5.0;
    bool switch_waiting = false;
    double switch_waiting_rem = 0.0;

    auto sw_policy = composite_policy_->getPolicy<SwitchWindowPolicy>();
    if (sw_policy) {
        presence_active = sw_policy->isFamilyPresenceActive(now);
        presence_rem = sw_policy->getPresenceRemainingSec(now);
        last_family = sw_policy->getLastVerifiedFamily();
        switch_win = sw_policy->getWindowSec();
        switch_waiting = sw_policy->isWaitingForPerson(now);
        switch_waiting_rem = sw_policy->getWaitingRemainingSec(now);
    }

    size_t snap_count = 0;
    bool snap_enabled = true;
    double snap_interval = 1.0;
    std::string snap_root;
    {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        snap_count = cached_snapshot_count_;
        snap_enabled = snapshot_cfg_.enabled;
        snap_interval = snapshot_cfg_.min_interval_sec;
        snap_root = snapshot_cfg_.root_dir;
    }

    return nlohmann::json{
        {"service", "myhome_cpp_verify_server"},
        {"state", verifying_ ? "VERIFYING_STREAM" : "IDLE_SLEEP"},
        {"model_engine", "LightPersonIDNet (MobileNetV3 + GeM)"},
        {"supported_modes", composite_policy_->getSupportedModeNames()},
        {"remaining_sec", verifying_ ? std::round(remaining_sec_ * 10.0) / 10.0 : 0.0},
        {"window_sec", active_window_sec_},
        {"switch_face_window_sec", switch_win},
        {"enrolled_members", person_engine_->getEnrolledNames()},
        {"live_person", live_person_},
        {"live_score", std::round(live_score_ * 1000.0f) / 1000.0f},
        {"live_has_stranger", live_has_stranger_},
        {"family_presence_active", presence_active},
        {"presence_remaining_sec", presence_rem},
        {"last_verified_family", last_family},
        {"switch_waiting_active", switch_waiting},
        {"switch_waiting_remaining_sec", switch_waiting_rem},
        {"cooldown_active", cooldown_rem > 0.0},
        {"cooldown_remaining_sec", cooldown_rem},
        {"snapshot_enabled", snap_enabled},
        {"snapshot_min_interval_sec", snap_interval},
        {"snapshot_root_dir", snap_root},
        {"snapshot_count", snap_count},
        {"relay_active", relay_on},
        {"unlock_count", unlock_count_.load()},
        {"last_result", {
            {"passed", last_decision_.passed},
            {"status", last_decision_.status},
            {"person", last_decision_.person},
            {"auth_mode", last_decision_.auth_mode},
            {"score", last_decision_.score},
            {"action", last_decision_.action},
            {"detail", last_decision_.detail},
            {"elapsed_ms", last_decision_.elapsed_ms},
            {"timestamp", last_decision_.timestamp}
        }},
        {"history", history_}
    };
}

std::vector<uchar> VerifyServiceCoordinator::getLatestJpeg() {
    std::lock_guard<std::mutex> lock(frame_mtx_);
    return latest_jpeg_;
}

nlohmann::json VerifyServiceCoordinator::getMembersSummary() {
    return person_engine_->getMembersSummary();
}

std::vector<uchar> VerifyServiceCoordinator::getPhotoBinary(const std::string& filename) {
    return person_engine_->getPhotoBinary(filename);
}

nlohmann::json VerifyServiceCoordinator::uploadPhoto(const std::string& name, const std::string& base64_image) {
    auto res = person_engine_->uploadPhoto(name, base64_image);
    if (res.value("status", "") == "success") {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        std::error_code ec;
        std::string safe_sub = sanitizeSubdirName(name, snapshot_cfg_.stranger_subdir);
        fs::create_directories(fs::path(snapshot_cfg_.root_dir) / safe_sub, ec);
    }
    return res;
}

nlohmann::json VerifyServiceCoordinator::deletePhoto(const std::string& filename) {
    return person_engine_->deletePhoto(filename);
}

nlohmann::json VerifyServiceCoordinator::reloadGallery() {
    auto res = person_engine_->reloadGallery();
    {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        ensureSnapshotSubdirsLocked();
    }
    return res;
}

void VerifyServiceCoordinator::ensureSnapshotSubdirsLocked() {
    if (snapshot_cfg_.root_dir.empty()) return;
    std::error_code ec;
    fs::path root(snapshot_cfg_.root_dir);
    fs::create_directories(root, ec);

    std::string stranger_dir = sanitizeSubdirName(snapshot_cfg_.stranger_subdir, "陌生人");
    snapshot_cfg_.stranger_subdir = stranger_dir;
    fs::create_directories(root / stranger_dir, ec);

    for (const auto& member : person_engine_->getEnrolledNames()) {
        std::string sub = sanitizeSubdirName(member, stranger_dir);
        fs::create_directories(root / sub, ec);
    }
}

size_t VerifyServiceCoordinator::scanSnapshotCountLocked() const {
    return collectAllSnapshotFiles(snapshot_cfg_.root_dir).size();
}

void VerifyServiceCoordinator::enforceSnapshotRetentionLocked() {
    auto records = collectAllSnapshotFiles(snapshot_cfg_.root_dir);
    cached_snapshot_count_ = records.size();
    if (snapshot_cfg_.max_total == 0 || records.size() <= snapshot_cfg_.max_total) {
        return;
    }

    // 按时间升序排列（最早保存的排在最前）
    std::sort(records.begin(), records.end(), [](const SnapshotFileRecord& a, const SnapshotFileRecord& b) {
        if (a.epoch_ms != b.epoch_ms) return a.epoch_ms < b.epoch_ms;
        return a.filename < b.filename;
    });

    size_t batch = std::max<size_t>(1, snapshot_cfg_.cleanup_batch);
    size_t overflow = records.size() - snapshot_cfg_.max_total;
    size_t to_delete = std::min(records.size(), std::max(batch, overflow));

    size_t deleted = 0;
    for (size_t i = 0; i < to_delete; ++i) {
        std::error_code ec;
        if (fs::remove(records[i].full_path, ec)) {
            deleted++;
        }
    }
    cached_snapshot_count_ = records.size() - deleted;
    std::cout << "🧹 [快照容量轮转] 快照总数超过 " << snapshot_cfg_.max_total
              << " 张，已一次性删除最早的 " << deleted
              << " 张快照，当前剩余: " << cached_snapshot_count_ << " 张。" << std::endl;
}

void VerifyServiceCoordinator::configureSnapshots(const SnapshotConfig& cfg) {
    std::lock_guard<std::mutex> slk(snapshot_mtx_);
    snapshot_cfg_ = cfg;
    if (snapshot_cfg_.min_interval_sec < 0.1) {
        snapshot_cfg_.min_interval_sec = 1.0;
    }
    if (snapshot_cfg_.config_file_path.empty()) {
        snapshot_cfg_.config_file_path = "./config.yaml";
    }
    ensureSnapshotSubdirsLocked();
    enforceSnapshotRetentionLocked();
    std::cout << "📸 [人脸自动快照] 已就绪: 根目录=" << snapshot_cfg_.root_dir
              << ", 陌生人子目录=" << snapshot_cfg_.stranger_subdir
              << ", 最小间隔=" << snapshot_cfg_.min_interval_sec << "s"
              << ", 容量上限=" << snapshot_cfg_.max_total
              << "张 (超限单次清理最早 " << snapshot_cfg_.cleanup_batch << " 张)"
              << ", 当前已有=" << cached_snapshot_count_ << " 张" << std::endl;
}

nlohmann::json VerifyServiceCoordinator::updateSnapshotSettings(const nlohmann::json& req) {
    if (!req.contains("min_interval_sec")) {
        return nlohmann::json{{"status", "error"}, {"error", "缺少参数 min_interval_sec"}};
    }

    double raw_val = 0.0;
    try {
        if (req["min_interval_sec"].is_number()) {
            raw_val = req["min_interval_sec"].get<double>();
        } else if (req["min_interval_sec"].is_string()) {
            raw_val = std::stod(req["min_interval_sec"].get<std::string>());
        } else {
            return nlohmann::json{{"status", "error"}, {"error", "min_interval_sec 格式不正确"}};
        }
    } catch (...) {
        return nlohmann::json{{"status", "error"}, {"error", "min_interval_sec 数值无效"}};
    }

    if (!std::isfinite(raw_val) || raw_val < 0.2 || raw_val > 3600.0) {
        return nlohmann::json{
            {"status", "error"},
            {"error", "抓拍间隔需在 0.2 ~ 3600 秒之间"}
        };
    }

    double normalized = std::round(raw_val * 100.0) / 100.0;

    std::lock_guard<std::mutex> slk(snapshot_mtx_);
    std::string cfg_path = snapshot_cfg_.config_file_path.empty() ? "./config.yaml" : snapshot_cfg_.config_file_path;
    std::string err_msg;
    if (!persistSnapshotIntervalToYaml(cfg_path, normalized, err_msg)) {
        return nlohmann::json{
            {"status", "error"},
            {"error", err_msg}
        };
    }

    snapshot_cfg_.min_interval_sec = normalized;
    std::cout << "⚙️ [人脸自动快照] 抓拍最小间隔已更新为 " << formatIntervalValue(normalized)
              << " 秒，并已持久化保存至 " << cfg_path << std::endl;

    return nlohmann::json{
        {"status", "success"},
        {"min_interval_sec", normalized},
        {"config_file", cfg_path},
        {"message", "抓拍最小间隔已更新为 " + formatIntervalValue(normalized) + " 秒，并已保存至 config.yaml"}
    };
}

void VerifyServiceCoordinator::maybeSaveFaceSnapshot(const cv::Mat& frame, const FrameRecognitionResult& frame_res) {
    if (frame.empty() || frame_res.face_count <= 0) return;

    auto steady_now = std::chrono::steady_clock::now();
    std::string root_dir;
    std::string stranger_dir;
    {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        if (!snapshot_cfg_.enabled || snapshot_cfg_.root_dir.empty()) return;
        if (last_snapshot_steady_.time_since_epoch().count() > 0) {
            double elapsed = std::chrono::duration<double>(steady_now - last_snapshot_steady_).count();
            if (elapsed < snapshot_cfg_.min_interval_sec) {
                return; // 距离上次快照未满最小间隔（1.0s），跳过保存
            }
        }
        last_snapshot_steady_ = steady_now;
        root_dir = snapshot_cfg_.root_dir;
        stranger_dir = snapshot_cfg_.stranger_subdir;
    }

    // 将检测到的人脸与画面中识别出的人物关联，决定归档到哪个子目录
    std::vector<std::string> target_subdirs;
    auto addUniqueSubdir = [&target_subdirs](const std::string& dir_name) {
        if (std::find(target_subdirs.begin(), target_subdirs.end(), dir_name) == target_subdirs.end()) {
            target_subdirs.push_back(dir_name);
        }
    };

    for (const auto& face : frame_res.faces) {
        cv::Point face_center(face.bbox.x + face.bbox.width / 2,
                              face.bbox.y + face.bbox.height / 2);

        const PersonDetail* matched_person = nullptr;
        double best_dist_sq = 1e18;

        for (const auto& p : frame_res.persons) {
            int pad_x = std::max(12, static_cast<int>(p.bbox.width * 0.15f));
            int pad_y = std::max(12, static_cast<int>(p.bbox.height * 0.15f));
            cv::Rect expanded(p.bbox.x - pad_x, p.bbox.y - pad_y,
                              p.bbox.width + 2 * pad_x, p.bbox.height + 2 * pad_y);
            if (expanded.contains(face_center)) {
                double cx = p.bbox.x + p.bbox.width * 0.5;
                double cy = p.bbox.y + p.bbox.height * 0.25; // 头部约在人体框上方 1/4 处
                double d2 = (face_center.x - cx) * (face_center.x - cx) +
                            (face_center.y - cy) * (face_center.y - cy);
                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    matched_person = &p;
                }
            }
        }

        if (!matched_person && frame_res.persons.size() == 1) {
            matched_person = &frame_res.persons[0];
        }

        if (matched_person && matched_person->is_family) {
            addUniqueSubdir(sanitizeSubdirName(matched_person->name, stranger_dir));
        } else {
            addUniqueSubdir(stranger_dir);
        }
    }

    if (target_subdirs.empty()) {
        addUniqueSubdir(stranger_dir);
    }

    std::vector<uchar> jpg_buf;
    if (!cv::imencode(".jpg", frame, jpg_buf, {cv::IMWRITE_JPEG_QUALITY, 90}) || jpg_buf.empty()) {
        return;
    }

    std::string fname = formatSnapshotFilenameNow();
    {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        std::error_code ec;
        for (const auto& sub : target_subdirs) {
            fs::path sub_path = fs::path(snapshot_cfg_.root_dir) / sub;
            fs::create_directories(sub_path, ec);
            fs::path file_path = sub_path / fname;

            std::ofstream ofs(file_path, std::ios::binary);
            if (ofs) {
                ofs.write(reinterpret_cast<const char*>(jpg_buf.data()), static_cast<std::streamsize>(jpg_buf.size()));
                ofs.close();
                cached_snapshot_count_++;
                std::cout << "📸 [人脸自动快照] 已保存 -> " << file_path.string()
                          << " (人脸数: " << frame_res.face_count << ")" << std::endl;
            }
        }

        if (snapshot_cfg_.max_total > 0 && cached_snapshot_count_ > snapshot_cfg_.max_total) {
            enforceSnapshotRetentionLocked();
        }
    }
}

nlohmann::json VerifyServiceCoordinator::getSnapshotsSummary() {
    std::lock_guard<std::mutex> slk(snapshot_mtx_);
    ensureSnapshotSubdirsLocked();
    auto records = collectAllSnapshotFiles(snapshot_cfg_.root_dir);
    cached_snapshot_count_ = records.size();

    // 按时间倒序排列（最新抓拍的排在最前面供前端展示）
    std::sort(records.begin(), records.end(), [](const SnapshotFileRecord& a, const SnapshotFileRecord& b) {
        if (a.epoch_ms != b.epoch_ms) return a.epoch_ms > b.epoch_ms;
        return a.filename > b.filename;
    });

    // 统计各子文件夹数量（包含尚无照片的成员文件夹与陌生人文件夹）
    std::vector<std::string> folder_order;
    for (const auto& m : person_engine_->getEnrolledNames()) {
        std::string s = sanitizeSubdirName(m, snapshot_cfg_.stranger_subdir);
        if (std::find(folder_order.begin(), folder_order.end(), s) == folder_order.end()) {
            folder_order.push_back(s);
        }
    }
    if (std::find(folder_order.begin(), folder_order.end(), snapshot_cfg_.stranger_subdir) == folder_order.end()) {
        folder_order.push_back(snapshot_cfg_.stranger_subdir);
    }

    std::error_code ec;
    if (fs::exists(snapshot_cfg_.root_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(snapshot_cfg_.root_dir, ec)) {
            if (entry.is_directory(ec)) {
                std::string dname = entry.path().filename().string();
                if (!dname.empty() && dname[0] != '.' &&
                    std::find(folder_order.begin(), folder_order.end(), dname) == folder_order.end()) {
                    folder_order.push_back(dname);
                }
            }
        }
    }

    nlohmann::json folders_json = nlohmann::json::array();
    for (const auto& fname : folder_order) {
        size_t cnt = 0;
        for (const auto& r : records) {
            if (r.subdir == fname) cnt++;
        }
        folders_json.push_back({
            {"name", fname},
            {"is_family", fname != snapshot_cfg_.stranger_subdir},
            {"count", cnt}
        });
    }

    nlohmann::json list_json = nlohmann::json::array();
    for (const auto& r : records) {
        list_json.push_back({
            {"subdir", r.subdir},
            {"filename", r.filename},
            {"rel_path", r.subdir + "/" + r.filename},
            {"time", r.display_time},
            {"is_family", r.subdir != snapshot_cfg_.stranger_subdir}
        });
    }

    std::string abs_root = snapshot_cfg_.root_dir;
    std::error_code abs_ec;
    auto abs_p = fs::weakly_canonical(fs::path(snapshot_cfg_.root_dir), abs_ec);
    if (!abs_ec) abs_root = abs_p.string();

    return nlohmann::json{
        {"enabled", snapshot_cfg_.enabled},
        {"root_dir", snapshot_cfg_.root_dir},
        {"abs_root_dir", abs_root},
        {"stranger_subdir", snapshot_cfg_.stranger_subdir},
        {"min_interval_sec", snapshot_cfg_.min_interval_sec},
        {"max_total", snapshot_cfg_.max_total},
        {"cleanup_batch", snapshot_cfg_.cleanup_batch},
        {"total", records.size()},
        {"folders", folders_json},
        {"snapshots", list_json}
    };
}

std::vector<uchar> VerifyServiceCoordinator::getSnapshotBinary(const std::string& subdir, const std::string& filename) {
    std::vector<uchar> data;
    if (!isSafePathSegment(subdir) || !isSafePathSegment(filename) || !isImageFilename(filename)) {
        return data;
    }

    std::string root_dir;
    {
        std::lock_guard<std::mutex> slk(snapshot_mtx_);
        root_dir = snapshot_cfg_.root_dir;
    }
    fs::path target = fs::path(root_dir) / subdir / filename;
    std::error_code ec;
    if (!fs::is_regular_file(target, ec)) return data;

    std::ifstream ifs(target, std::ios::binary);
    if (!ifs) return data;
    return std::vector<uchar>((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
}

nlohmann::json VerifyServiceCoordinator::deleteSnapshot(const std::string& subdir,
                                                        const std::string& filename,
                                                        bool clear_subdir,
                                                        bool clear_all) {
    std::lock_guard<std::mutex> slk(snapshot_mtx_);
    size_t deleted = 0;
    std::error_code ec;

    if (clear_all) {
        auto records = collectAllSnapshotFiles(snapshot_cfg_.root_dir);
        for (const auto& r : records) {
            if (fs::remove(r.full_path, ec)) deleted++;
        }
        ensureSnapshotSubdirsLocked();
        cached_snapshot_count_ = scanSnapshotCountLocked();
        return nlohmann::json{{"status", "success"}, {"deleted_count", deleted}, {"total", cached_snapshot_count_}};
    }

    if (clear_subdir) {
        if (!isSafePathSegment(subdir)) {
            return nlohmann::json{{"error", "Invalid subdir"}};
        }
        fs::path sub_path = fs::path(snapshot_cfg_.root_dir) / subdir;
        if (fs::is_directory(sub_path, ec)) {
            for (const auto& entry : fs::directory_iterator(sub_path, ec)) {
                if (entry.is_regular_file(ec) && isImageFilename(entry.path().filename().string())) {
                    if (fs::remove(entry.path(), ec)) deleted++;
                }
            }
        }
        cached_snapshot_count_ = scanSnapshotCountLocked();
        return nlohmann::json{{"status", "success"}, {"deleted_count", deleted}, {"total", cached_snapshot_count_}};
    }

    if (!isSafePathSegment(subdir) || !isSafePathSegment(filename)) {
        return nlohmann::json{{"error", "Invalid subdir or filename"}};
    }
    fs::path target = fs::path(snapshot_cfg_.root_dir) / subdir / filename;
    if (fs::is_regular_file(target, ec) && fs::remove(target, ec)) {
        if (cached_snapshot_count_ > 0) cached_snapshot_count_--;
        return nlohmann::json{{"status", "success"}, {"deleted", subdir + "/" + filename}, {"total", cached_snapshot_count_}};
    }
    return nlohmann::json{{"error", "Snapshot file not found"}};
}

void VerifyServiceCoordinator::runMainThreadLoop() {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay", 1);

    while (running_) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            live_person_ = "None";
            live_score_ = 0.0f;
            live_has_stranger_ = false;
            remaining_sec_ = 0.0;
            cv_wakeup_.wait(lock, [this]() { return verifying_ || !running_; });
            if (!running_) break;
        }

        auto t_start = std::chrono::steady_clock::now();
        std::cout << "\n📡 [C++ 核验服务] 收到唤醒信号！正在打开摄像头视频流..." << std::endl;

        cv::VideoCapture cap;
        if (cam_source_ == "0" || cam_source_.empty()) {
            cap.open(0);
        } else {
            cap.open(cam_source_, cv::CAP_FFMPEG);
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        UnlockDecision final_decision;

        if (!cap.isOpened()) {
            std::cerr << "❌ [C++ 核验服务] 无法打开视频源: " << cam_source_ << std::endl;
            std::lock_guard<std::mutex> lk(mtx_);
            if (pending_switch_.has_value()) {
                pending_switch_->camera_unavailable = true;
            }
        } else {
            gesture_engine_->resetStreak();
            cv::Mat frame;

            while (running_) {
                auto now = std::chrono::steady_clock::now();
                double remaining_sec = 0.0;
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    finalizeExpiredSwitchLocked(now);
                    if (stop_requested_) {
                        std::cout << "⏹️ [C++ 核验服务] 收到手动停止指令，释放摄像头。" << std::endl;
                        break;
                    }
                    if (now >= verify_deadline_) {
                        std::cout << "⏱️ [C++ 核验服务] 窗口期结束，关闭视频流回到 0% CPU 休眠态。" << std::endl;
                        break;
                    }
                    remaining_sec = std::chrono::duration<double>(verify_deadline_ - now).count();
                    remaining_sec_ = remaining_sec;
                }

                if (!cap.read(frame) || frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                    continue;
                }

                // 1. 高性能纯净视觉感知 (无无用 JPEG 渲染，节约 90% CPU)
                FrameRecognitionResult frame_res;
                std::string gesture_name = "NONE";
                try {
                    frame_res = person_engine_->detectAndRecognize(frame);
                    auto gesture_res = gesture_engine_->detectGesture(frame);
                    gesture_name = gesture_res.value_or("NONE");
                } catch (const std::exception& e) {
                    std::cerr << "⚠️ [C++ 核验服务] 视觉帧推理异常: " << e.what() << std::endl;
                }
                int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count());

                // 2. 人脸自动快照：只要检测到人脸（不论自家人或陌生人）均分目录保存干净原图，且严格保证间隔 >= 1s
                if (frame_res.face_count > 0) {
                    maybeSaveFaceSnapshot(frame, frame_res);
                }

                // 更新状态缓存
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    live_has_stranger_ = (frame_res.stranger_count > 0);
                    if (pending_switch_.has_value() && live_has_stranger_) {
                        pending_switch_->saw_stranger = true;
                    }
                    if (frame_res.family_count > 0 && !frame_res.family_members.empty()) {
                        live_person_ = frame_res.family_members[0];
                        live_score_ = 0.90f;
                    } else {
                        live_person_ = live_has_stranger_ ? "Stranger" : "None";
                        live_score_ = 0.0f;
                    }
                }

                // 3. 构建领域事件并由策略组合器统一评估 (遵循 Strategy + OCP)
                AuthEvent event;
                event.type = AuthEventType::FrameRecognized;
                event.timestamp = now;
                event.frame_result = frame_res;
                event.gesture_name = gesture_name;
                event.elapsed_ms = elapsed_ms;

                auto decision = composite_policy_->evaluateAll(event);
                if (decision.has_value() && decision->passed) {
                    final_decision = *decision;
                    bool executed = actuator_->executeUnlock(final_decision);

                    if (final_decision.action == "SWITCH_PREV_CLICKED") {
                        resolvePendingSwitchUnlock(final_decision, executed);
                        if (executed) {
                            cv::rectangle(frame, cv::Point(15, 15), cv::Point(780, 72), cv::Scalar(0, 180, 60), -1);
                            std::string banner = "UNLOCKED! [" + final_decision.person + " + " + final_decision.action + "] (" +
                                                 std::to_string(elapsed_ms) + "ms)";
                            cv::putText(frame, banner, cv::Point(28, 54), cv::FONT_HERSHEY_SIMPLEX,
                                        0.85, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                            std::vector<uchar> buf;
                            cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
                            std::lock_guard<std::mutex> flk(frame_mtx_);
                            latest_jpeg_ = std::move(buf);
                        }
                        break;
                    } else if (executed) {
                        // 手势等连续帧策略：仅在实际触发开门时记录并关流，避免冷却期按帧刷屏
                        cv::rectangle(frame, cv::Point(15, 15), cv::Point(780, 72), cv::Scalar(0, 180, 60), -1);
                        std::string banner = "UNLOCKED! [" + final_decision.person + " + " + final_decision.action + "] (" +
                                             std::to_string(elapsed_ms) + "ms)";
                        cv::putText(frame, banner, cv::Point(28, 54), cv::FONT_HERSHEY_SIMPLEX,
                                    0.85, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                        {
                            std::vector<uchar> buf;
                            cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
                            std::lock_guard<std::mutex> flk(frame_mtx_);
                            latest_jpeg_ = std::move(buf);
                        }
                        recordAuditEvent(final_decision, "unlocked", "门前手势多帧确认通过，门锁已开启", true);
                        break;
                    }
                }

                // 编码最新帧供前端实时预览 (/api/frame.jpg)
                {
                    std::vector<uchar> buf;
                    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 75});
                    std::lock_guard<std::mutex> flk(frame_mtx_);
                    latest_jpeg_ = std::move(buf);
                }
            }
            cap.release();
        }

        {
            std::unique_lock<std::mutex> lock(mtx_);
            finalizeExpiredSwitchLocked(std::chrono::steady_clock::now());
            verifying_ = false;
            stop_requested_ = false;
            remaining_sec_ = 0.0;
            verify_done_ = true;
            cv_done_.notify_all();
        }
    }
}

} // namespace myhome
