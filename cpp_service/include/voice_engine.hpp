#pragma once

#include "domain_types.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace myhome {

struct VoiceEngineConfig {
    bool enabled = true;
    std::string stream_source = "auto";
    std::string db_dir = "./audio/db/voices";
    int max_members = 10;
    std::string kws_model_path = "./audio/models/command_kws_mdtc.onnx";
    std::string sv_model_path = "./audio/models/speaker_campp_lite.onnx";
    std::string commands_meta_path = "./audio/models/commands.json";
    float command_threshold = 0.55f;
    float similarity_threshold = 0.60f;
    float vad_energy_threshold = 0.015f;
    std::vector<std::string> unlock_commands = {
        "芝麻开门", "开门开门", "请打开门", "打开房门", "我回来了",
        "芝麻请开门", "你好开门", "欢迎回家", "请开一下门", "解锁开门"
    };
};

class CppVoiceEngine {
public:
    explicit CppVoiceEngine(const VoiceEngineConfig& cfg = VoiceEngineConfig{});
    ~CppVoiceEngine();

    bool isHealthy() const;
    VoiceEngineConfig getConfig() const;
    void updateCommandsAndThresholds(const std::vector<std::string>& commands,
                                     float command_threshold,
                                     float similarity_threshold,
                                     float vad_energy_threshold);

    // 核心方案 A 两级级联识别：输入 16kHz 单声道归一化 PCM 波形
    VoiceRecognitionResult recognizeSamples(const std::vector<float>& pcm_16k);

    // 对 Base64 WAV 音频进行两级语音口令 + 声纹核验
    VoiceRecognitionResult recognizeBase64Wav(const std::string& base64_wav);

    // 家庭成员声纹样本库管理 (支持最多 10 位成员，支持 <人名>__<口令> 专属模板)
    nlohmann::json getVoiceSummary();
    nlohmann::json uploadVoiceSample(const std::string& person,
                                     const std::string& command_tag,
                                     const std::string& base64_wav);
    nlohmann::json deleteVoiceSample(const std::string& filename);
    std::vector<uchar> getVoiceFileBinary(const std::string& filename);
    nlohmann::json reloadGallery();

    // 静态音频工具：解析任意 PCM WAV 二进制并重采样至 16kHz 单声道 float32
    static bool decodeWavTo16kMono(const std::vector<uchar>& wav_bytes, std::vector<float>& out_pcm16k);

private:
    struct VoiceSampleEntry {
        std::string filename;
        std::string person;
        std::string command_tag; // 空字符串表示通用语料，非空表示特定口令专属模板
        std::vector<float> cmd_embedding; // 128 维保序口令特征
        std::vector<float> spk_embedding; // 192 维声纹特征
    };

    struct MemberVoiceProfile {
        std::string person;
        std::vector<std::string> files;
        std::vector<float> global_spk_proto; // 192 维全局均值声纹向量
        std::unordered_map<std::string, std::vector<std::vector<float>>> cmd_spk_templates;
    };

    bool loadModelsLocked();
    void rebuildGalleryLocked();
    bool vadTrimAndCenter(const std::vector<float>& pcm_16k,
                          std::vector<float>& out_aligned,
                          float& out_peak_rms) const;
    cv::Mat computeFbankBlob(const std::vector<float>& aligned_25600) const;
    bool extractEmbeddingsLocked(const cv::Mat& fbank_blob,
                                 std::vector<float>& out_logits,
                                 std::vector<float>& out_cmd_emb,
                                 std::vector<float>& out_spk_emb,
                                 bool run_stage2);

    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    static void l2Normalize(std::vector<float>& v);

    VoiceEngineConfig cfg_;
    mutable std::mutex mtx_;

    cv::dnn::Net kws_net_;
    cv::dnn::Net sv_net_;
    bool kws_loaded_ = false;
    bool sv_loaded_ = false;

    std::vector<std::string> model_commands_;
    std::unordered_map<std::string, std::vector<float>> meta_cmd_prototypes_;
    std::unordered_map<std::string, std::vector<std::vector<float>>> enrolled_cmd_templates_;
    std::unordered_map<std::string, MemberVoiceProfile> member_profiles_;

    // 预计算的汉明窗 (400) 与 HTK 80x257 梅尔三角滤波器组
    std::vector<float> hamming_win_;
    std::vector<float> mel_fb_; // 80 * 257
};

} // namespace myhome
