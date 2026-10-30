#include "voice_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace myhome {

namespace {

constexpr int kSampleRate = 16000;
constexpr int kTargetSamples = 25600; // 1.60s
constexpr int kNfft = 512;
constexpr int kNumBins = kNfft / 2 + 1; // 257
constexpr int kWinLength = 400;         // 25ms
constexpr int kHopLength = 160;         // 10ms
constexpr int kNumFrames = 160;
constexpr int kNumMels = 80;
constexpr float kFmin = 20.0f;
constexpr float kFmax = 7600.0f;
constexpr float kPreEmphasis = 0.97f;
constexpr double kPi = 3.14159265358979323846;

float hzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// 原地 512 点基 2 Cooley-Tukey 复数 FFT (无任何外部音频库依赖，耗时 < 0.05ms)
void fft512(float* real, float* imag) {
    constexpr int N = 512;
    int j = 0;
    for (int i = 1; i < N; ++i) {
        int bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    for (int len = 2; len <= N; len <<= 1) {
        double ang = -2.0 * kPi / static_cast<double>(len);
        float wlen_r = static_cast<float>(std::cos(ang));
        float wlen_i = static_cast<float>(std::sin(ang));
        int half = len >> 1;
        for (int i = 0; i < N; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (int k = 0; k < half; ++k) {
                float ur = real[i + k];
                float ui = imag[i + k];
                float vr = real[i + k + half] * wr - imag[i + k + half] * wi;
                float vi = real[i + k + half] * wi + imag[i + k + half] * wr;
                real[i + k] = ur + vr;
                imag[i + k] = ui + vi;
                real[i + k + half] = ur - vr;
                imag[i + k + half] = ui - vi;

                float next_wr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = next_wr;
            }
        }
    }
}

std::vector<uchar> base64DecodeAudio(const std::string& encoded) {
    std::string clean = encoded;
    auto comma = clean.find(',');
    if (comma != std::string::npos) {
        clean = clean.substr(comma + 1);
    }
    static const int T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    std::vector<uchar> out;
    out.reserve(clean.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : clean) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uchar>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

bool isSafeAudioFilename(const std::string& s) {
    if (s.empty() || s == "." || s == "..") return false;
    if (s.find("..") != std::string::npos) return false;
    for (char c : s) {
        if (c == '/' || c == '\\' || c == '\0' || c == ':') return false;
    }
    return true;
}

std::string sanitizeToken(const std::string& raw) {
    std::string out;
    for (char c : raw) {
        if (c != '/' && c != '\\' && c != '\0' && c != ':' && c != '.') {
            out.push_back(c);
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// 解析文件名规则:
//   1) <人名>__<口令>_<序号>.wav  -> person=<人名>, command_tag=<口令>
//   2) <人名>__<口令>.wav         -> person=<人名>, command_tag=<口令>
//   3) <人名>_<序号>.wav 或 <人名>.wav -> person=<人名>, command_tag=""
void parseVoiceFilename(const std::string& stem, std::string& out_person, std::string& out_cmd) {
    out_cmd.clear();
    auto dpos = stem.find("__");
    if (dpos != std::string::npos) {
        out_person = stem.substr(0, dpos);
        std::string rest = stem.substr(dpos + 2);
        auto upos = rest.rfind('_');
        if (upos != std::string::npos) {
            out_cmd = rest.substr(0, upos);
        } else {
            out_cmd = rest;
        }
        return;
    }
    auto upos = stem.find('_');
    if (upos != std::string::npos) {
        out_person = stem.substr(0, upos);
    } else {
        out_person = stem;
    }
}

uint16_t readLE16(const uchar* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readLE32(const uchar* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

} // namespace

CppVoiceEngine::CppVoiceEngine(const VoiceEngineConfig& cfg) : cfg_(cfg) {
    // 1. 预计算 400 点汉明窗
    hamming_win_.resize(kWinLength);
    for (int n = 0; n < kWinLength; ++n) {
        hamming_win_[n] = static_cast<float>(0.54 - 0.46 * std::cos(2.0 * kPi * n / (kWinLength - 1)));
    }

    // 2. 预计算 80x257 HTK 梅尔滤波器组 (与 Python core/fbank.py 完全一致)
    mel_fb_.assign(kNumMels * kNumBins, 0.0f);
    float mel_min = hzToMel(kFmin);
    float mel_max = hzToMel(kFmax);
    std::vector<int> bin_pts(kNumMels + 2);
    for (int i = 0; i < kNumMels + 2; ++i) {
        float m = mel_min + (mel_max - mel_min) * static_cast<float>(i) / static_cast<float>(kNumMels + 1);
        float hz = melToHz(m);
        bin_pts[i] = static_cast<int>(std::floor((kNfft + 1) * hz / static_cast<float>(kSampleRate)));
    }
    for (int m = 1; m <= kNumMels; ++m) {
        int left = bin_pts[m - 1];
        int center = std::max(left + 1, bin_pts[m]);
        int right = std::max(center + 1, bin_pts[m + 1]);
        for (int k = left; k < std::min(center, kNumBins); ++k) {
            mel_fb_[(m - 1) * kNumBins + k] = static_cast<float>(k - left) / static_cast<float>(center - left);
        }
        for (int k = center; k < std::min(right, kNumBins); ++k) {
            mel_fb_[(m - 1) * kNumBins + k] = static_cast<float>(right - k) / static_cast<float>(right - center);
        }
    }

    std::error_code ec;
    fs::create_directories(cfg_.db_dir, ec);

    std::lock_guard<std::mutex> lk(mtx_);
    loadModelsLocked();
    rebuildGalleryLocked();
}

CppVoiceEngine::~CppVoiceEngine() = default;

bool CppVoiceEngine::isHealthy() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return kws_loaded_ && sv_loaded_;
}

VoiceEngineConfig CppVoiceEngine::getConfig() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return cfg_;
}

void CppVoiceEngine::updateCommandsAndThresholds(const std::vector<std::string>& commands,
                                                 float command_threshold,
                                                 float similarity_threshold,
                                                 float vad_energy_threshold) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!commands.empty()) {
        cfg_.unlock_commands = commands;
    }
    cfg_.command_threshold = std::clamp(command_threshold, 0.1f, 0.99f);
    cfg_.similarity_threshold = std::clamp(similarity_threshold, 0.1f, 0.99f);
    cfg_.vad_energy_threshold = std::clamp(vad_energy_threshold, 0.001f, 0.5f);
}

bool CppVoiceEngine::loadModelsLocked() {
    kws_loaded_ = false;
    sv_loaded_ = false;
    model_commands_ = cfg_.unlock_commands;
    meta_cmd_prototypes_.clear();

    if (!cfg_.commands_meta_path.empty() && fs::exists(cfg_.commands_meta_path)) {
        try {
            std::ifstream ifs(cfg_.commands_meta_path);
            auto meta = nlohmann::json::parse(ifs);
            if (meta.contains("commands") && meta["commands"].is_array() && !meta["commands"].empty()) {
                model_commands_.clear();
                for (const auto& c : meta["commands"]) {
                    model_commands_.push_back(c.get<std::string>());
                }
            }
            if (meta.contains("prototypes") && meta["prototypes"].is_object()) {
                for (auto it = meta["prototypes"].begin(); it != meta["prototypes"].end(); ++it) {
                    std::vector<float> vec;
                    for (const auto& v : it.value()) vec.push_back(v.get<float>());
                    if (!vec.empty()) {
                        l2Normalize(vec);
                        meta_cmd_prototypes_[it.key()] = std::move(vec);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "⚠️ [CppVoiceEngine] 解析 commands.json 提示: " << e.what() << std::endl;
        }
    }

    if (fs::exists(cfg_.kws_model_path)) {
        try {
            kws_net_ = cv::dnn::readNetFromONNX(cfg_.kws_model_path);
            kws_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            kws_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            kws_loaded_ = !kws_net_.empty();
        } catch (const cv::Exception& e) {
            std::cerr << "⚠️ [CppVoiceEngine] 加载第一级口令模型失败: " << e.what() << std::endl;
        }
    }

    if (fs::exists(cfg_.sv_model_path)) {
        try {
            sv_net_ = cv::dnn::readNetFromONNX(cfg_.sv_model_path);
            sv_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            sv_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            sv_loaded_ = !sv_net_.empty();
        } catch (const cv::Exception& e) {
            std::cerr << "⚠️ [CppVoiceEngine] 加载第二级声纹模型失败: " << e.what() << std::endl;
        }
    }

    if (kws_loaded_ && sv_loaded_) {
        std::cout << "🎙️ [CppVoiceEngine] 方案 A 两级超轻量语音引擎就绪！" << std::endl;
        std::cout << "   • Stage-1 口令模型: " << cfg_.kws_model_path
                  << " (支持 " << cfg_.unlock_commands.size() << " 种口令 + 闲聊拒识)" << std::endl;
        std::cout << "   • Stage-2 声纹模型: " << cfg_.sv_model_path
                  << " (最大支持 " << cfg_.max_members << " 位家人声纹比对)" << std::endl;
    } else {
        std::cout << "ℹ️ [CppVoiceEngine] 语音 ONNX 权重尚未就绪 (可运行 python door_voice_detector/train.py 导出至 ./audio/models/)" << std::endl;
    }
    return kws_loaded_ && sv_loaded_;
}

bool CppVoiceEngine::decodeWavTo16kMono(const std::vector<uchar>& wav_bytes, std::vector<float>& out_pcm16k) {
    out_pcm16k.clear();
    if (wav_bytes.size() < 44) return false;
    if (std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0) {
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uchar* data_ptr = nullptr;
    uint32_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= wav_bytes.size()) {
        const uchar* chunk_hdr = wav_bytes.data() + offset;
        uint32_t chunk_sz = readLE32(chunk_hdr + 4);
        offset += 8;
        if (offset + chunk_sz > wav_bytes.size()) {
            chunk_sz = static_cast<uint32_t>(wav_bytes.size() - offset);
        }

        if (std::memcmp(chunk_hdr, "fmt ", 4) == 0 && chunk_sz >= 16) {
            audio_format = readLE16(wav_bytes.data() + offset);
            num_channels = readLE16(wav_bytes.data() + offset + 2);
            sample_rate = readLE32(wav_bytes.data() + offset + 4);
            bits_per_sample = readLE16(wav_bytes.data() + offset + 14);
        } else if (std::memcmp(chunk_hdr, "data", 4) == 0) {
            data_ptr = wav_bytes.data() + offset;
            data_size = chunk_sz;
            break;
        }
        offset += chunk_sz + (chunk_sz & 1u);
    }

    if (!data_ptr || data_size == 0 || num_channels == 0 || sample_rate == 0) {
        return false;
    }

    size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0) return false;
    size_t frame_stride = bytes_per_sample * num_channels;
    size_t num_frames = data_size / frame_stride;
    if (num_frames == 0) return false;

    std::vector<float> mono(num_frames, 0.0f);
    for (size_t i = 0; i < num_frames; ++i) {
        float sum = 0.0f;
        for (uint16_t ch = 0; ch < num_channels; ++ch) {
            const uchar* sp = data_ptr + i * frame_stride + ch * bytes_per_sample;
            float v = 0.0f;
            if (audio_format == 1 && bits_per_sample == 16) {
                int16_t s = static_cast<int16_t>(readLE16(sp));
                v = static_cast<float>(s) / 32768.0f;
            } else if (audio_format == 1 && bits_per_sample == 8) {
                v = (static_cast<float>(sp[0]) - 128.0f) / 128.0f;
            } else if (audio_format == 1 && bits_per_sample == 32) {
                int32_t s = static_cast<int32_t>(readLE32(sp));
                v = static_cast<float>(static_cast<double>(s) / 2147483648.0);
            } else if (audio_format == 3 && bits_per_sample == 32) {
                float f = 0.0f;
                std::memcpy(&f, sp, sizeof(float));
                v = f;
            } else {
                return false;
            }
            sum += v;
        }
        mono[i] = sum / static_cast<float>(num_channels);
    }

    if (sample_rate == static_cast<uint32_t>(kSampleRate)) {
        out_pcm16k = std::move(mono);
        return true;
    }

    // 线性重采样至 16kHz
    size_t out_len = std::max<size_t>(
        1, static_cast<size_t>(std::llround(static_cast<double>(mono.size()) * kSampleRate / sample_rate)));
    out_pcm16k.resize(out_len);
    if (mono.size() == 1) {
        std::fill(out_pcm16k.begin(), out_pcm16k.end(), mono[0]);
        return true;
    }
    double scale = static_cast<double>(mono.size() - 1) / static_cast<double>(std::max<size_t>(1, out_len - 1));
    for (size_t i = 0; i < out_len; ++i) {
        double src_pos = i * scale;
        size_t li = static_cast<size_t>(std::floor(src_pos));
        size_t ri = std::min(li + 1, mono.size() - 1);
        float frac = static_cast<float>(src_pos - li);
        out_pcm16k[i] = mono[li] * (1.0f - frac) + mono[ri] * frac;
    }
    return true;
}

bool CppVoiceEngine::vadTrimAndCenter(const std::vector<float>& pcm_16k,
                                      std::vector<float>& out_aligned,
                                      float& out_peak_rms) const {
    out_peak_rms = 0.0f;
    if (pcm_16k.size() < static_cast<size_t>(kHopLength * 15)) {
        return false;
    }

    // 去直流偏移
    double mean_dc = 0.0;
    for (float x : pcm_16k) mean_dc += x;
    mean_dc /= static_cast<double>(pcm_16k.size());

    size_t num_frames = pcm_16k.size() / kHopLength;
    std::vector<float> rms(num_frames, 0.0f);
    for (size_t f = 0; f < num_frames; ++f) {
        double sq = 0.0;
        size_t base = f * kHopLength;
        for (int i = 0; i < kHopLength; ++i) {
            double v = pcm_16k[base + i] - mean_dc;
            sq += v * v;
        }
        rms[f] = static_cast<float>(std::sqrt(sq / kHopLength + 1e-12));
        if (rms[f] > out_peak_rms) out_peak_rms = rms[f];
    }

    if (out_peak_rms < cfg_.vad_energy_threshold) {
        return false;
    }

    float dyn_thresh = std::max(cfg_.vad_energy_threshold * 0.6f, out_peak_rms * 0.10f);
    int first_active = -1;
    int last_active = -1;
    int active_count = 0;
    for (int f = 0; f < static_cast<int>(num_frames); ++f) {
        if (rms[f] >= dyn_thresh) {
            if (first_active < 0) first_active = f;
            last_active = f;
            active_count++;
        }
    }

    if (active_count < 18 || first_active < 0) {
        return false;
    }

    int start_f = std::max(0, first_active - 12);
    int end_f = std::min(static_cast<int>(num_frames), last_active + 13);
    size_t s_idx = static_cast<size_t>(start_f) * kHopLength;
    size_t e_idx = static_cast<size_t>(end_f) * kHopLength;

    float peak_abs = 0.0f;
    for (size_t i = s_idx; i < e_idx; ++i) {
        float a = std::abs(static_cast<float>(pcm_16k[i] - mean_dc));
        if (a > peak_abs) peak_abs = a;
    }
    float gain = (peak_abs > 1e-4f) ? (0.90f / peak_abs) : 1.0f;

    size_t seg_len = e_idx - s_idx;
    out_aligned.assign(kTargetSamples, 0.0f);
    if (seg_len >= static_cast<size_t>(kTargetSamples)) {
        size_t off = (seg_len - kTargetSamples) / 2;
        for (int i = 0; i < kTargetSamples; ++i) {
            out_aligned[i] = static_cast<float>(pcm_16k[s_idx + off + i] - mean_dc) * gain;
        }
    } else {
        size_t off = (kTargetSamples - seg_len) / 2;
        for (size_t i = 0; i < seg_len; ++i) {
            out_aligned[off + i] = static_cast<float>(pcm_16k[s_idx + i] - mean_dc) * gain;
        }
    }
    return true;
}

cv::Mat CppVoiceEngine::computeFbankBlob(const std::vector<float>& aligned_25600) const {
    // 1. 0.97 预加重 + 尾部补 240 零
    std::vector<float> padded(kTargetSamples + (kWinLength - kHopLength), 0.0f);
    if (!aligned_25600.empty()) {
        padded[0] = aligned_25600[0];
        for (size_t i = 1; i < aligned_25600.size() && i < static_cast<size_t>(kTargetSamples); ++i) {
            padded[i] = aligned_25600[i] - kPreEmphasis * aligned_25600[i - 1];
        }
    }

    int sz[4] = {1, 1, kNumMels, kNumFrames};
    cv::Mat blob(4, sz, CV_32F, cv::Scalar(0.0f));
    float* out_ptr = blob.ptr<float>(); // 按 [mel][frame] 连续排布 (80 x 160)

    float fft_r[kNfft];
    float fft_i[kNfft];
    float power[kNumBins];

    for (int t = 0; t < kNumFrames; ++t) {
        std::memset(fft_r, 0, sizeof(fft_r));
        std::memset(fft_i, 0, sizeof(fft_i));
        int base = t * kHopLength;
        for (int n = 0; n < kWinLength; ++n) {
            fft_r[n] = padded[base + n] * hamming_win_[n];
        }
        fft512(fft_r, fft_i);

        for (int k = 0; k < kNumBins; ++k) {
            power[k] = fft_r[k] * fft_r[k] + fft_i[k] * fft_i[k];
        }

        for (int m = 0; m < kNumMels; ++m) {
            const float* fb_row = &mel_fb_[m * kNumBins];
            float mel_e = 0.0f;
            for (int k = 0; k < kNumBins; ++k) {
                mel_e += fb_row[k] * power[k];
            }
            out_ptr[m * kNumFrames + t] = std::log(std::max(mel_e, 1e-10f));
        }
    }

    // 句级频带均值归一化 (CMN)
    for (int m = 0; m < kNumMels; ++m) {
        float* row = out_ptr + m * kNumFrames;
        double sum = 0.0;
        for (int t = 0; t < kNumFrames; ++t) sum += row[t];
        float mean = static_cast<float>(sum / kNumFrames);
        for (int t = 0; t < kNumFrames; ++t) row[t] -= mean;
    }

    return blob;
}

float CppVoiceEngine::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

void CppVoiceEngine::l2Normalize(std::vector<float>& v) {
    double sq = 0.0;
    for (float x : v) sq += static_cast<double>(x) * x;
    float inv = static_cast<float>(1.0 / std::sqrt(std::max(sq, 1e-12)));
    for (float& x : v) x *= inv;
}

bool CppVoiceEngine::extractEmbeddingsLocked(const cv::Mat& fbank_blob,
                                             std::vector<float>& out_logits,
                                             std::vector<float>& out_cmd_emb,
                                             std::vector<float>& out_spk_emb,
                                             bool run_stage2) {
    if (!kws_loaded_) return false;

    kws_net_.setInput(fbank_blob);
    cv::Mat out1 = kws_net_.forward(); // [1, (1 + K) + 128]
    int total_d1 = static_cast<int>(out1.total());
    int num_cls = static_cast<int>(model_commands_.size()) + 1;
    if (total_d1 <= num_cls) return false;

    const float* p1 = out1.ptr<float>();
    out_logits.assign(p1, p1 + num_cls);
    out_cmd_emb.assign(p1 + num_cls, p1 + total_d1);
    l2Normalize(out_cmd_emb);

    if (run_stage2 && sv_loaded_) {
        sv_net_.setInput(fbank_blob);
        cv::Mat out2 = sv_net_.forward(); // [1, 192]
        const float* p2 = out2.ptr<float>();
        out_spk_emb.assign(p2, p2 + out2.total());
        l2Normalize(out_spk_emb);
    }
    return true;
}

void CppVoiceEngine::rebuildGalleryLocked() {
    member_profiles_.clear();
    enrolled_cmd_templates_.clear();

    std::error_code ec;
    if (!fs::exists(cfg_.db_dir, ec)) return;

    std::vector<fs::path> wav_files;
    for (const auto& entry : fs::directory_iterator(cfg_.db_dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".wav") {
            wav_files.push_back(entry.path());
        }
    }
    std::sort(wav_files.begin(), wav_files.end());

    std::unordered_map<std::string, std::vector<std::vector<float>>> person_all_spk;

    for (const auto& wp : wav_files) {
        std::string fname = wp.filename().string();
        std::string stem = wp.stem().string();
        std::string person, cmd_tag;
        parseVoiceFilename(stem, person, cmd_tag);
        if (person.empty()) continue;

        if (member_profiles_.find(person) == member_profiles_.end() &&
            static_cast<int>(member_profiles_.size()) >= cfg_.max_members) {
            continue;
        }

        auto& prof = member_profiles_[person];
        prof.person = person;
        prof.files.push_back(fname);

        if (!kws_loaded_ || !sv_loaded_) continue;

        std::ifstream ifs(wp, std::ios::binary);
        if (!ifs) continue;
        std::vector<uchar> raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        std::vector<float> pcm;
        if (!decodeWavTo16kMono(raw, pcm)) continue;

        std::vector<float> aligned;
        float peak_rms = 0.0f;
        if (!vadTrimAndCenter(pcm, aligned, peak_rms)) {
            aligned.assign(kTargetSamples, 0.0f);
            size_t cp = std::min<size_t>(pcm.size(), kTargetSamples);
            std::copy(pcm.begin(), pcm.begin() + cp, aligned.begin());
        }

        cv::Mat blob = computeFbankBlob(aligned);
        std::vector<float> logits, cmd_emb, spk_emb;
        if (extractEmbeddingsLocked(blob, logits, cmd_emb, spk_emb, true) && !spk_emb.empty()) {
            person_all_spk[person].push_back(spk_emb);
            if (!cmd_tag.empty()) {
                prof.cmd_spk_templates[cmd_tag].push_back(spk_emb);
                enrolled_cmd_templates_[cmd_tag].push_back(cmd_emb);
            }
        }
    }

    // 计算每位成员的全局均值声纹向量
    for (auto& kv : member_profiles_) {
        const auto& vecs = person_all_spk[kv.first];
        if (vecs.empty()) continue;
        std::vector<float> mean_v(vecs[0].size(), 0.0f);
        for (const auto& v : vecs) {
            for (size_t i = 0; i < mean_v.size(); ++i) mean_v[i] += v[i];
        }
        l2Normalize(mean_v);
        kv.second.global_spk_proto = std::move(mean_v);
    }
}

VoiceRecognitionResult CppVoiceEngine::recognizeSamples(const std::vector<float>& pcm_16k) {
    VoiceRecognitionResult res;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<float> aligned;
    float peak_rms = 0.0f;
    if (!vadTrimAndCenter(pcm_16k, aligned, peak_rms)) {
        res.vad_passed = false;
        res.stage = "VAD_SILENCE";
        auto t1 = std::chrono::steady_clock::now();
        res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return res;
    }
    res.vad_passed = true;

    cv::Mat fbank_blob = computeFbankBlob(aligned);

    std::lock_guard<std::mutex> lk(mtx_);
    if (!kws_loaded_ || !sv_loaded_) {
        res.stage = "MODEL_NOT_READY";
        return res;
    }

    // ==================== 第一级：口令识别与非口令短路 ====================
    std::vector<float> logits, cmd_emb, spk_emb;
    if (!extractEmbeddingsLocked(fbank_blob, logits, cmd_emb, spk_emb, false)) {
        res.stage = "STAGE1_COMMAND_REJECT";
        return res;
    }

    // 1) 计算分类头 Softmax 概率 (logits[0] 为 UNKNOWN 闲聊拒识通道)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    std::vector<float> probs(logits.size(), 0.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = static_cast<float>(std::exp(logits[i] - max_logit));
        sum_exp += probs[i];
    }
    for (float& p : probs) p /= static_cast<float>(sum_exp);

    std::string best_cmd = "NONE";
    float best_cmd_score = 0.0f;

    // 融合分类头概率与声学保序模板余弦得分（兼顾预训练口令与用户自定义新口令）
    for (const auto& allowed_cmd : cfg_.unlock_commands) {
        float cls_prob = 0.0f;
        auto it_m = std::find(model_commands_.begin(), model_commands_.end(), allowed_cmd);
        if (it_m != model_commands_.end()) {
            size_t idx = static_cast<size_t>(std::distance(model_commands_.begin(), it_m)) + 1;
            if (idx < probs.size() && probs[idx] > probs[0]) {
                cls_prob = probs[idx];
            }
        }

        float tpl_sim = 0.0f;
        auto it_enr = enrolled_cmd_templates_.find(allowed_cmd);
        if (it_enr != enrolled_cmd_templates_.end()) {
            for (const auto& tv : it_enr->second) {
                tpl_sim = std::max(tpl_sim, cosineSimilarity(cmd_emb, tv));
            }
        }
        auto it_meta = meta_cmd_prototypes_.find(allowed_cmd);
        if (it_meta != meta_cmd_prototypes_.end()) {
            tpl_sim = std::max(tpl_sim, cosineSimilarity(cmd_emb, it_meta->second));
        }

        float fused = std::max(cls_prob, tpl_sim);
        if (fused > best_cmd_score) {
            best_cmd_score = fused;
            best_cmd = allowed_cmd;
        }
    }

    res.command_score = std::round(best_cmd_score * 1000.0f) / 1000.0f;
    if (best_cmd == "NONE" || best_cmd_score < cfg_.command_threshold) {
        // 第一级未命中 10 种口令：直接短路丢弃，绝不唤醒第二级声纹网！
        res.command_matched = false;
        res.command = "NONE";
        res.stage = "STAGE1_COMMAND_REJECT";
        auto t1 = std::chrono::steady_clock::now();
        res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return res;
    }

    res.command_matched = true;
    res.command = best_cmd;

    // ==================== 第二级：声纹特征提取与 1:10 家人比对 ====================
    sv_net_.setInput(fbank_blob);
    cv::Mat out2 = sv_net_.forward();
    const float* p2 = out2.ptr<float>();
    spk_emb.assign(p2, p2 + out2.total());
    l2Normalize(spk_emb);

    std::string best_person = "陌生人";
    float best_spk_score = 0.0f;

    for (const auto& kv : member_profiles_) {
        const auto& prof = kv.second;
        float member_best = -1.0f;

        // 优先比对该成员在当前同一句口令下的专属模板 (1s 短语音同文本比对最精准)
        auto it_cmd = prof.cmd_spk_templates.find(best_cmd);
        if (it_cmd != prof.cmd_spk_templates.end() && !it_cmd->second.empty()) {
            for (const auto& tvec : it_cmd->second) {
                member_best = std::max(member_best, cosineSimilarity(spk_emb, tvec));
            }
        }

        // 同时结合该成员的全局均值声纹模板兜底
        if (!prof.global_spk_proto.empty()) {
            member_best = std::max(member_best, cosineSimilarity(spk_emb, prof.global_spk_proto));
        }

        if (member_best > best_spk_score) {
            best_spk_score = member_best;
            best_person = prof.person;
        }
    }

    res.speaker_score = std::round(std::max(0.0f, best_spk_score) * 1000.0f) / 1000.0f;
    if (!member_profiles_.empty() && best_spk_score >= cfg_.similarity_threshold) {
        res.is_family = true;
        res.person = best_person;
        res.stage = "STAGE2_VERIFIED";
    } else {
        res.is_family = false;
        res.person = "陌生人";
        res.stage = "STAGE2_STRANGER_REJECT";
    }

    auto t1 = std::chrono::steady_clock::now();
    res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

VoiceRecognitionResult CppVoiceEngine::recognizeBase64Wav(const std::string& base64_wav) {
    auto bytes = base64DecodeAudio(base64_wav);
    std::vector<float> pcm;
    if (!decodeWavTo16kMono(bytes, pcm)) {
        VoiceRecognitionResult r;
        r.stage = "INVALID_WAV";
        return r;
    }
    return recognizeSamples(pcm);
}

nlohmann::json CppVoiceEngine::getVoiceSummary() {
    std::lock_guard<std::mutex> lk(mtx_);
    nlohmann::json members_arr = nlohmann::json::array();
    for (const auto& kv : member_profiles_) {
        const auto& prof = kv.second;
        nlohmann::json cmd_tags = nlohmann::json::array();
        for (const auto& ckv : prof.cmd_spk_templates) {
            cmd_tags.push_back(ckv.first);
        }
        members_arr.push_back({
            {"name", prof.person},
            {"sample_count", prof.files.size()},
            {"files", prof.files},
            {"enrolled_commands", cmd_tags}
        });
    }

    return nlohmann::json{
        {"enabled", cfg_.enabled},
        {"models_ready", kws_loaded_ && sv_loaded_},
        {"kws_model_path", cfg_.kws_model_path},
        {"sv_model_path", cfg_.sv_model_path},
        {"db_dir", cfg_.db_dir},
        {"max_members", cfg_.max_members},
        {"enrolled_member_count", member_profiles_.size()},
        {"command_threshold", cfg_.command_threshold},
        {"similarity_threshold", cfg_.similarity_threshold},
        {"vad_energy_threshold", cfg_.vad_energy_threshold},
        {"unlock_commands", cfg_.unlock_commands},
        {"members", members_arr}
    };
}

nlohmann::json CppVoiceEngine::uploadVoiceSample(const std::string& person,
                                                 const std::string& command_tag,
                                                 const std::string& base64_wav) {
    std::string safe_person = sanitizeToken(person);
    std::string safe_cmd = sanitizeToken(command_tag);
    if (safe_person.empty()) {
        return nlohmann::json{{"status", "error"}, {"error", "成员姓名不能为空"}};
    }

    auto wav_bytes = base64DecodeAudio(base64_wav);
    std::vector<float> pcm;
    if (!decodeWavTo16kMono(wav_bytes, pcm)) {
        return nlohmann::json{{"status", "error"}, {"error", "无效的 WAV 音频格式（需为标准 PCM WAV）"}};
    }

    std::vector<float> aligned;
    float peak_rms = 0.0f;
    if (!vadTrimAndCenter(pcm, aligned, peak_rms)) {
        return nlohmann::json{{"status", "error"}, {"error", "录音音量过低或有效语音时长不足 0.2 秒，请靠近麦克风清晰朗读口令"}};
    }

    std::lock_guard<std::mutex> lk(mtx_);
    if (member_profiles_.find(safe_person) == member_profiles_.end() &&
        static_cast<int>(member_profiles_.size()) >= cfg_.max_members) {
        return nlohmann::json{
            {"status", "error"},
            {"error", "家庭声纹成员已达上限 (" + std::to_string(cfg_.max_members) + " 人)"}
        };
    }

    std::error_code ec;
    fs::create_directories(cfg_.db_dir, ec);

    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() % 1000000;
    std::string fname = safe_cmd.empty()
        ? (safe_person + "_" + std::to_string(now_ms) + ".wav")
        : (safe_person + "__" + safe_cmd + "_" + std::to_string(now_ms) + ".wav");

    fs::path out_path = fs::path(cfg_.db_dir) / fname;
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        return nlohmann::json{{"status", "error"}, {"error", "保存声纹样本文件失败"}};
    }
    ofs.write(reinterpret_cast<const char*>(wav_bytes.data()), static_cast<std::streamsize>(wav_bytes.size()));
    ofs.close();

    rebuildGalleryLocked();
    return nlohmann::json{
        {"status", "success"},
        {"file", fname},
        {"person", safe_person},
        {"command", safe_cmd},
        {"enrolled_member_count", member_profiles_.size()}
    };
}

nlohmann::json CppVoiceEngine::deleteVoiceSample(const std::string& filename) {
    if (!isSafeAudioFilename(filename)) {
        return nlohmann::json{{"status", "error"}, {"error", "非法文件名"}};
    }
    std::lock_guard<std::mutex> lk(mtx_);
    fs::path target = fs::path(cfg_.db_dir) / filename;
    std::error_code ec;
    if (fs::is_regular_file(target, ec) && fs::remove(target, ec)) {
        rebuildGalleryLocked();
        return nlohmann::json{{"status", "success"}, {"deleted", filename}};
    }
    return nlohmann::json{{"status", "error"}, {"error", "声纹文件不存在"}};
}

std::vector<uchar> CppVoiceEngine::getVoiceFileBinary(const std::string& filename) {
    std::vector<uchar> out;
    if (!isSafeAudioFilename(filename)) return out;
    std::string db_dir;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        db_dir = cfg_.db_dir;
    }
    fs::path target = fs::path(db_dir) / filename;
    std::error_code ec;
    if (!fs::is_regular_file(target, ec)) return out;
    std::ifstream ifs(target, std::ios::binary);
    if (!ifs) return out;
    return std::vector<uchar>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

nlohmann::json CppVoiceEngine::reloadGallery() {
    std::lock_guard<std::mutex> lk(mtx_);
    loadModelsLocked();
    rebuildGalleryLocked();
    return nlohmann::json{
        {"status", "success"},
        {"models_ready", kws_loaded_ && sv_loaded_},
        {"enrolled_member_count", member_profiles_.size()}
    };
}

} // namespace myhome
