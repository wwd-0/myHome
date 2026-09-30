#include "face_engine.hpp"
#include <filesystem>
#include <iostream>
#include <regex>
#include <cmath>
#include <mutex>

namespace fs = std::filesystem;

namespace myhome {

CppFaceEngine::CppFaceEngine(const std::string& db_dir,
                             const std::string& det_model_path,
                             const std::string& rec_model_path,
                             float sim_threshold,
                             int min_face_width_px)
    : db_dir_(db_dir),
      det_model_path_(det_model_path),
      rec_model_path_(rec_model_path),
      sim_threshold_(sim_threshold),
      min_face_width_px_(min_face_width_px) {
    detector_ = cv::FaceDetectorYN::create(det_model_path_, "", cv::Size(640, 480), 0.65f, 0.3f, 5000);
    recognizer_ = cv::FaceRecognizerSF::create(rec_model_path_, "");
    reloadFamilyDb();
}

std::string CppFaceEngine::parsePersonName(const std::string& filename) {
    std::string stem = fs::path(filename).stem().string();
    auto pos = stem.find('_');
    if (pos != std::string::npos) {
        stem = stem.substr(0, pos);
    }
    stem = std::regex_replace(stem, std::regex("\\d+$"), "");
    return stem.empty() ? "Family" : stem;
}

int CppFaceEngine::reloadFamilyDb() {
    family_db_.clear();
    if (!fs::exists(db_dir_)) {
        fs::create_directories(db_dir_);
        return 0;
    }

    std::map<std::string, std::vector<cv::Mat>> temp_feats;

    for (const auto& entry : fs::directory_iterator(db_dir_)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".webp") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        detector_->setInputSize(img.size());
        cv::Mat faces;
        detector_->detect(img, faces);
        if (faces.rows < 1) continue;

        int best_row = 0;
        float max_area = 0.0f;
        for (int r = 0; r < faces.rows; ++r) {
            float area = faces.at<float>(r, 2) * faces.at<float>(r, 3);
            if (area > max_area) {
                max_area = area;
                best_row = r;
            }
        }

        cv::Mat aligned_face, feature;
        recognizer_->alignCrop(img, faces.row(best_row), aligned_face);
        recognizer_->feature(aligned_face, feature);
        cv::Mat norm_feat;
        cv::normalize(feature.clone(), norm_feat, 1.0, 0.0, cv::NORM_L2);

        std::string person = parsePersonName(entry.path().filename().string());
        temp_feats[person].push_back(norm_feat);
    }

    for (auto& [person, feat_list] : temp_feats) {
        cv::Mat sum_feat = cv::Mat::zeros(feat_list[0].size(), CV_32F);
        for (const auto& f : feat_list) {
            sum_feat += f;
        }
        sum_feat /= static_cast<float>(feat_list.size());
        cv::Mat fused_feat;
        cv::normalize(sum_feat, fused_feat, 1.0, 0.0, cv::NORM_L2);
        family_db_[person] = fused_feat;
        std::cout << "✅ [CppFaceEngine] 已入库家人 [" << person << "] (融合了 "
                  << feat_list.size() << " 张照片, 特征维度: " << fused_feat.cols << ")" << std::endl;
    }

    return static_cast<int>(family_db_.size());
}

std::pair<std::string, float> CppFaceEngine::matchFeature(const cv::Mat& norm_feat) const {
    std::string best_name = "Stranger";
    float best_score = -1.0f;

    for (const auto& [name, db_feat] : family_db_) {
        float score = static_cast<float>(norm_feat.dot(db_feat));
        if (score > best_score) {
            best_score = score;
            best_name = name;
        }
    }

    if (best_score >= sim_threshold_) {
        return {best_name, best_score};
    }
    return {"Stranger", best_score};
}

std::optional<FaceMatchResult> CppFaceEngine::analyzeFrame(cv::Mat& frame, bool& has_stranger_out) {
    has_stranger_out = false;
    if (frame.empty() || !detector_ || !recognizer_) return std::nullopt;

    detector_->setInputSize(frame.size());
    cv::Mat faces;
    detector_->detect(frame, faces);

    std::optional<FaceMatchResult> best_family = std::nullopt;

    for (int r = 0; r < faces.rows; ++r) {
        float x = faces.at<float>(r, 0);
        float y = faces.at<float>(r, 1);
        float w = faces.at<float>(r, 2);
        float h = faces.at<float>(r, 3);
        cv::Rect bbox(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h));

        cv::Mat aligned_face, feature, norm_feat;
        recognizer_->alignCrop(frame, faces.row(r), aligned_face);
        recognizer_->feature(aligned_face, feature);
        cv::normalize(feature, norm_feat, 1.0, 0.0, cv::NORM_L2);

        auto [name, score] = matchFeature(norm_feat);
        cv::Scalar color;
        std::string label;

        if (name != "Stranger") {
            if (static_cast<int>(w) >= min_face_width_px_) {
                if (!best_family.has_value() || score > best_family->score) {
                    best_family = FaceMatchResult{name, score, bbox};
                }
                color = cv::Scalar(0, 255, 0);
                label = "Family (" + cv::format("%.2f", score) + ")";
            } else {
                color = cv::Scalar(0, 200, 200);
                label = "Far (" + cv::format("%.2f", score) + ")";
            }
        } else {
            has_stranger_out = true;
            color = cv::Scalar(0, 0, 255);
            label = "Stranger (" + cv::format("%.2f", score) + ")";
        }

        cv::rectangle(frame, bbox, color, 2);
        for (int k = 0; k < 5; ++k) {
            int kx = static_cast<int>(faces.at<float>(r, 4 + k * 2));
            int ky = static_cast<int>(faces.at<float>(r, 4 + k * 2 + 1));
            cv::circle(frame, cv::Point(kx, ky), 2, cv::Scalar(255, 255, 0), -1);
        }
        cv::putText(frame, label, cv::Point(bbox.x, std::max(25, bbox.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2);
    }

    return best_family;
}

std::vector<std::string> CppFaceEngine::getEnrolledNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : family_db_) {
        names.push_back(name);
    }
    return names;
}

} // namespace myhome
