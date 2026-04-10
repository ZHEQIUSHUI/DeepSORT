#pragma once

#include "model.h"
#include "reid_feature_extractor.hpp"

#include "SimpleCV.hpp"

#include <string>

// NPU-native ReID feature extractor for DeepSORT.
//
// - No OpenCV / onnxruntime dependency.
// - Expects model output dim == k_feature_dim.
class FeatureTensor {
public:
    static FeatureTensor* getInstance();

    // Initialize ReID axmodel. Must be called before getRectsFeature().
    // color_order: "rgb" (default) or "bgr" for model input pixels.
    bool init(const std::string& model_path,
              const std::string& color_order = "rgb",
              int device_id = -1,
              std::string* error = nullptr);

    void deinit();

    // Fill detection.feature for each detection. Detections with failed feature extraction are dropped.
    bool getRectsFeature(const SimpleCV::Mat& bgr, DETECTIONS& d, std::string* error = nullptr);

    int input_width() const noexcept { return input_w_; }
    int input_height() const noexcept { return input_h_; }

private:
    FeatureTensor() = default;
    FeatureTensor(const FeatureTensor&) = delete;
    FeatureTensor& operator=(const FeatureTensor&) = delete;
    ~FeatureTensor() = default;

    static FeatureTensor* instance;

    bool inited_{false};
    int input_w_{0};
    int input_h_{0};
    deepsort::npu::ReidFeatureExtractor extractor_{};
};
