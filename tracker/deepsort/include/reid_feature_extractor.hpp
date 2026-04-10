#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "SimpleCV.hpp"

namespace deepsort::npu {

enum class BackendType {
    kAuto = 0,
    kAx650,
    kAxcl,
};

enum class ColorOrder {
    kBgr = 0,
    kRgb,
};

struct ReidFeatureExtractorOptions {
    BackendType backend{BackendType::kAuto};
    int device_id{-1};
    std::string model_path;

    // Model expects uint8 pixels in given channel order.
    ColorOrder color_order{ColorOrder::kRgb};

    // Postprocess
    bool l2_normalize{true};
};

class ReidFeatureExtractor {
public:
    ~ReidFeatureExtractor();

    bool Init(const ReidFeatureExtractorOptions& opt, std::string* error);
    void Deinit();

    // Input: BGR uint8 (SimpleCV default). Internally resized to model input resolution.
    // Output: 512-D float feature (size determined from model output tensor).
    bool Extract(const SimpleCV::Mat& bgr, std::vector<float>* feature, std::string* error);

    // Fast path: the input is already in model expected layout and dtype, and lives in device memory.
    // Currently supported for AXCL backend only.
    bool ExtractFromDevice(std::uint64_t device_addr,
                           std::size_t device_bytes,
                           std::vector<float>* feature,
                           std::string* error);

    int input_width() const noexcept { return input_w_; }
    int input_height() const noexcept { return input_h_; }
    int feature_dim() const noexcept { return feature_dim_; }

private:
    struct Impl;
    struct ImplDeleter {
        void operator()(Impl* p) noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_{};

    ReidFeatureExtractorOptions opt_{};
    int input_w_{0};
    int input_h_{0};
    int input_c_{0};
    bool input_nhwc_{true};
    int feature_dim_{0};
};

}  // namespace deepsort::npu
