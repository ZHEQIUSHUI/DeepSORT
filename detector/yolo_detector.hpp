#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "SimpleCV.hpp"

namespace deepsort {
namespace detector {

enum class BackendType {
    kAuto = 0,
    kAx650,
    kAxcl,
};

enum class YoloModelType {
    kYolov5 = 0,
    kYolov8 = 1,  // YOLOv8 native (DFL)
};

struct Detection {
    float x0{0};
    float y0{0};
    float x1{0};
    float y1{0};
    float score{0};
    int class_id{-1};
};

struct YoloPostprocessOptions {
    YoloModelType model_type{YoloModelType::kYolov5};
    int num_classes{80};
    float conf_threshold{0.25F};
    float nms_threshold{0.45F};

    // Common YOLO strides, ordered to match output tensors.
    std::vector<int> strides{8, 16, 32};

    // YOLOv5 anchors in pixels, order is [stride8(3 pairs), stride16(3 pairs), stride32(3 pairs)].
    std::vector<float> yolov5_anchors = {
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326,
    };

    // YOLOv8 native DFL reg_max.
    int yolov8_reg_max{16};
};

struct YoloDetectorOptions {
    BackendType backend{BackendType::kAuto};
    int device_id{-1};  // for AXCL: device index in axclrtGetDeviceList
    std::string model_path;

    // Letterbox background fill in BGR(A) order.
    std::vector<unsigned char> background{114, 114, 114, 255};

    YoloPostprocessOptions post{};
};

enum class YoloInputFormat {
    kBgr24 = 0,
    kNv12 = 1,
};

class YoloDetector {
public:
    ~YoloDetector();

    bool Init(const YoloDetectorOptions& opt, std::string* error);
    void Deinit();

    bool Detect(const SimpleCV::Mat& bgr, std::vector<Detection>* out, std::string* error);
    // Device-input inference for AXCL (zero-copy style).
    // The device buffer must already be in the model's expected input format and size.
    bool DetectFromDevice(std::uint64_t device_addr,
                          std::size_t device_bytes,
                          int src_w,
                          int src_h,
                          std::vector<Detection>* out,
                          std::string* error);

    int input_width() const noexcept { return input_w_; }
    int input_height() const noexcept { return input_h_; }
    YoloInputFormat input_format() const noexcept { return input_fmt_; }

private:
    struct Impl;
    struct ImplDeleter {
        void operator()(Impl* p) noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_{};

    YoloDetectorOptions opt_{};
    int input_w_{0};
    int input_h_{0};
    YoloInputFormat input_fmt_{YoloInputFormat::kBgr24};
};

}  // namespace detector
}  // namespace deepsort
