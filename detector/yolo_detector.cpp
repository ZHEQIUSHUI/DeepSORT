#include "detector/yolo_detector.hpp"

#include "npu/runner/ax_model_runner.hpp"
#include "npu/runner/ax650/ax_model_runner_ax650.hpp"
#include "npu/runner/axcl/ax_model_runner_axcl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace deepsort {
namespace detector {

namespace {

struct LetterboxInfo {
    float scale{1.0F};
    float pad_x{0.0F};
    float pad_y{0.0F};
    int dst_w{0};
    int dst_h{0};
};

LetterboxInfo ComputeLetterbox(int src_w, int src_h, int dst_w, int dst_h) {
    LetterboxInfo lb{};
    lb.dst_w = dst_w;
    lb.dst_h = dst_h;
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return lb;

    const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
    lb.scale = std::min(sx, sy);

    const int new_w = static_cast<int>(std::round(static_cast<float>(src_w) * lb.scale));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src_h) * lb.scale));
    const int pad_w = std::max(0, dst_w - new_w);
    const int pad_h = std::max(0, dst_h - new_h);

    const int pad_x0 = pad_w / 2;
    const int pad_y0 = pad_h / 2;
    lb.pad_x = static_cast<float>(pad_x0);
    lb.pad_y = static_cast<float>(pad_y0);
    return lb;
}

void UndoLetterbox(const LetterboxInfo& lb, int src_w, int src_h, std::vector<Detection>* dets) {
    if (!dets || dets->empty()) return;
    if (lb.scale <= 0.0F) return;

    for (auto& d : *dets) {
        d.x0 = (d.x0 - lb.pad_x) / lb.scale;
        d.y0 = (d.y0 - lb.pad_y) / lb.scale;
        d.x1 = (d.x1 - lb.pad_x) / lb.scale;
        d.y1 = (d.y1 - lb.pad_y) / lb.scale;

        if (d.x1 < d.x0) std::swap(d.x0, d.x1);
        if (d.y1 < d.y0) std::swap(d.y0, d.y1);

        d.x0 = std::max(0.0F, std::min(d.x0, static_cast<float>(src_w)));
        d.y0 = std::max(0.0F, std::min(d.y0, static_cast<float>(src_h)));
        d.x1 = std::max(0.0F, std::min(d.x1, static_cast<float>(src_w)));
        d.y1 = std::max(0.0F, std::min(d.y1, static_cast<float>(src_h)));
    }
}

bool ReadFileToBuffer(const std::string& path, std::vector<char>* out, std::string* error) {
    if (!out) return false;
    out->clear();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (error) *error = "open failed: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const auto end = ifs.tellg();
    if (end <= 0) {
        if (error) *error = "empty file: " + path;
        return false;
    }
    out->resize(static_cast<std::size_t>(end));
    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(out->data(), static_cast<std::streamsize>(out->size()))) {
        if (error) *error = "read failed: " + path;
        return false;
    }
    return true;
}

struct InputSpec {
    int w{0};
    int h{0};
    enum class Format { kUnknown = 0, kBgr24, kNv12 } fmt{Format::kUnknown};
};

InputSpec InferInputSpec(const ax_runner_tensor_t& in) {
    InputSpec spec{};
    const std::size_t bytes = (in.nSize > 0) ? static_cast<std::size_t>(in.nSize) : 0U;
    const auto try_set = [&](int w, int h) {
        if (spec.w != 0 || spec.h != 0) return;
        if (w <= 0 || h <= 0) return;
        const std::size_t wh = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        if (wh == 0) return;
        if (bytes == wh * 3U) {
            spec.w = w;
            spec.h = h;
            spec.fmt = InputSpec::Format::kBgr24;
            return;
        }
        if (bytes == wh * 3U / 2U) {
            spec.w = w;
            spec.h = h;
            spec.fmt = InputSpec::Format::kNv12;
            return;
        }
    };

    if (in.vShape.size() == 4) {
        // Prefer NHWC candidates.
        try_set(static_cast<int>(in.vShape[2]), static_cast<int>(in.vShape[1]));
        // Fallback NCHW.
        try_set(static_cast<int>(in.vShape[3]), static_cast<int>(in.vShape[2]));
    } else if (in.vShape.size() == 3) {
        // HWC
        try_set(static_cast<int>(in.vShape[1]), static_cast<int>(in.vShape[0]));
    }

    if (spec.w == 0 || spec.h == 0) {
        // Last resort: assume 640x640 packed.
        spec.w = 640;
        spec.h = 640;
        if (bytes == static_cast<std::size_t>(spec.w) * static_cast<std::size_t>(spec.h) * 3U) {
            spec.fmt = InputSpec::Format::kBgr24;
        } else if (bytes == static_cast<std::size_t>(spec.w) * static_cast<std::size_t>(spec.h) * 3U / 2U) {
            spec.fmt = InputSpec::Format::kNv12;
        }
    }
    return spec;
}

bool CopyBgrToPacked(const SimpleCV::Mat& bgr, void* dst, std::size_t dst_bytes, std::string* error) {
    if (!dst) return false;
    if (bgr.data == nullptr || bgr.width <= 0 || bgr.height <= 0) {
        if (error) *error = "invalid input mat";
        return false;
    }
    if (bgr.channels != 3) {
        if (error) *error = "expected BGR mat with 3 channels";
        return false;
    }
    const std::size_t need = static_cast<std::size_t>(bgr.width) * static_cast<std::size_t>(bgr.height) * 3U;
    if (dst_bytes < need) {
        if (error) *error = "runner input buffer too small";
        return false;
    }
    auto* out = static_cast<std::uint8_t*>(dst);
    const int row_bytes = bgr.width * 3;
    if (bgr.step == row_bytes) {
        std::memcpy(out, bgr.data, need);
        return true;
    }
    for (int y = 0; y < bgr.height; ++y) {
        std::memcpy(out + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes),
                    bgr.data + static_cast<std::size_t>(y) * static_cast<std::size_t>(bgr.step),
                    static_cast<std::size_t>(row_bytes));
    }
    return true;
}

// Simple CPU BGR->NV12 conversion (BT.601 limited-range-like).
bool ConvertBgrToNv12(const SimpleCV::Mat& bgr, void* dst, std::size_t dst_bytes, std::string* error) {
    if (!dst) return false;
    if (bgr.data == nullptr || bgr.width <= 0 || bgr.height <= 0) {
        if (error) *error = "invalid input mat";
        return false;
    }
    if (bgr.channels != 3) {
        if (error) *error = "expected BGR mat with 3 channels";
        return false;
    }
    const int w = bgr.width;
    const int h = bgr.height;
    const std::size_t need = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U / 2U;
    if (dst_bytes < need) {
        if (error) *error = "runner input buffer too small";
        return false;
    }
    auto* y_plane = static_cast<std::uint8_t*>(dst);
    auto* uv_plane = y_plane + static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

    // Y plane
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = bgr.data + static_cast<std::size_t>(y) * static_cast<std::size_t>(bgr.step);
        for (int x = 0; x < w; ++x) {
            const int b = row[x * 3 + 0];
            const int g = row[x * 3 + 1];
            const int r = row[x * 3 + 2];
            int Y = (66 * r + 129 * g + 25 * b + 128) >> 8;
            Y += 16;
            Y = std::max(0, std::min(255, Y));
            y_plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] =
                static_cast<std::uint8_t>(Y);
        }
    }

    // UV plane (2x2 subsampling)
    for (int y = 0; y < h; y += 2) {
        const std::uint8_t* row0 = bgr.data + static_cast<std::size_t>(y) * static_cast<std::size_t>(bgr.step);
        const std::uint8_t* row1 = (y + 1 < h)
                                       ? (bgr.data + static_cast<std::size_t>(y + 1) * static_cast<std::size_t>(bgr.step))
                                       : row0;
        for (int x = 0; x < w; x += 2) {
            // Average 2x2
            int r = 0, g = 0, b = 0, cnt = 0;
            const std::uint8_t* p00 = row0 + x * 3;
            r += p00[2]; g += p00[1]; b += p00[0]; cnt++;
            if (x + 1 < w) {
                const std::uint8_t* p01 = row0 + (x + 1) * 3;
                r += p01[2]; g += p01[1]; b += p01[0]; cnt++;
            }
            if (y + 1 < h) {
                const std::uint8_t* p10 = row1 + x * 3;
                r += p10[2]; g += p10[1]; b += p10[0]; cnt++;
                if (x + 1 < w) {
                    const std::uint8_t* p11 = row1 + (x + 1) * 3;
                    r += p11[2]; g += p11[1]; b += p11[0]; cnt++;
                }
            }
            r /= cnt; g /= cnt; b /= cnt;
            int U = (-38 * r - 74 * g + 112 * b + 128) >> 8;
            int V = (112 * r - 94 * g - 18 * b + 128) >> 8;
            U += 128;
            V += 128;
            U = std::max(0, std::min(255, U));
            V = std::max(0, std::min(255, V));
            const std::size_t uv_index = static_cast<std::size_t>(y / 2) * static_cast<std::size_t>(w) +
                                         static_cast<std::size_t>(x);
            uv_plane[uv_index + 0] = static_cast<std::uint8_t>(U);
            if (x + 1 < w) uv_plane[uv_index + 1] = static_cast<std::uint8_t>(V);
        }
    }
    return true;
}

struct TensorView {
    const float* data{nullptr};
    std::vector<unsigned int> shape;
    std::string name;
    std::size_t bytes{0};
};

enum class Layout {
    kUnknown = 0,
    kNHWC,
    kNCHW,
};

inline float Sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

inline float Clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

inline float Logit(float p) {
    const float pp = Clamp(p, 1e-6F, 1.0F - 1e-6F);
    return std::log(pp / (1.0F - pp));
}

float IoU(const Detection& a, const Detection& b) {
    const float x0 = std::max(a.x0, b.x0);
    const float y0 = std::max(a.y0, b.y0);
    const float x1 = std::min(a.x1, b.x1);
    const float y1 = std::min(a.y1, b.y1);
    const float w = std::max(0.0F, x1 - x0);
    const float h = std::max(0.0F, y1 - y0);
    const float inter = w * h;
    const float area_a = std::max(0.0F, a.x1 - a.x0) * std::max(0.0F, a.y1 - a.y0);
    const float area_b = std::max(0.0F, b.x1 - b.x0) * std::max(0.0F, b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0F) return 0.0F;
    return inter / uni;
}

void Nms(std::vector<Detection>* dets, float nms_threshold) {
    if (!dets || dets->empty()) return;
    std::sort(dets->begin(), dets->end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    keep.reserve(dets->size());
    for (const auto& d : *dets) {
        bool ok = true;
        for (const auto& k : keep) {
            if (IoU(d, k) > nms_threshold) {
                ok = false;
                break;
            }
        }
        if (ok) keep.push_back(d);
    }
    *dets = std::move(keep);
}

struct FeatureView {
    const float* ptr{nullptr};
    Layout layout{Layout::kUnknown};
    int feat_h{0};
    int feat_w{0};
    int channels{0};
};

bool MakeFeatureView(const TensorView& t, int expected_channels, FeatureView* out) {
    if (!out) return false;
    *out = {};
    if (t.data == nullptr) return false;
    if (t.shape.size() != 4) return false;

    const int d1 = static_cast<int>(t.shape[1]);
    const int d2 = static_cast<int>(t.shape[2]);
    const int d3 = static_cast<int>(t.shape[3]);

    if (d1 == expected_channels) {
        out->layout = Layout::kNCHW;
        out->channels = d1;
        out->feat_h = d2;
        out->feat_w = d3;
        out->ptr = t.data;
        return out->feat_h > 0 && out->feat_w > 0;
    }
    if (d3 == expected_channels) {
        out->layout = Layout::kNHWC;
        out->channels = d3;
        out->feat_h = d1;
        out->feat_w = d2;
        out->ptr = t.data;
        return out->feat_h > 0 && out->feat_w > 0;
    }

    // Fallback: assume NCHW if ambiguous.
    out->layout = Layout::kNCHW;
    out->channels = d1;
    out->feat_h = d2;
    out->feat_w = d3;
    out->ptr = t.data;
    return out->feat_h > 0 && out->feat_w > 0;
}

inline float At(const FeatureView& tv, int h, int w, int c) {
    if (tv.layout == Layout::kNHWC) {
        return tv.ptr[(h * tv.feat_w + w) * tv.channels + c];
    }
    return tv.ptr[(c * tv.feat_h + h) * tv.feat_w + w];
}

bool DecodeYolov5One(const FeatureView& tv,
                     int stride,
                     const float* anchors6,
                     int num_classes,
                     float conf_thr,
                     std::vector<Detection>* out) {
    if (!anchors6 || !out) return false;
    const int A = 3;
    const int step = num_classes + 5;
    if (tv.channels != A * step) return false;

    const float obj_logit_thr = Logit(conf_thr);
    for (int h = 0; h < tv.feat_h; ++h) {
        for (int w = 0; w < tv.feat_w; ++w) {
            for (int a = 0; a < A; ++a) {
                const int base = a * step;
                const float obj_logit = At(tv, h, w, base + 4);
                if (obj_logit < obj_logit_thr) continue;

                int best_cls = 0;
                float best_cls_logit = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < num_classes; ++c) {
                    const float v = At(tv, h, w, base + 5 + c);
                    if (v > best_cls_logit) {
                        best_cls_logit = v;
                        best_cls = c;
                    }
                }
                if (best_cls_logit < obj_logit_thr) continue;

                const float obj = Sigmoid(obj_logit);
                const float score = obj * Sigmoid(best_cls_logit);
                if (score < conf_thr) continue;

                const float dx = Sigmoid(At(tv, h, w, base + 0));
                const float dy = Sigmoid(At(tv, h, w, base + 1));
                const float dw = Sigmoid(At(tv, h, w, base + 2));
                const float dh = Sigmoid(At(tv, h, w, base + 3));

                const float cx = (dx * 2.0F - 0.5F + static_cast<float>(w)) * static_cast<float>(stride);
                const float cy = (dy * 2.0F - 0.5F + static_cast<float>(h)) * static_cast<float>(stride);

                const float aw = anchors6[a * 2 + 0];
                const float ah = anchors6[a * 2 + 1];
                const float bw = dw * dw * 4.0F * aw;
                const float bh = dh * dh * 4.0F * ah;

                Detection det{};
                det.x0 = cx - bw * 0.5F;
                det.y0 = cy - bh * 0.5F;
                det.x1 = cx + bw * 0.5F;
                det.y1 = cy + bh * 0.5F;
                det.score = score;
                det.class_id = best_cls;
                out->push_back(det);
            }
        }
    }
    return true;
}

bool DecodeYolov8NativeOne(const FeatureView& tv,
                           int stride,
                           int num_classes,
                           int reg_max,
                           float conf_thr,
                           std::vector<Detection>* out) {
    if (!out) return false;
    const int step = num_classes + 4 * reg_max;
    if (tv.channels != step) return false;

    const int cls_offset = 4 * reg_max;
    const float logit_thr = Logit(conf_thr);
    for (int h = 0; h < tv.feat_h; ++h) {
        for (int w = 0; w < tv.feat_w; ++w) {
            int best_cls = 0;
            float best_logit = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < num_classes; ++c) {
                const float v = At(tv, h, w, cls_offset + c);
                if (v > best_logit) {
                    best_logit = v;
                    best_cls = c;
                }
            }
            if (best_logit < logit_thr) continue;
            const float score = Sigmoid(best_logit);
            if (score < conf_thr) continue;

            float ltrb[4];
            for (int k = 0; k < 4; ++k) {
                const int base = k * reg_max;
                float alpha = At(tv, h, w, base);
                for (int i = 1; i < reg_max; ++i) {
                    alpha = std::max(alpha, At(tv, h, w, base + i));
                }
                float expsum = 0.0F;
                float exwsum = 0.0F;
                for (int i = 0; i < reg_max; ++i) {
                    const float raw = At(tv, h, w, base + i);
                    const float e = std::exp(raw - alpha);
                    expsum += e;
                    exwsum += static_cast<float>(i) * e;
                }
                const float dis = (expsum <= 0.0F) ? 0.0F : (exwsum / expsum);
                ltrb[k] = dis * static_cast<float>(stride);
            }

            const float cx = (static_cast<float>(w) + 0.5F) * static_cast<float>(stride);
            const float cy = (static_cast<float>(h) + 0.5F) * static_cast<float>(stride);

            Detection det{};
            det.x0 = cx - ltrb[0];
            det.y0 = cy - ltrb[1];
            det.x1 = cx + ltrb[2];
            det.y1 = cy + ltrb[3];
            det.score = score;
            det.class_id = best_cls;
            out->push_back(det);
        }
    }
    return true;
}

bool YoloPostprocess(const std::vector<TensorView>& outputs,
                     const YoloPostprocessOptions& opt,
                     const LetterboxInfo& lb,
                     int src_w,
                     int src_h,
                     std::vector<Detection>* out,
                     std::string* error) {
    if (!out) return false;
    out->clear();

    if (opt.strides.empty()) {
        if (error) *error = "strides is empty";
        return false;
    }
    if (outputs.size() != opt.strides.size()) {
        if (error) *error = "unexpected output tensor count: got " + std::to_string(outputs.size()) +
                            " expect " + std::to_string(opt.strides.size());
        return false;
    }

    std::vector<Detection> dets;
    dets.reserve(1024);

    if (opt.model_type == YoloModelType::kYolov5) {
        if (opt.yolov5_anchors.size() != opt.strides.size() * 6U) {
            if (error) *error = "yolov5_anchors size mismatch";
            return false;
        }
        const int cls = opt.num_classes;
        const float conf = opt.conf_threshold;
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const int stride = opt.strides[i];
            const int step = cls + 5;
            const int expected_ch = 3 * step;
            FeatureView tv{};
            if (!MakeFeatureView(outputs[i], expected_ch, &tv)) {
                if (error) *error = "invalid yolov5 output shape at index " + std::to_string(i);
                return false;
            }
            const float* anchors6 = opt.yolov5_anchors.data() + i * 6U;
            if (!DecodeYolov5One(tv, stride, anchors6, cls, conf, &dets)) {
                if (error) *error = "DecodeYolov5One failed at index " + std::to_string(i);
                return false;
            }
        }
    } else {
        const int cls = opt.num_classes;
        const int reg = opt.yolov8_reg_max;
        const int expected_ch = cls + 4 * reg;
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const int stride = opt.strides[i];
            FeatureView tv{};
            if (!MakeFeatureView(outputs[i], expected_ch, &tv)) {
                if (error) *error = "invalid yolov8 output shape at index " + std::to_string(i);
                return false;
            }
            if (!DecodeYolov8NativeOne(tv, stride, cls, reg, opt.conf_threshold, &dets)) {
                if (error) *error = "DecodeYolov8NativeOne failed at index " + std::to_string(i);
                return false;
            }
        }
    }

    Nms(&dets, opt.nms_threshold);
    UndoLetterbox(lb, src_w, src_h, &dets);
    *out = std::move(dets);
    return true;
}

}  // namespace

struct YoloDetector::Impl {
    BackendType backend{BackendType::kAuto};
    std::shared_ptr<ax_runner_base> runner{};
    std::vector<char> model_bytes{};
    InputSpec input{};
};

void YoloDetector::ImplDeleter::operator()(Impl* p) noexcept {
    delete p;
}

YoloDetector::~YoloDetector() {
    Deinit();
}

bool YoloDetector::Init(const YoloDetectorOptions& opt, std::string* error) {
    Deinit();
    opt_ = opt;

    if (opt_.model_path.empty()) {
        if (error) *error = "model_path is empty";
        return false;
    }

    impl_.reset(new Impl());
    impl_->backend = opt_.backend;

    if (!ReadFileToBuffer(opt_.model_path, &impl_->model_bytes, error)) {
        impl_.reset();
        return false;
    }

    BackendType backend = opt_.backend;
    if (backend == BackendType::kAuto) {
#if defined(DEEPSORT_HAVE_AXCL)
        backend = BackendType::kAxcl;
#elif defined(DEEPSORT_HAVE_AX650)
        backend = BackendType::kAx650;
#else
        if (error) *error = "no backend enabled (need DEEPSORT_HAVE_AXCL or DEEPSORT_HAVE_AX650)";
        impl_.reset();
        return false;
#endif
    }

    if (backend == BackendType::kAxcl) {
#if defined(DEEPSORT_HAVE_AXCL)
        impl_->runner = std::make_shared<ax_runner_axcl>();
#else
        if (error) *error = "AXCL backend not enabled in build";
        impl_.reset();
        return false;
#endif
    } else if (backend == BackendType::kAx650) {
#if defined(DEEPSORT_HAVE_AX650)
        impl_->runner = std::make_shared<ax_runner_ax650>();
#else
        if (error) *error = "AX650 backend not enabled in build";
        impl_.reset();
        return false;
#endif
    } else {
        if (error) *error = "unknown backend";
        impl_.reset();
        return false;
    }

    const int devid = opt_.device_id;
    const int ret = impl_->runner->init(impl_->model_bytes.data(),
                                        static_cast<unsigned int>(impl_->model_bytes.size()),
                                        devid);
    if (ret != 0) {
        if (error) *error = "runner init failed ret=" + std::to_string(ret);
        impl_.reset();
        return false;
    }

    if (impl_->runner->get_num_inputs() < 1) {
        if (error) *error = "model has no inputs";
        Deinit();
        return false;
    }

    const auto& in = impl_->runner->get_input(0);
    impl_->input = InferInputSpec(in);
    if (impl_->input.w <= 0 || impl_->input.h <= 0 || impl_->input.fmt == InputSpec::Format::kUnknown) {
        if (error) *error = "failed to infer model input spec";
        Deinit();
        return false;
    }
    input_w_ = impl_->input.w;
    input_h_ = impl_->input.h;
    input_fmt_ = (impl_->input.fmt == InputSpec::Format::kNv12) ? YoloInputFormat::kNv12 : YoloInputFormat::kBgr24;

    // quick sanity: output count matches postprocess settings
    if (impl_->runner->get_num_outputs() != static_cast<int>(opt_.post.strides.size())) {
        if (error) {
            *error = "output count mismatch: model has " + std::to_string(impl_->runner->get_num_outputs()) +
                     " outputs, but strides has " + std::to_string(opt_.post.strides.size());
        }
        Deinit();
        return false;
    }

    return true;
}

void YoloDetector::Deinit() {
    if (impl_ && impl_->runner) {
        impl_->runner->deinit();
    }
    impl_.reset();
    opt_ = {};
    input_w_ = 0;
    input_h_ = 0;
    input_fmt_ = YoloInputFormat::kBgr24;
}

bool YoloDetector::Detect(const SimpleCV::Mat& bgr, std::vector<Detection>* out, std::string* error) {
    if (!impl_ || !impl_->runner) {
        if (error) *error = "detector not initialized";
        return false;
    }
    if (!out) return false;
    out->clear();
    if (bgr.data == nullptr || bgr.width <= 0 || bgr.height <= 0) {
        if (error) *error = "invalid input image";
        return false;
    }

    SimpleCV::Mat src = bgr;
    if (src.channels != 3) {
        src = SimpleCV::cvtColor(bgr, SimpleCV::ColorSpace::BGR);
    }

    const LetterboxInfo lb = ComputeLetterbox(src.width, src.height, input_w_, input_h_);
    const int new_w = static_cast<int>(std::round(static_cast<float>(src.width) * lb.scale));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src.height) * lb.scale));
    const int pad_w = std::max(0, input_w_ - new_w);
    const int pad_h = std::max(0, input_h_ - new_h);
    const int left = pad_w / 2;
    const int right = pad_w - left;
    const int top = pad_h / 2;
    const int bottom = pad_h - top;

    SimpleCV::Mat resized;
    SimpleCV::resize(src, resized, new_w, new_h);
    SimpleCV::Mat letterboxed;
    SimpleCV::copyMakeBorder(resized, letterboxed, top, bottom, left, right,
                             SimpleCV::BorderType::CONSTANT, opt_.background);

    if (letterboxed.width != input_w_ || letterboxed.height != input_h_ || letterboxed.channels != 3) {
        if (error) *error = "letterbox result has unexpected shape";
        return false;
    }

    auto& runner = *impl_->runner;
    const auto& in = runner.get_input(0);
    if (in.pVirAddr == nullptr || in.nSize <= 0) {
        if (error) *error = "runner input buffer is invalid";
        return false;
    }

    std::string copy_err;
    bool ok = false;
    if (impl_->input.fmt == InputSpec::Format::kBgr24) {
        ok = CopyBgrToPacked(letterboxed, in.pVirAddr, static_cast<std::size_t>(in.nSize), &copy_err);
    } else if (impl_->input.fmt == InputSpec::Format::kNv12) {
        ok = ConvertBgrToNv12(letterboxed, in.pVirAddr, static_cast<std::size_t>(in.nSize), &copy_err);
    }
    if (!ok) {
        if (error) *error = "prepare input failed: " + copy_err;
        return false;
    }

    const int ret = runner.inference();
    if (ret != 0) {
        if (error) *error = "runner inference failed ret=" + std::to_string(ret);
        return false;
    }

    std::vector<TensorView> outputs;
    outputs.reserve(static_cast<std::size_t>(runner.get_num_outputs()));
    for (int i = 0; i < runner.get_num_outputs(); ++i) {
        const auto& t = runner.get_output(i);
        if (t.pVirAddr == nullptr || t.nSize <= 0) {
            if (error) *error = "invalid output buffer at index " + std::to_string(i);
            return false;
        }
        if ((t.nSize % static_cast<int>(sizeof(float))) != 0) {
            if (error) *error = "output tensor is not fp32 at index " + std::to_string(i);
            return false;
        }
        TensorView tv{};
        tv.data = reinterpret_cast<const float*>(t.pVirAddr);
        tv.shape = t.vShape;
        tv.name = t.sName;
        tv.bytes = static_cast<std::size_t>(t.nSize);
        outputs.push_back(std::move(tv));
    }

    std::string pp_err;
    if (!YoloPostprocess(outputs, opt_.post, lb, src.width, src.height, out, &pp_err)) {
        if (error) *error = "postprocess failed: " + pp_err;
        return false;
    }
    return true;
}

bool YoloDetector::DetectFromDevice(std::uint64_t device_addr,
                                    std::size_t device_bytes,
                                    int src_w,
                                    int src_h,
                                    std::vector<Detection>* out,
                                    std::string* error) {
    if (!impl_ || !impl_->runner) {
        if (error) *error = "detector not initialized";
        return false;
    }
    if (!out) return false;
    out->clear();
    if (device_addr == 0 || device_bytes == 0) {
        if (error) *error = "invalid device buffer";
        return false;
    }
    if (src_w <= 0 || src_h <= 0) {
        if (error) *error = "invalid source dimensions";
        return false;
    }

    ax_runner_axcl* axcl_runner = nullptr;
    ax_runner_ax650* ax650_runner = nullptr;
#if defined(DEEPSORT_HAVE_AXCL)
    axcl_runner = dynamic_cast<ax_runner_axcl*>(impl_->runner.get());
#endif
#if defined(DEEPSORT_HAVE_AX650)
    ax650_runner = dynamic_cast<ax_runner_ax650*>(impl_->runner.get());
#endif
    if (axcl_runner == nullptr && ax650_runner == nullptr) {
        if (error) *error = "device input only supported by AXCL/AX650 backends";
        return false;
    }

    auto& runner = *impl_->runner;
    const auto& in = runner.get_input(0);
    if (in.nSize <= 0) {
        if (error) *error = "runner input meta invalid";
        return false;
    }
    const std::size_t need = static_cast<std::size_t>(in.nSize);
    if (device_bytes < need) {
        if (error) *error = "device buffer too small";
        return false;
    }

    struct AxclRestoreGuard {
        ax_runner_axcl* r{nullptr};
        bool prev_before{true};
        std::uint64_t default_dev{0};
        unsigned long size{0};
        ~AxclRestoreGuard() {
            if (!r) return;
            if (default_dev != 0 && size != 0) {
                (void)r->set_input(0, 0, default_dev, size);
            }
            r->set_auto_sync_before_inference(prev_before);
        }
    } axcl_guard{};

    struct Ax650RestoreGuard {
        ax_runner_ax650* r{nullptr};
        std::uint64_t default_phy{0};
        unsigned long size{0};
        ~Ax650RestoreGuard() {
            if (!r) return;
            if (default_phy != 0 && size != 0) {
                (void)r->set_input(0, 0, default_phy, size);
            }
        }
    } ax650_guard{};

    if (axcl_runner != nullptr) {
#if defined(DEEPSORT_HAVE_AXCL)
        axcl_guard.r = axcl_runner;
        axcl_guard.prev_before = axcl_runner->auto_sync_before_inference();
        axcl_guard.default_dev = static_cast<std::uint64_t>(in.phyAddr);
        axcl_guard.size = static_cast<unsigned long>(in.nSize);

        axcl_runner->set_auto_sync_before_inference(false);
        if (axcl_runner->set_input(0, 0, device_addr, axcl_guard.size) != 0) {
            if (error) *error = "axcl set_input failed";
            return false;
        }
#endif
    } else if (ax650_runner != nullptr) {
#if defined(DEEPSORT_HAVE_AX650)
        ax650_guard.r = ax650_runner;
        ax650_guard.default_phy = static_cast<std::uint64_t>(in.phyAddr);
        ax650_guard.size = static_cast<unsigned long>(in.nSize);

        if (ax650_runner->set_input(0, 0, device_addr, ax650_guard.size) != 0) {
            if (error) *error = "ax650 set_input failed";
            return false;
        }
#endif
    }

    const int ret = runner.inference();
    if (ret != 0) {
        if (error) *error = "runner inference failed ret=" + std::to_string(ret);
        return false;
    }

    std::vector<TensorView> outputs;
    outputs.reserve(static_cast<std::size_t>(runner.get_num_outputs()));
    for (int i = 0; i < runner.get_num_outputs(); ++i) {
        const auto& t = runner.get_output(i);
        if (t.pVirAddr == nullptr || t.nSize <= 0) {
            if (error) *error = "invalid output buffer at index " + std::to_string(i);
            return false;
        }
        if ((t.nSize % static_cast<int>(sizeof(float))) != 0) {
            if (error) *error = "output tensor is not fp32 at index " + std::to_string(i);
            return false;
        }
        TensorView tv{};
        tv.data = reinterpret_cast<const float*>(t.pVirAddr);
        tv.shape = t.vShape;
        tv.name = t.sName;
        tv.bytes = static_cast<std::size_t>(t.nSize);
        outputs.push_back(std::move(tv));
    }

    const LetterboxInfo lb = ComputeLetterbox(src_w, src_h, input_w_, input_h_);
    std::string pp_err;
    if (!YoloPostprocess(outputs, opt_.post, lb, src_w, src_h, out, &pp_err)) {
        if (error) *error = "postprocess failed: " + pp_err;
        return false;
    }
    return true;
}

}  // namespace detector
}  // namespace deepsort
