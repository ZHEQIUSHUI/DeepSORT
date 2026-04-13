#include "reid_feature_extractor.hpp"

#include "npu/runner/ax_model_runner.hpp"
#if defined(DEEPSORT_HAVE_AX650)
#include "npu/runner/ax650/ax_model_runner_ax650.hpp"
#endif
#if defined(DEEPSORT_HAVE_AXCL)
#include "npu/runner/axcl/ax_model_runner_axcl.hpp"
#endif

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

namespace deepsort::npu {

namespace {

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

std::size_t Product(const std::vector<unsigned int>& dims) {
    std::size_t p = 1;
    for (auto d : dims) p *= static_cast<std::size_t>(d);
    return p;
}

struct InputLayoutSpec {
    int w{0};
    int h{0};
    int c{0};
    bool nhwc{true};
};

bool InferInputLayout(const ax_runner_tensor_t& in, InputLayoutSpec* out, std::string* error) {
    if (!out) return false;
    *out = {};
    if (in.vShape.empty()) {
        if (error) *error = "input shape is empty";
        return false;
    }
    if (in.nSize <= 0) {
        if (error) *error = "input buffer size is invalid";
        return false;
    }

    if (in.vShape.size() == 4) {
        const int d0 = static_cast<int>(in.vShape[0]);
        const int d1 = static_cast<int>(in.vShape[1]);
        const int d2 = static_cast<int>(in.vShape[2]);
        const int d3 = static_cast<int>(in.vShape[3]);
        (void)d0;

        if (d3 == 3) {
            out->nhwc = true;
            out->h = d1;
            out->w = d2;
            out->c = d3;
            return true;
        }
        if (d1 == 3) {
            out->nhwc = false;
            out->c = d1;
            out->h = d2;
            out->w = d3;
            return true;
        }
        if (error) *error = "unsupported input shape (need NHWC or NCHW with 3 channels)";
        return false;
    }

    if (in.vShape.size() == 3) {
        // HWC
        const int h = static_cast<int>(in.vShape[0]);
        const int w = static_cast<int>(in.vShape[1]);
        const int c = static_cast<int>(in.vShape[2]);
        if (c != 3) {
            if (error) *error = "unsupported input channels: " + std::to_string(c);
            return false;
        }
        out->nhwc = true;
        out->h = h;
        out->w = w;
        out->c = c;
        return true;
    }

    if (error) *error = "unsupported input rank: " + std::to_string(in.vShape.size());
    return false;
}

// half -> float (IEEE 754)
float HalfToFloat(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000U) << 16U;
    const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10U) & 0x1FU;
    const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFU;

    std::uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            std::uint32_t m = mant;
            std::uint32_t e = 0;
            while ((m & 0x400U) == 0) {
                m <<= 1U;
                e++;
            }
            m &= 0x3FFU;
            const std::uint32_t exp_f = (127U - 15U - e) << 23U;
            const std::uint32_t mant_f = m << 13U;
            f = sign | exp_f | mant_f;
        }
    } else if (exp == 31) {
        // Inf/NaN
        f = sign | 0x7F800000U | (mant << 13U);
    } else {
        // normal
        const std::uint32_t exp_f = (exp + (127U - 15U)) << 23U;
        const std::uint32_t mant_f = mant << 13U;
        f = sign | exp_f | mant_f;
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

bool PackToModelInput(const SimpleCV::Mat& bgr,
                      const InputLayoutSpec& spec,
                      ColorOrder model_color,
                      void* dst,
                      std::size_t dst_bytes,
                      std::string* error) {
    if (!dst) return false;
    if (bgr.empty() || bgr.data == nullptr) {
        if (error) *error = "empty input image";
        return false;
    }
    if (bgr.channels != 3) {
        if (error) *error = "expected 3-channel BGR image";
        return false;
    }
    if (spec.w <= 0 || spec.h <= 0 || spec.c != 3) {
        if (error) *error = "invalid model input spec";
        return false;
    }

    SimpleCV::Mat resized;
    SimpleCV::resize(bgr, resized, spec.w, spec.h);
    if (resized.width != spec.w || resized.height != spec.h || resized.channels != 3) {
        if (error) *error = "resize failed";
        return false;
    }

    const std::size_t need = static_cast<std::size_t>(spec.w) * static_cast<std::size_t>(spec.h) * 3U;
    if (dst_bytes < need) {
        if (error) *error = "input buffer too small";
        return false;
    }

    const bool want_rgb = (model_color == ColorOrder::kRgb);
    const int row_bytes = spec.w * 3;
    const std::uint8_t* src0 = resized.data;
    auto* out_u8 = static_cast<std::uint8_t*>(dst);

    if (spec.nhwc) {
        // NHWC packed
        if (!want_rgb) {
            // BGR -> BGR, just copy row-wise
            if (resized.step == row_bytes) {
                std::memcpy(out_u8, src0, need);
                return true;
            }
            for (int y = 0; y < spec.h; ++y) {
                std::memcpy(out_u8 + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes),
                            src0 + static_cast<std::size_t>(y) * static_cast<std::size_t>(resized.step),
                            static_cast<std::size_t>(row_bytes));
            }
            return true;
        }

        // BGR -> RGB swizzle
        for (int y = 0; y < spec.h; ++y) {
            const std::uint8_t* src = src0 + static_cast<std::size_t>(y) * static_cast<std::size_t>(resized.step);
            std::uint8_t* dst_row = out_u8 + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes);
            for (int x = 0; x < spec.w; ++x) {
                const std::uint8_t b = src[x * 3 + 0];
                const std::uint8_t g = src[x * 3 + 1];
                const std::uint8_t r = src[x * 3 + 2];
                dst_row[x * 3 + 0] = r;
                dst_row[x * 3 + 1] = g;
                dst_row[x * 3 + 2] = b;
            }
        }
        return true;
    }

    // NCHW planar
    std::uint8_t* plane0 = out_u8;
    std::uint8_t* plane1 = out_u8 + static_cast<std::size_t>(spec.w) * static_cast<std::size_t>(spec.h);
    std::uint8_t* plane2 = plane1 + static_cast<std::size_t>(spec.w) * static_cast<std::size_t>(spec.h);

    for (int y = 0; y < spec.h; ++y) {
        const std::uint8_t* src = src0 + static_cast<std::size_t>(y) * static_cast<std::size_t>(resized.step);
        for (int x = 0; x < spec.w; ++x) {
            const std::uint8_t b = src[x * 3 + 0];
            const std::uint8_t g = src[x * 3 + 1];
            const std::uint8_t r = src[x * 3 + 2];
            const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(spec.w) + static_cast<std::size_t>(x);
            if (want_rgb) {
                plane0[idx] = r;
                plane1[idx] = g;
                plane2[idx] = b;
            } else {
                plane0[idx] = b;
                plane1[idx] = g;
                plane2[idx] = r;
            }
        }
    }
    return true;
}

void L2Normalize(std::vector<float>* v) {
    if (!v || v->empty()) return;
    double sum = 0.0;
    for (float x : *v) sum += static_cast<double>(x) * static_cast<double>(x);
    const double norm = std::sqrt(std::max(0.0, sum));
    if (norm <= 1e-12) return;
    const float inv = static_cast<float>(1.0 / norm);
    for (auto& x : *v) x *= inv;
}

}  // namespace

struct ReidFeatureExtractor::Impl {
    BackendType backend{BackendType::kAuto};
    std::shared_ptr<ax_runner_base> runner{};
    std::vector<char> model_bytes{};
    InputLayoutSpec input{};
    int feature_dim{0};
    bool out_fp16{false};
};

void ReidFeatureExtractor::ImplDeleter::operator()(Impl* p) noexcept {
    delete p;
}

ReidFeatureExtractor::~ReidFeatureExtractor() {
    Deinit();
}

bool ReidFeatureExtractor::Init(const ReidFeatureExtractorOptions& opt, std::string* error) {
    Deinit();
    opt_ = opt;

    if (opt_.model_path.empty()) {
        if (error) *error = "model_path is empty";
        return false;
    }

    impl_.reset(new Impl());
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

    const int ret = impl_->runner->init(impl_->model_bytes.data(),
                                        static_cast<unsigned int>(impl_->model_bytes.size()),
                                        opt_.device_id);
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
    if (impl_->runner->get_num_outputs() < 1) {
        if (error) *error = "model has no outputs";
        Deinit();
        return false;
    }

    std::string infer_err;
    if (!InferInputLayout(impl_->runner->get_input(0), &impl_->input, &infer_err)) {
        if (error) *error = "infer input layout failed: " + infer_err;
        Deinit();
        return false;
    }
    input_w_ = impl_->input.w;
    input_h_ = impl_->input.h;
    input_c_ = impl_->input.c;
    input_nhwc_ = impl_->input.nhwc;

    const auto& out0 = impl_->runner->get_output(0);
    const std::size_t elems = Product(out0.vShape);
    if (elems == 0 || out0.nSize <= 0) {
        if (error) *error = "invalid output meta";
        Deinit();
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(out0.nSize);
    if (bytes == elems * sizeof(float)) {
        impl_->out_fp16 = false;
    } else if (bytes == elems * sizeof(std::uint16_t)) {
        impl_->out_fp16 = true;
    } else {
        if (error) *error = "unsupported output dtype (bytes=" + std::to_string(bytes) +
                            ", elems=" + std::to_string(elems) + ")";
        Deinit();
        return false;
    }

    impl_->feature_dim = static_cast<int>(elems);
    feature_dim_ = impl_->feature_dim;
    return true;
}

void ReidFeatureExtractor::Deinit() {
    if (impl_ && impl_->runner) {
        impl_->runner->deinit();
    }
    impl_.reset();
    opt_ = {};
    input_w_ = 0;
    input_h_ = 0;
    input_c_ = 0;
    input_nhwc_ = true;
    feature_dim_ = 0;
}

bool ReidFeatureExtractor::Extract(const SimpleCV::Mat& bgr, std::vector<float>* feature, std::string* error) {
    if (!impl_ || !impl_->runner) {
        if (error) *error = "extractor not initialized";
        return false;
    }
    if (!feature) return false;
    feature->clear();

    if (bgr.empty() || bgr.data == nullptr || bgr.width <= 0 || bgr.height <= 0) {
        if (error) *error = "invalid input image";
        return false;
    }

    auto& runner = *impl_->runner;
    const auto& in = runner.get_input(0);
    if (in.pVirAddr == nullptr || in.nSize <= 0) {
        if (error) *error = "runner input buffer invalid";
        return false;
    }

    std::string pack_err;
    if (!PackToModelInput(bgr, impl_->input, opt_.color_order, in.pVirAddr, static_cast<std::size_t>(in.nSize), &pack_err)) {
        if (error) *error = "pack input failed: " + pack_err;
        return false;
    }

    const int ret = runner.inference();
    if (ret != 0) {
        if (error) *error = "runner inference failed ret=" + std::to_string(ret);
        return false;
    }

    const auto& out0 = runner.get_output(0);
    if (out0.pVirAddr == nullptr || out0.nSize <= 0) {
        if (error) *error = "runner output buffer invalid";
        return false;
    }

    feature->resize(static_cast<std::size_t>(impl_->feature_dim));
    if (!impl_->out_fp16) {
        std::memcpy(feature->data(), out0.pVirAddr, feature->size() * sizeof(float));
    } else {
        const auto* h = static_cast<const std::uint16_t*>(out0.pVirAddr);
        for (std::size_t i = 0; i < feature->size(); ++i) {
            (*feature)[i] = HalfToFloat(h[i]);
        }
    }

    if (opt_.l2_normalize) {
        L2Normalize(feature);
    }
    return true;
}

bool ReidFeatureExtractor::ExtractFromDevice(std::uint64_t device_addr,
                                            std::size_t device_bytes,
                                            std::vector<float>* feature,
                                            std::string* error) {
    if (!impl_ || !impl_->runner) {
        if (error) *error = "extractor not initialized";
        return false;
    }
    if (!feature) return false;
    feature->clear();

    if (device_addr == 0 || device_bytes == 0) {
        if (error) *error = "invalid device buffer";
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

#if !defined(DEEPSORT_HAVE_AXCL) && !defined(DEEPSORT_HAVE_AX650)
    if (error) *error = "device input not enabled in this build";
    return false;
#endif

    bool bound = false;

#if defined(DEEPSORT_HAVE_AXCL)
    auto* axcl_runner = dynamic_cast<ax_runner_axcl*>(impl_->runner.get());
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

    if (axcl_runner != nullptr) {
        axcl_guard.r = axcl_runner;
        axcl_guard.prev_before = axcl_runner->auto_sync_before_inference();
        axcl_guard.default_dev = static_cast<std::uint64_t>(in.phyAddr);
        axcl_guard.size = static_cast<unsigned long>(in.nSize);

        axcl_runner->set_auto_sync_before_inference(false);
        if (axcl_runner->set_input(0, 0, device_addr, axcl_guard.size) != 0) {
            if (error) *error = "axcl set_input failed";
            return false;
        }
        bound = true;
    }
#endif

#if defined(DEEPSORT_HAVE_AX650)
    auto* ax650_runner = dynamic_cast<ax_runner_ax650*>(impl_->runner.get());
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

    if (!bound && ax650_runner != nullptr) {
        ax650_guard.r = ax650_runner;
        ax650_guard.default_phy = static_cast<std::uint64_t>(in.phyAddr);
        ax650_guard.size = static_cast<unsigned long>(in.nSize);

        if (ax650_runner->set_input(0, 0, device_addr, ax650_guard.size) != 0) {
            if (error) *error = "ax650 set_input failed";
            return false;
        }
        bound = true;
    }
#endif

    if (!bound) {
        if (error) *error = "device input only supported by AXCL/AX650 backends";
        return false;
    }

    const int ret = runner.inference();
    if (ret != 0) {
        if (error) *error = "runner inference failed ret=" + std::to_string(ret);
        return false;
    }

    const auto& out0 = runner.get_output(0);
    if (out0.pVirAddr == nullptr || out0.nSize <= 0) {
        if (error) *error = "runner output buffer invalid";
        return false;
    }

    feature->resize(static_cast<std::size_t>(impl_->feature_dim));
    if (!impl_->out_fp16) {
        std::memcpy(feature->data(), out0.pVirAddr, feature->size() * sizeof(float));
    } else {
        const auto* h = static_cast<const std::uint16_t*>(out0.pVirAddr);
        for (std::size_t i = 0; i < feature->size(); ++i) {
            (*feature)[i] = HalfToFloat(h[i]);
        }
    }

    if (opt_.l2_normalize) {
        L2Normalize(feature);
    }
    return true;
}

}  // namespace deepsort::npu
