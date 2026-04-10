/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 04:32:26
*/

#include "FeatureTensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

FeatureTensor* FeatureTensor::instance = nullptr;

FeatureTensor* FeatureTensor::getInstance() {
    if (instance == nullptr) {
        instance = new FeatureTensor();
    }
    return instance;
}

namespace {

int ClampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

}  // namespace

void FeatureTensor::deinit() {
    extractor_.Deinit();
    inited_ = false;
    input_w_ = 0;
    input_h_ = 0;
}

bool FeatureTensor::init(const std::string& model_path,
                         const std::string& color_order,
                         int device_id,
                         std::string* error) {
    deepsort::npu::ReidFeatureExtractorOptions opt{};
    opt.model_path = model_path;
    opt.device_id = device_id;
    opt.color_order = (color_order == "bgr") ? deepsort::npu::ColorOrder::kBgr : deepsort::npu::ColorOrder::kRgb;

    std::string err;
    if (!extractor_.Init(opt, &err)) {
        inited_ = false;
        if (error) *error = err;
        return false;
    }

    if (extractor_.feature_dim() != k_feature_dim) {
        extractor_.Deinit();
        inited_ = false;
        if (error) {
            *error = "unexpected feature dim: " + std::to_string(extractor_.feature_dim()) +
                     " (expected " + std::to_string(k_feature_dim) + ")";
        }
        return false;
    }

    input_w_ = extractor_.input_width();
    input_h_ = extractor_.input_height();
    inited_ = true;

    // AXCL runtime registers its own exit handler; ensure we deinit before that.
    static bool atexit_hooked = false;
    if (!atexit_hooked) {
        atexit_hooked = true;
        std::atexit([]() { FeatureTensor::getInstance()->deinit(); });
    }
    return true;
}

bool FeatureTensor::getRectsFeature(const SimpleCV::Mat& img, DETECTIONS& d, std::string* error) {
    if (!inited_) {
        if (error) *error = "FeatureTensor not initialized (call init first)";
        return false;
    }
    if (img.empty() || img.data == nullptr || img.width <= 0 || img.height <= 0) {
        if (error) *error = "invalid input image";
        return false;
    }
    if (img.channels != 3) {
        if (error) *error = "expected 3-channel BGR image";
        return false;
    }
    if (d.empty()) return true;

    DETECTIONS out;
    out.reserve(d.size());

    std::string first_err;
    for (const DETECTION_ROW& det : d) {
        const float x0 = det.tlwh(0);
        const float y0 = det.tlwh(1);
        const float w0 = det.tlwh(2);
        const float h0 = det.tlwh(3);
        if (w0 <= 1.0F || h0 <= 1.0F) continue;

        // Match legacy DeepSORT implementation:
        //   rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        //   rc.width = rc.height * 0.5;
        const float new_w = h0 * 0.5F;
        const float new_x = x0 + (w0 - new_w) * 0.5F;
        const float new_y = y0;

        int rx = static_cast<int>(new_x);
        int ry = static_cast<int>(new_y);
        int rw = static_cast<int>(new_w);
        int rh = static_cast<int>(h0);

        rx = ClampInt(rx, 0, img.width - 1);
        ry = ClampInt(ry, 0, img.height - 1);
        rw = std::max(1, std::min(rw, img.width - rx));
        rh = std::max(1, std::min(rh, img.height - ry));

        unsigned char* p = img.data + static_cast<std::size_t>(ry) * static_cast<std::size_t>(img.step) +
                           static_cast<std::size_t>(rx) * 3U;
        SimpleCV::Mat patch(rh, rw, 3, p, img.step, false);

        std::vector<float> feat;
        std::string ext_err;
        if (!extractor_.Extract(patch, &feat, &ext_err) || feat.size() != static_cast<std::size_t>(k_feature_dim)) {
            if (first_err.empty()) first_err = ext_err.empty() ? "feature extract failed" : std::move(ext_err);
            continue;
        }

        DETECTION_ROW row = det;
        for (int i = 0; i < k_feature_dim; ++i) {
            row.feature[i] = feat[static_cast<std::size_t>(i)];
        }
        out.push_back(std::move(row));
    }

    d.swap(out);
    if (error && !first_err.empty()) {
        *error = first_err;
    }
    return true;
}
