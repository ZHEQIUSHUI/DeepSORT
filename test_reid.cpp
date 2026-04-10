#include "reid_feature_extractor.hpp"

#include "SimpleCV.hpp"

#include "cmdline.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(DEEPSORT_HAVE_AXCL)
#include "npu/runner/axcl/axcl_runtime.hpp"
#endif

#if defined(DEEPSORT_HAVE_AX650)
#include <ax_sys_api.h>
#endif

namespace {

bool FileExists(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    return ifs.good();
}

}  // namespace

int main(int argc, char** argv) {
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "reid axmodel path", true);
    parser.add<std::string>("image", 'i', "input image path", true);
    parser.add<std::string>("color", 'c', "input color order: rgb|bgr", false, "rgb");
    parser.add<int>("device_id", 'd', "axcl device index", false, 0);
    parser.parse_check(argc, argv);

    const std::string model_path = parser.get<std::string>("model");
    const std::string image_path = parser.get<std::string>("image");
    const std::string color = parser.get<std::string>("color");
    const int device_id = parser.get<int>("device_id");

    std::string err;

    if (!FileExists(model_path)) {
        std::cout << "[test_reid] SKIP missing model: " << model_path << "\n";
        return 0;
    }
    if (!FileExists(image_path)) {
        std::cout << "[test_reid] SKIP missing image: " << image_path << "\n";
        return 0;
    }

#if defined(DEEPSORT_HAVE_AX650)
    const int sys_ret = AX_SYS_Init();
    if (sys_ret != 0) {
        std::cerr << "[test_reid] AX_SYS_Init failed ret=" << sys_ret << "\n";
        return 1;
    }
    struct SysGuard {
        ~SysGuard() { (void)AX_SYS_Deinit(); }
    } sys_guard;
#endif

#if defined(DEEPSORT_HAVE_AXCL)
    deepsort::npu::AxclStandaloneRuntime axcl_runtime;
    if (!axcl_runtime.Init(device_id, &err)) {
        std::cerr << "[test_reid] AXCL init failed: " << err << "\n";
        return 1;
    }
#endif

    SimpleCV::Mat img = SimpleCV::imread(image_path, SimpleCV::ColorSpace::BGR);
    if (img.empty()) {
        std::cerr << "[test_reid] imread failed: " << image_path << "\n";
        return 1;
    }

    deepsort::npu::ReidFeatureExtractorOptions opt{};
    opt.model_path = model_path;
    opt.device_id = device_id;
    opt.color_order = (color == "bgr") ? deepsort::npu::ColorOrder::kBgr : deepsort::npu::ColorOrder::kRgb;

    deepsort::npu::ReidFeatureExtractor ext;
    if (!ext.Init(opt, &err)) {
        std::cerr << "[test_reid] init failed: " << err << "\n";
        return 1;
    }

    std::vector<float> feat;
    if (!ext.Extract(img, &feat, &err)) {
        std::cerr << "[test_reid] extract failed: " << err << "\n";
        return 1;
    }

    std::cout << "[test_reid] input=" << ext.input_width() << "x" << ext.input_height()
              << " feat_dim=" << feat.size() << "\n";
    if (!feat.empty()) {
        std::cout << "[test_reid] feat[0..7]=";
        const std::size_t n = std::min<std::size_t>(8, feat.size());
        for (std::size_t i = 0; i < n; ++i) {
            std::cout << feat[i] << (i + 1 == n ? "" : ",");
        }
        std::cout << "\n";
    }
    return 0;
}
