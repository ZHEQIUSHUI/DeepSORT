#include "detector/yolo_detector.hpp"

#include "SimpleCV.hpp"

#include "cmdline.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

int main(int argc, char* argv[]) {
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "yolo axmodel path", false, "/home/axera/ax-pipeline/models/ax650/yolov5s.axmodel");
    parser.add<std::string>("image", 'i', "input image path (optional)", false, "");
    parser.add<std::string>("type", 't', "model type: yolov5|yolov8", false, "yolov5");
    parser.add<int>("device_id", 'd', "axcl device index", false, 0);
    parser.parse_check(argc, argv);

    std::string model_path = parser.get<std::string>("model");
    std::string image_path = parser.get<std::string>("image");
    std::string type = parser.get<std::string>("type");
    int device_id = parser.get<int>("device_id");

    // Backward compatible positional args:
    //   test_detector [model.axmodel] [image] [yolov5|yolov8] [device_id]
    const auto& rest = parser.rest();
    if (!rest.empty()) {
        if (!parser.exist("model") && rest.size() >= 1) model_path = rest[0];
        if (!parser.exist("image") && rest.size() >= 2) image_path = rest[1];
        if (!parser.exist("type") && rest.size() >= 3) type = rest[2];
        if (!parser.exist("device_id") && rest.size() >= 4) device_id = std::atoi(rest[3].c_str());
        if (rest.size() > 4) {
            std::cerr << "[test_detector] too many positional args\n";
            std::cerr << parser.usage();
            return 2;
        }
    }

    if (!FileExists(model_path)) {
        std::cout << "[test_detector] SKIP missing model: " << model_path << "\n";
        return 0;
    }

#if defined(DEEPSORT_HAVE_AX650)
    const int sys_ret = AX_SYS_Init();
    if (sys_ret != 0) {
        std::cerr << "[test_detector] AX_SYS_Init failed ret=" << sys_ret << "\n";
        return 1;
    }
    struct SysGuard {
        ~SysGuard() { (void)AX_SYS_Deinit(); }
    } sys_guard;
#endif

    std::string err;

#if defined(DEEPSORT_HAVE_AXCL)
    deepsort::npu::AxclStandaloneRuntime axcl_runtime;
    if (!axcl_runtime.Init(device_id, &err)) {
        std::cerr << "[test_detector] AXCL init failed: " << err << "\n";
        return 1;
    }
#endif

    deepsort::detector::YoloDetectorOptions opt{};
    opt.model_path = model_path;
    opt.device_id = device_id;
    opt.post.model_type = (type == "yolov8") ? deepsort::detector::YoloModelType::kYolov8
                                             : deepsort::detector::YoloModelType::kYolov5;

    deepsort::detector::YoloDetector det;
    if (!det.Init(opt, &err)) {
        std::cerr << "[test_detector] init failed: " << err << "\n";
        return 1;
    }

    SimpleCV::Mat img;
    if (!image_path.empty()) {
        img = SimpleCV::imread(image_path, SimpleCV::ColorSpace::BGR);
        if (img.empty()) {
            std::cerr << "[test_detector] failed to read image: " << image_path << "\n";
            return 1;
        }
    } else {
        img = SimpleCV::Mat(det.input_height(), det.input_width(), 3);
        std::memset(img.data, 0, static_cast<std::size_t>(img.height) * static_cast<std::size_t>(img.step));
    }

    std::vector<deepsort::detector::Detection> dets;
    const auto t0 = std::chrono::steady_clock::now();
    if (!det.Detect(img, &dets, &err)) {
        std::cerr << "[test_detector] detect failed: " << err << "\n";
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[test_detector] dets=" << dets.size() << " time_ms=" << ms << "\n";

    if (!image_path.empty()) {
        for (const auto& d : dets) {
            const int x = std::max(0, static_cast<int>(std::round(d.x0)));
            const int y = std::max(0, static_cast<int>(std::round(d.y0)));
            const int w = std::max(0, static_cast<int>(std::round(d.x1 - d.x0)));
            const int h = std::max(0, static_cast<int>(std::round(d.y1 - d.y0)));
            SimpleCV::rectangle(img, SimpleCV::Rect(x, y, w, h), SimpleCV::Scalar(0, 0, 255), 2);
        }
        const std::string out_path = "out.jpg";
        // SimpleCV codec writes as RGB(A); convert from our working BGR buffer.
        const SimpleCV::Mat rgb = SimpleCV::cvtColor(img, SimpleCV::ColorSpace::RGB, SimpleCV::ColorSpace::BGR);
        (void)SimpleCV::imwrite(out_path, rgb);
        std::cout << "[test_detector] wrote " << out_path << "\n";
    }

    return 0;
}
