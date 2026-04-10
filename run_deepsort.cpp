#include "detector/yolo_detector.hpp"

#include "tracker/deepsort/include/dataType.h"
#include "tracker/deepsort/include/FeatureTensor.h"
#include "tracker/deepsort/include/model.h"
#include "tracker/deepsort/include/tracker.h"

#include "SimpleCV.hpp"

#include "cmdline.hpp"

#if defined(DEEPSORT_HAVE_AXCL)
#include "npu/runner/axcl/axcl_runtime.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#if defined(DEEPSORT_HAVE_AX650)
#include <ax_sys_api.h>
#endif

namespace fs = std::filesystem;

namespace {

int ClampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

SimpleCV::Scalar ColorForId(int id) {
    // Deterministic palette in BGR order.
    const unsigned int x = static_cast<unsigned int>(id * 2654435761u);
    const unsigned char b = static_cast<unsigned char>((x >> 0) & 0xFF);
    const unsigned char g = static_cast<unsigned char>((x >> 8) & 0xFF);
    const unsigned char r = static_cast<unsigned char>((x >> 16) & 0xFF);
    return SimpleCV::Scalar(b, g, r);
}

void PrintUsage(const char* argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " <in_glob> <out_dir> <yolo.axmodel> <reid.axmodel> [yolov5|yolov8] [rgb|bgr] [max_frames] [device_id]\n"
              << "\n"
              << "Examples:\n"
              << "  " << argv0 << " 'frames_in/*.jpg' frames_out yolov5s.axmodel ckpt_bs1.axmodel yolov5 rgb 300 0\n";
}

}  // namespace

int main(int argc, char** argv) {
    cmdline::parser parser;
    parser.add<std::string>("input", 'i', "input frame glob", false, "");
    parser.add<std::string>("output_dir", 'o', "output directory", false, "frames_out");
    parser.add<std::string>("yolo", 'y', "yolo axmodel path", false, "");
    parser.add<std::string>("reid", 'r', "reid axmodel path", false, "");
    parser.add<std::string>("yolo_type", 't', "yolo type: yolov5|yolov8", false, "yolov5");
    parser.add<std::string>("reid_color", 'c', "reid input color: rgb|bgr", false, "rgb");
    parser.add<int>("device_id", 'd', "axcl device index", false, 0);
    parser.add<int>("max_frames", 'n', "max frames to process (0=all)", false, 0);
    parser.parse_check(argc, argv);

    std::string in_glob = parser.get<std::string>("input");
    std::string out_dir = parser.get<std::string>("output_dir");
    std::string yolo_model = parser.get<std::string>("yolo");
    std::string reid_model = parser.get<std::string>("reid");
    std::string yolo_type = parser.get<std::string>("yolo_type");
    std::string reid_color = parser.get<std::string>("reid_color");
    int device_id = parser.get<int>("device_id");
    int max_frames = std::max(0, parser.get<int>("max_frames"));

    // Backward compatible positional args:
    //   run_deepsort <in_glob> <out_dir> <yolo.axmodel> <reid.axmodel> [yolov5|yolov8] [rgb|bgr] [max_frames] [device_id]
    const auto& rest = parser.rest();
    if (!rest.empty()) {
        if (!parser.exist("input") && rest.size() >= 1) in_glob = rest[0];
        if (!parser.exist("output_dir") && rest.size() >= 2) out_dir = rest[1];
        if (!parser.exist("yolo") && rest.size() >= 3) yolo_model = rest[2];
        if (!parser.exist("reid") && rest.size() >= 4) reid_model = rest[3];
        if (!parser.exist("yolo_type") && rest.size() >= 5) yolo_type = rest[4];
        if (!parser.exist("reid_color") && rest.size() >= 6) reid_color = rest[5];
        if (!parser.exist("max_frames") && rest.size() >= 7) max_frames = std::max(0, std::atoi(rest[6].c_str()));
        if (!parser.exist("device_id") && rest.size() >= 8) device_id = std::atoi(rest[7].c_str());
        if (rest.size() > 8) {
            std::cerr << "[run_deepsort] too many positional args\n";
            std::cerr << parser.usage();
            return 2;
        }
    }

    if (in_glob.empty() || out_dir.empty() || yolo_model.empty() || reid_model.empty()) {
        PrintUsage(argv[0]);
        std::cerr << "\n" << parser.usage();
        return 2;
    }

    const std::vector<std::string> files = SimpleCV::glob(in_glob);
    if (files.empty()) {
        std::cerr << "[run_deepsort] no input frames matched: " << in_glob << "\n";
        return 1;
    }

#if defined(DEEPSORT_HAVE_AX650)
    const int sys_ret = AX_SYS_Init();
    if (sys_ret != 0) {
        std::cerr << "[run_deepsort] AX_SYS_Init failed ret=" << sys_ret << "\n";
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
        std::cerr << "[run_deepsort] AXCL init failed: " << err << "\n";
        return 1;
    }
#endif

    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "[run_deepsort] create out_dir failed: " << out_dir << " err=" << ec.message() << "\n";
        return 1;
    }

    deepsort::detector::YoloDetectorOptions det_opt{};
    det_opt.model_path = yolo_model;
    det_opt.device_id = device_id;
    det_opt.post.model_type = (yolo_type == "yolov8") ? deepsort::detector::YoloModelType::kYolov8
                                                      : deepsort::detector::YoloModelType::kYolov5;

    deepsort::detector::YoloDetector det;
    if (!det.Init(det_opt, &err)) {
        std::cerr << "[run_deepsort] detector init failed: " << err << "\n";
        return 1;
    }

    FeatureTensor* reid = FeatureTensor::getInstance();
    if (!reid->init(reid_model, reid_color, device_id, &err)) {
        std::cerr << "[run_deepsort] reid init failed: " << err << "\n";
        return 1;
    }

    // DeepSORT params (same as original main.cpp)
    const int nn_budget = 100;
    const float max_cosine_distance = 0.2F;
    tracker trk(max_cosine_distance, nn_budget);

    const int total = (max_frames > 0) ? std::min<int>(static_cast<int>(files.size()), max_frames)
                                       : static_cast<int>(files.size());

    for (int i = 0; i < total; ++i) {
        const std::string& path = files[static_cast<std::size_t>(i)];
        SimpleCV::Mat frame = SimpleCV::imread(path, SimpleCV::ColorSpace::BGR);
        if (frame.empty()) {
            std::cerr << "[run_deepsort] imread failed: " << path << "\n";
            return 1;
        }

        std::vector<deepsort::detector::Detection> dets;
        if (!det.Detect(frame, &dets, &err)) {
            std::cerr << "[run_deepsort] detect failed at " << path << ": " << err << "\n";
            return 1;
        }

        DETECTIONS detections;
        detections.reserve(dets.size());

        for (const auto& d : dets) {
            if (d.class_id != 0) continue;  // person

            const float x0 = d.x0;
            const float y0 = d.y0;
            const float w0 = std::max(0.0F, d.x1 - d.x0);
            const float h0 = std::max(0.0F, d.y1 - d.y0);
            if (w0 <= 1.0F || h0 <= 1.0F) continue;

            DETECTION_ROW row;
            row.tlwh = DETECTBOX(x0, y0, w0, h0);
            row.confidence = d.score;
            detections.push_back(row);
        }

        if (!reid->getRectsFeature(frame, detections, &err)) {
            std::cerr << "[run_deepsort] reid extract failed at " << path << ": " << err << "\n";
            return 1;
        }

        trk.predict();
        trk.update(detections);

        // Render tracking boxes (confirmed tracks only).
        for (Track& t : trk.tracks) {
            if (!t.is_confirmed() || t.time_since_update > 1) continue;
            const DETECTBOX tlwh = t.to_tlwh();
            const int x = ClampInt(static_cast<int>(tlwh(0)), 0, frame.width - 1);
            const int y = ClampInt(static_cast<int>(tlwh(1)), 0, frame.height - 1);
            const int w = std::max(1, std::min(static_cast<int>(tlwh(2)), frame.width - x));
            const int h = std::max(1, std::min(static_cast<int>(tlwh(3)), frame.height - y));

            const SimpleCV::Scalar color = ColorForId(t.track_id);
            SimpleCV::rectangle(frame, SimpleCV::Rect(x, y, w, h), color, 2);
            SimpleCV::putText(frame, std::to_string(t.track_id), SimpleCV::Point(x, std::max(0, y - 2)), 0.6, color, 2);
        }

        // Save as RGB for stb codecs.
        const SimpleCV::Mat rgb = SimpleCV::cvtColor(frame, SimpleCV::ColorSpace::RGB, SimpleCV::ColorSpace::BGR);
        const std::string out_path = (fs::path(out_dir) / fs::path(path).filename()).string();
        if (!SimpleCV::imwrite(out_path, rgb)) {
            std::cerr << "[run_deepsort] imwrite failed: " << out_path << "\n";
            return 1;
        }

        if ((i % 50) == 0) {
            std::cout << "[run_deepsort] " << (i + 1) << "/" << total << "\n";
        }
    }

    std::cout << "[run_deepsort] done frames=" << total << " out_dir=" << out_dir << "\n";
    return 0;
}
