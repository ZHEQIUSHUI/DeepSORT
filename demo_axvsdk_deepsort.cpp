#include "detector/yolo_detector.hpp"

#include "tracker/deepsort/include/model.h"
#include "tracker/deepsort/include/track.h"
#include "tracker/deepsort/include/tracker.h"
#include "tracker/deepsort/include/reid_feature_extractor.hpp"

#include "common/ax_image_processor.h"
#include "common/ax_system.h"
#include "pipeline/ax_pipeline.h"

#include "cmdline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(DEEPSORT_HAVE_AXCL)
#include "npu/runner/axcl/axcl_runtime.hpp"
#endif

namespace {

int ClampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

std::uint32_t ColorForIdRgb(int id) {
    const std::uint32_t x = static_cast<std::uint32_t>(id) * 2654435761u;
    const std::uint8_t r = static_cast<std::uint8_t>((x >> 16) & 0xFF);
    const std::uint8_t g = static_cast<std::uint8_t>((x >> 8) & 0xFF);
    const std::uint8_t b = static_cast<std::uint8_t>((x >> 0) & 0xFF);
    return (static_cast<std::uint32_t>(r) << 16) | (static_cast<std::uint32_t>(g) << 8) | b;
}

std::uint32_t BackgroundBgrToRgb(const std::vector<unsigned char>& bgr) {
    const std::uint8_t b = bgr.size() >= 1 ? bgr[0] : 0;
    const std::uint8_t g = bgr.size() >= 2 ? bgr[1] : 0;
    const std::uint8_t r = bgr.size() >= 3 ? bgr[2] : 0;
    return (static_cast<std::uint32_t>(r) << 16) | (static_cast<std::uint32_t>(g) << 8) | b;
}

bool MakeReidCropRect(const axvsdk::common::AxImage& frame,
                      float x0,
                      float y0,
                      float w0,
                      float h0,
                      axvsdk::common::CropRect* out) {
    if (!out) return false;
    *out = {};

    if (w0 <= 1.0F || h0 <= 1.0F) return false;

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

    rx = ClampInt(rx, 0, static_cast<int>(frame.width()) - 1);
    ry = ClampInt(ry, 0, static_cast<int>(frame.height()) - 1);
    rw = std::max(1, std::min(rw, static_cast<int>(frame.width()) - rx));
    rh = std::max(1, std::min(rh, static_cast<int>(frame.height()) - ry));

    // NV12 crop must be even aligned.
    if (frame.format() == axvsdk::common::PixelFormat::kNv12) {
        rx &= ~1;
        ry &= ~1;
        rw &= ~1;
        rh &= ~1;
        if (rw < 2 || rh < 2) return false;
        if (rx + rw > static_cast<int>(frame.width())) {
            rw = (static_cast<int>(frame.width()) - rx) & ~1;
        }
        if (ry + rh > static_cast<int>(frame.height())) {
            rh = (static_cast<int>(frame.height()) - ry) & ~1;
        }
        if (rw < 2 || rh < 2) return false;
    }

    out->x = rx;
    out->y = ry;
    out->width = static_cast<std::uint32_t>(rw);
    out->height = static_cast<std::uint32_t>(rh);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    cmdline::parser parser;
    parser.add<std::string>("input", 'i', "input video path/uri", true,
                            "/home/axera/ax_video_sdk/data/pedestrian_thailand_1920x1080_30fps_5Mbps_hevc.mp4");
    parser.add<std::string>("output", 'o', "output mp4 path", false, "tracked_axvsdk.mp4");
    parser.add<std::string>("yolo", 'y', "yolo axmodel path", false, "/home/axera/ax-pipeline/models/ax650/yolov5s.axmodel");
    parser.add<std::string>("reid", 'r', "reid axmodel path", false,
                            "/home/axera/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt_bs1.axmodel");
    parser.add<std::string>("yolo_type", 't', "yolo type: yolov5|yolov8", false, "yolov5");
    parser.add<std::string>("reid_color", 'c', "reid input color: rgb|bgr", false, "rgb");
    parser.add<int>("device_id", 'd', "axcl device index", false, 0);
    parser.add<int>("max_frames", 'n', "max frames to process (0=all)", false, 0);
    parser.parse_check(argc, argv);

    const std::string input_path = parser.get<std::string>("input");
    const std::string output_path = parser.get<std::string>("output");
    const std::string yolo_model = parser.get<std::string>("yolo");
    const std::string reid_model = parser.get<std::string>("reid");
    const std::string yolo_type = parser.get<std::string>("yolo_type");
    const std::string reid_color = parser.get<std::string>("reid_color");
    const int device_id = parser.get<int>("device_id");
    const int max_frames = std::max(0, parser.get<int>("max_frames"));

    axvsdk::common::SystemOptions system_options{};
    system_options.device_id = device_id;
    system_options.enable_vdec = true;
    system_options.enable_venc = true;
    system_options.enable_ivps = true;
    if (!axvsdk::common::InitializeSystem(system_options)) {
        std::cerr << "[demo_axvsdk_deepsort] InitializeSystem failed\n";
        return 3;
    }

    struct SystemGuard {
        ~SystemGuard() {
            axvsdk::common::ShutdownSystem();
        }
    } system_guard;

#if defined(DEEPSORT_HAVE_AXCL)
    deepsort::npu::AxclEngineGuard axcl_engine;
    std::string axcl_err;
    if (!axcl_engine.Init(&axcl_err)) {
        std::cerr << "[demo_axvsdk_deepsort] AXCL engine init failed: " << axcl_err << "\n";
        return 3;
    }
#endif

    auto processor = axvsdk::common::CreateImageProcessor();
    if (!processor) {
        std::cerr << "[demo_axvsdk_deepsort] CreateImageProcessor failed\n";
        return 3;
    }

    deepsort::detector::YoloDetectorOptions det_opt{};
    det_opt.model_path = yolo_model;
    det_opt.device_id = device_id;
    det_opt.post.model_type = (yolo_type == "yolov8") ? deepsort::detector::YoloModelType::kYolov8
                                                      : deepsort::detector::YoloModelType::kYolov5;

    deepsort::detector::YoloDetector detector;
    std::string err;
    if (!detector.Init(det_opt, &err)) {
        std::cerr << "[demo_axvsdk_deepsort] detector init failed: " << err << "\n";
        return 4;
    }

    deepsort::npu::ReidFeatureExtractorOptions reid_opt{};
    reid_opt.model_path = reid_model;
    reid_opt.device_id = device_id;
    reid_opt.color_order = (reid_color == "bgr") ? deepsort::npu::ColorOrder::kBgr : deepsort::npu::ColorOrder::kRgb;
    deepsort::npu::ReidFeatureExtractor reid;
    if (!reid.Init(reid_opt, &err)) {
        std::cerr << "[demo_axvsdk_deepsort] reid init failed: " << err << "\n";
        return 4;
    }
    if (reid.feature_dim() != k_feature_dim) {
        std::cerr << "[demo_axvsdk_deepsort] unexpected reid feature dim: " << reid.feature_dim()
                  << " expected=" << k_feature_dim << "\n";
        return 4;
    }

    const bool use_device_input =
#if defined(DEEPSORT_HAVE_AXCL)
        true;
#else
        false;
#endif

    const auto det_fmt =
        use_device_input ? ((detector.input_format() == deepsort::detector::YoloInputFormat::kNv12)
                                ? axvsdk::common::PixelFormat::kNv12
                                : axvsdk::common::PixelFormat::kBgr24)
                         : axvsdk::common::PixelFormat::kBgr24;
    const auto reid_fmt =
        use_device_input ? ((reid_opt.color_order == deepsort::npu::ColorOrder::kRgb) ? axvsdk::common::PixelFormat::kRgb24
                                                                                      : axvsdk::common::PixelFormat::kBgr24)
                         : axvsdk::common::PixelFormat::kBgr24;

    axvsdk::common::AxImage::Ptr det_input;
    if (use_device_input) {
        det_input = axvsdk::common::AxImage::Create(det_fmt,
                                                    static_cast<std::uint32_t>(detector.input_width()),
                                                    static_cast<std::uint32_t>(detector.input_height()));
        if (!det_input) {
            std::cerr << "[demo_axvsdk_deepsort] create det input image failed\n";
            return 5;
        }
    }

    auto reid_input =
        axvsdk::common::AxImage::Create(reid_fmt, static_cast<std::uint32_t>(reid.input_width()),
                                        static_cast<std::uint32_t>(reid.input_height()));
    if (!reid_input) {
        std::cerr << "[demo_axvsdk_deepsort] create reid input image failed\n";
        return 5;
    }

    auto pipeline = axvsdk::pipeline::CreatePipeline();
    if (!pipeline) {
        std::cerr << "[demo_axvsdk_deepsort] CreatePipeline failed\n";
        return 6;
    }

    axvsdk::pipeline::PipelineConfig cfg{};
    cfg.device_id = device_id;
    cfg.input.uri = input_path;
    cfg.input.realtime_playback = true;
    cfg.input.loop_playback = false;

    axvsdk::pipeline::PipelineOutputConfig out{};
    out.codec = axvsdk::codec::VideoCodecType::kH265;
    out.uris.push_back(output_path);
    cfg.outputs.push_back(out);

    // Frame output: prefer zero-copy NV12 decoded frames.
    cfg.frame_output.output_image.format = axvsdk::common::PixelFormat::kNv12;

    if (!pipeline->Open(cfg)) {
        std::cerr << "[demo_axvsdk_deepsort] pipeline Open failed\n";
        return 6;
    }
    if (!pipeline->Start()) {
        std::cerr << "[demo_axvsdk_deepsort] pipeline Start failed\n";
        pipeline->Close();
        return 6;
    }

    struct PipelineGuard {
        axvsdk::pipeline::Pipeline* p{nullptr};
        ~PipelineGuard() {
            if (p) {
                p->Stop();
                p->Close();
            }
        }
    } pipeline_guard{pipeline.get()};

    std::mutex frame_mu;
    std::condition_variable frame_cv;
    axvsdk::common::AxImage::Ptr latest_frame;
    std::uint64_t latest_seq = 0;

    pipeline->SetFrameCallback([&](axvsdk::common::AxImage::Ptr frame) {
        if (!frame) return;
        std::lock_guard<std::mutex> lock(frame_mu);
        latest_frame = std::move(frame);
        ++latest_seq;
        frame_cv.notify_one();
    });

    // DeepSORT params (same as original main.cpp)
    const int nn_budget = 100;
    const float max_cosine_distance = 0.2F;
    tracker trk(max_cosine_distance, nn_budget);

    const std::uint32_t bg_rgb = BackgroundBgrToRgb(det_opt.background);

    std::uint64_t seen_seq = 0;
    std::uint64_t processed_frames = 0;
    auto last_progress = std::chrono::steady_clock::now();
    axvsdk::pipeline::PipelineStats last_stats{};
    std::uint64_t last_encoded_packets = 0;
    axvsdk::common::AxImage::Ptr frame_bgr;

    while (true) {
        axvsdk::common::AxImage::Ptr frame;
        {
            std::unique_lock<std::mutex> lock(frame_mu);
            frame_cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return latest_seq != seen_seq; });
            if (latest_seq != seen_seq) {
                seen_seq = latest_seq;
                frame = latest_frame;
            }
        }

        // Periodically check for pipeline idle (end of stream).
        const auto stats = pipeline->GetStats();
        const std::uint64_t encoded_packets =
            (!stats.output_stats.empty() ? stats.output_stats.front().encoded_packets : 0ULL);
        if (stats.decoded_frames != last_stats.decoded_frames || encoded_packets != last_encoded_packets) {
            last_progress = std::chrono::steady_clock::now();
            last_stats = stats;
            last_encoded_packets = encoded_packets;
        }

        const auto idle = std::chrono::steady_clock::now() - last_progress;
        if (idle > std::chrono::seconds(2) && processed_frames > 0) {
            break;
        }

        if (!frame) {
            continue;
        }

        std::vector<deepsort::detector::Detection> dets;
        if (use_device_input) {
            // 1) Preprocess for YOLO using hardware IVPS (letterbox + CSC if needed).
            axvsdk::common::ImageProcessRequest det_req{};
            det_req.output_image.format = det_fmt;
            det_req.output_image.width = static_cast<std::uint32_t>(detector.input_width());
            det_req.output_image.height = static_cast<std::uint32_t>(detector.input_height());
            det_req.resize.mode = axvsdk::common::ResizeMode::kKeepAspectRatio;
            det_req.resize.horizontal_align = axvsdk::common::ResizeAlign::kCenter;
            det_req.resize.vertical_align = axvsdk::common::ResizeAlign::kCenter;
            det_req.resize.background_color = bg_rgb;
            if (!processor->Process(*frame, det_req, *det_input)) {
                std::cerr << "[demo_axvsdk_deepsort] det preprocess failed\n";
                return 7;
            }

            // 2) YOLO inference (device input, AXCL only).
            if (!detector.DetectFromDevice(det_input->physical_address(0),
                                           det_input->byte_size(),
                                           static_cast<int>(frame->width()),
                                           static_cast<int>(frame->height()),
                                           &dets,
                                           &err)) {
                std::cerr << "[demo_axvsdk_deepsort] detect failed: " << err << "\n";
                return 7;
            }
        } else {
            // AX650/MSP path: convert frame to BGR and run the standard detector path.
            if (!frame_bgr || frame_bgr->width() != frame->width() || frame_bgr->height() != frame->height()) {
                frame_bgr = axvsdk::common::AxImage::Create(axvsdk::common::PixelFormat::kBgr24, frame->width(),
                                                           frame->height());
                if (!frame_bgr) {
                    std::cerr << "[demo_axvsdk_deepsort] create frame_bgr failed\n";
                    return 7;
                }
            }

            axvsdk::common::ImageProcessRequest det_req{};
            det_req.output_image.format = axvsdk::common::PixelFormat::kBgr24;
            det_req.output_image.width = frame->width();
            det_req.output_image.height = frame->height();
            det_req.resize.mode = axvsdk::common::ResizeMode::kStretch;
            if (!processor->Process(*frame, det_req, *frame_bgr)) {
                std::cerr << "[demo_axvsdk_deepsort] det CSC failed\n";
                return 7;
            }

            SimpleCV::Mat frame_mat(static_cast<int>(frame_bgr->height()), static_cast<int>(frame_bgr->width()), 3,
                                    static_cast<unsigned char*>(frame_bgr->virtual_address(0)),
                                    frame_bgr->stride(0));
            if (!detector.Detect(frame_mat, &dets, &err)) {
                std::cerr << "[demo_axvsdk_deepsort] detect failed: " << err << "\n";
                return 7;
            }
        }

        // 3) Prepare DeepSORT detections with ReID features.
        DETECTIONS detections;
        detections.reserve(dets.size());

        for (const auto& d : dets) {
            if (d.class_id != 0) continue;  // person
            const float w0 = std::max(0.0F, d.x1 - d.x0);
            const float h0 = std::max(0.0F, d.y1 - d.y0);
            if (w0 <= 1.0F || h0 <= 1.0F) continue;

            axvsdk::common::CropRect crop{};
            if (!MakeReidCropRect(*frame, d.x0, d.y0, w0, h0, &crop)) {
                continue;
            }

            axvsdk::common::ImageProcessRequest reid_req{};
            reid_req.enable_crop = true;
            reid_req.crop = crop;
            reid_req.output_image.format = reid_fmt;
            reid_req.output_image.width = static_cast<std::uint32_t>(reid.input_width());
            reid_req.output_image.height = static_cast<std::uint32_t>(reid.input_height());
            reid_req.resize.mode = axvsdk::common::ResizeMode::kStretch;

            if (!processor->Process(*frame, reid_req, *reid_input)) {
                continue;
            }

            std::vector<float> feat;
            if (use_device_input) {
                if (!reid.ExtractFromDevice(reid_input->physical_address(0), reid_input->byte_size(), &feat, &err) ||
                    feat.size() != static_cast<std::size_t>(k_feature_dim)) {
                    continue;
                }
            } else {
                SimpleCV::Mat reid_mat(static_cast<int>(reid_input->height()), static_cast<int>(reid_input->width()), 3,
                                       static_cast<unsigned char*>(reid_input->virtual_address(0)),
                                       reid_input->stride(0));
                if (!reid.Extract(reid_mat, &feat, &err) || feat.size() != static_cast<std::size_t>(k_feature_dim)) {
                    continue;
                }
            }

            DETECTION_ROW row;
            row.tlwh = DETECTBOX(d.x0, d.y0, w0, h0);
            row.confidence = d.score;
            for (int i = 0; i < k_feature_dim; ++i) {
                row.feature[i] = feat[static_cast<std::size_t>(i)];
            }
            detections.push_back(std::move(row));
        }

        // 4) DeepSORT update.
        trk.predict();
        trk.update(detections);

        // 5) Draw OSD (same ID keeps same color).
        axvsdk::common::DrawFrame osd{};
        osd.hold_frames = 3;
        for (Track& t : trk.tracks) {
            if (!t.is_confirmed() || t.time_since_update > 1) continue;
            const DETECTBOX tlwh = t.to_tlwh();
            int x = ClampInt(static_cast<int>(tlwh(0)), 0, static_cast<int>(frame->width()) - 1);
            int y = ClampInt(static_cast<int>(tlwh(1)), 0, static_cast<int>(frame->height()) - 1);
            int w = std::max(1, std::min(static_cast<int>(tlwh(2)), static_cast<int>(frame->width()) - x));
            int h = std::max(1, std::min(static_cast<int>(tlwh(3)), static_cast<int>(frame->height()) - y));

            // AXCL IVPS OSD on NV12 is safest with even-aligned geometry.
            const int frame_w = static_cast<int>(frame->width());
            const int frame_h = static_cast<int>(frame->height());
            if (frame->format() == axvsdk::common::PixelFormat::kNv12) {
                x &= ~1;
                y &= ~1;
                w &= ~1;
                h &= ~1;
                if (x + w > frame_w) {
                    w = (frame_w - x) & ~1;
                }
                if (y + h > frame_h) {
                    h = (frame_h - y) & ~1;
                }
            }

            // AXCL_IVPS_DrawRect seems to reject rectangles that touch the bottom/right edge.
            // Keep a 1px margin to be safe.
            const int w_max = frame_w - x - 1;
            const int h_max = frame_h - y - 1;
            if (w_max <= 0 || h_max <= 0) continue;
            w = std::min(w, w_max);
            h = std::min(h, h_max);
            if (frame->format() == axvsdk::common::PixelFormat::kNv12) {
                w = std::min(w & ~1, w_max & ~1);
                h = std::min(h & ~1, h_max & ~1);
            }
            if (w < 2 || h < 2) continue;
            std::uint16_t thickness = 2;
            osd.rects.push_back(axvsdk::common::DrawRect{
                x,
                y,
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(h),
                thickness,
                255,
                ColorForIdRgb(t.track_id),
                false,
                false,
                0,
                0,
            });
        }
        (void)pipeline->SetOsd(osd);

        ++processed_frames;
        if (max_frames > 0 && static_cast<int>(processed_frames) >= max_frames) {
            break;
        }

        if ((processed_frames % 50) == 0) {
            std::cerr << "[demo_axvsdk_deepsort] processed=" << processed_frames
                      << " decoded=" << stats.decoded_frames
                      << " encoded=" << encoded_packets << "\n";
        }
    }

    std::cerr << "[demo_axvsdk_deepsort] stopping pipeline...\n";
    pipeline->Stop();
    pipeline->Close();
    std::cerr << "[demo_axvsdk_deepsort] done output=" << output_path << "\n";
    return 0;
}
