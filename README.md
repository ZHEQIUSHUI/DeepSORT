# DeepSORT (AXCL / NPU)

这是一个面向 AXERA NPU（AXCL）使用场景的 **DeepSORT 跟踪分支**：

- **Detector**：YOLOv5 / YOLOv8（`.axmodel`），AXCL 推理 + CPU 后处理
- **ReID**：DeepSORT 特征提取（`.axmodel`），AXCL 推理（输出 512 维特征）
- **Video Demo**：集成 `ax-video-sdk`，完成 **解码 → IVPS 预处理/抠图 → 检测+ReID+DeepSORT → OSD → 编码** 的端到端示例
- **不依赖 OpenCV / onnxruntime**：图片读写与基础绘制使用 `SimpleCV` 子模块

## 模型下载（从 Release 获取）

模型不随仓库提交。请从本仓库的 **GitHub Releases** 下载并自行放置到本地路径，然后通过命令行传入：

- YOLO：`yolov5s.axmodel`（或你自己的 yolov8 `.axmodel`）
- ReID：`ckpt_bs1.axmodel`
  - 输入：`1 x 128 x 64 x 3`，`uint8`（HWC）
  - 输出：`1 x 512`，`float32`

## 编译

依赖：

- CMake >= 3.16，C++17
- `Eigen3`
- AXCL SDK（默认按 `/usr/include/axcl` + `/usr/lib/axcl` 查找）
- 子模块：`SimpleCV`、`ax-video-sdk`
  - 仓库内提供 `build_axcl_x86.sh / build_axcl_aarch64.sh / build_ax650.sh`（参考 `ax-video-sdk`）用于 CI/本地一键编译与打包

拉取子模块并编译：

```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake ..
make -j
```

> 说明：默认 `DEEPSORT_AXCL_INIT_IN_RUNNER=OFF`，AXCL 运行时/Context/Engine 初始化在应用侧（示例见 `test_detector.cpp`/`test_reid.cpp`/`demo_axvsdk_deepsort.cpp`）。
> 若你希望 Runner 内部自初始化（不依赖外部初始化流程），可配置 `-DDEEPSORT_AXCL_INIT_IN_RUNNER=ON`。

## 运行

### 1) 视频跟踪 demo（推荐）

```bash
./demo_axvsdk_deepsort \
  --input  /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --yolo   /path/to/yolo.axmodel \
  --reid   /path/to/ckpt_bs1.axmodel \
  --yolo_type yolov5 \
  --reid_color rgb \
  --device_id 0
```

参数说明（节选）：

- `--yolo_type`：`yolov5|yolov8`
- `--reid_color`：`rgb|bgr`
- `--max_frames`：限制处理帧数（`0` 表示全量）

### 2) 单张图检测

```bash
./test_detector --model /path/to/yolo.axmodel --image /path/to/image.jpg --type yolov5 --device_id 0
```

### 3) 单张图 ReID 特征提取

```bash
./test_reid --model /path/to/ckpt_bs1.axmodel --image /path/to/image.jpg --color rgb --device_id 0
```

### 4) 图片序列跟踪（离线）

```bash
./run_deepsort 'frames_in/*.jpg' frames_out /path/to/yolo.axmodel /path/to/ckpt_bs1.axmodel yolov5 rgb 300
```

## 备注

- 若你的环境需要显式 AXCL 配置文件，可设置：`AXCL_JSON=/usr/bin/axcl/axcl.json`（不同系统路径可能不同）。
- OSD 使用 **同一 track id 对应固定颜色**，便于观察跟踪一致性。
