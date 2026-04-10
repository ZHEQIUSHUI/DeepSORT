#pragma once

#if defined(DEEPSORT_HAVE_AXCL)

#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_device.h>

#include <cstdlib>
#include <fstream>
#include <string>

namespace deepsort::npu {

inline const char* ResolveAxclConfigPath() {
    const char* config_path = std::getenv("AXCL_JSON");
    if (config_path == nullptr || config_path[0] == '\0') {
        config_path = std::getenv("AXCL_CONFIG");
    }
    if (config_path == nullptr || config_path[0] == '\0') {
        static constexpr const char* kDefaultAxclJson = "/usr/bin/axcl/axcl.json";
        if (std::ifstream(kDefaultAxclJson).good()) {
            config_path = kDefaultAxclJson;
        } else {
            config_path = nullptr;
        }
    }
    return config_path;
}

class AxclStandaloneRuntime {
public:
    AxclStandaloneRuntime() = default;
    ~AxclStandaloneRuntime() { Shutdown(); }

    AxclStandaloneRuntime(const AxclStandaloneRuntime&) = delete;
    AxclStandaloneRuntime& operator=(const AxclStandaloneRuntime&) = delete;

    bool Init(int device_index, std::string* error) {
#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
        (void)device_index;
        (void)error;
        return true;
#else
        Shutdown();

        const char* config_path = ResolveAxclConfigPath();
        axclError ret = axclInit(config_path);
        if (ret != AXCL_SUCC && config_path != nullptr) {
            ret = axclInit(nullptr);
        }
        if (ret != AXCL_SUCC) {
            if (error) *error = "axclInit failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }
        axcl_inited_ = true;

        axclrtDeviceList lst{};
        ret = axclrtGetDeviceList(&lst);
        if (ret != AXCL_SUCC || lst.num == 0) {
            if (error) *error = "axclrtGetDeviceList failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }
        if (device_index < 0) device_index = 0;
        if (device_index >= static_cast<int>(lst.num)) {
            if (error) *error = "invalid device index: " + std::to_string(device_index);
            Shutdown();
            return false;
        }

        runtime_device_id_ = lst.devices[device_index];
        ret = axclrtSetDevice(runtime_device_id_);
        if (ret != AXCL_SUCC) {
            if (error) *error = "axclrtSetDevice failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }

        ret = axclrtCreateContext(&ctx_, runtime_device_id_);
        if (ret != AXCL_SUCC || ctx_ == nullptr) {
            if (error) *error = "axclrtCreateContext failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }
        ret = axclrtSetCurrentContext(ctx_);
        if (ret != AXCL_SUCC) {
            if (error) *error = "axclrtSetCurrentContext failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }

        ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (ret != AXCL_SUCC) {
            if (error) *error = "axclrtEngineInit failed ret=0x" + std::to_string(ret);
            Shutdown();
            return false;
        }
        engine_inited_ = true;
        return true;
#endif
    }

    void Shutdown() {
#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
        return;
#else
        if (runtime_device_id_ >= 0) {
            (void)axclrtSetDevice(runtime_device_id_);
        }
        if (ctx_ != nullptr) {
            (void)axclrtSetCurrentContext(ctx_);
        }
        if (engine_inited_) {
            (void)axclrtEngineFinalize();
            engine_inited_ = false;
        }
        if (ctx_ != nullptr) {
            (void)axclrtDestroyContext(ctx_);
            ctx_ = nullptr;
        }
        if (axcl_inited_) {
            (void)axclFinalize();
            axcl_inited_ = false;
        }
        runtime_device_id_ = -1;
#endif
    }

    int runtime_device_id() const noexcept { return runtime_device_id_; }
    axclrtContext context() const noexcept { return ctx_; }

private:
    bool axcl_inited_{false};
    bool engine_inited_{false};
    int runtime_device_id_{-1};
    axclrtContext ctx_{nullptr};
};

class AxclEngineGuard {
public:
    AxclEngineGuard() = default;
    ~AxclEngineGuard() { Shutdown(); }

    AxclEngineGuard(const AxclEngineGuard&) = delete;
    AxclEngineGuard& operator=(const AxclEngineGuard&) = delete;

    bool Init(std::string* error) {
#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
        (void)error;
        return true;
#else
        Shutdown();

        const axclError ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (ret != AXCL_SUCC) {
            if (error) *error = "axclrtEngineInit failed ret=0x" + std::to_string(ret);
            return false;
        }
        engine_inited_ = true;
        return true;
#endif
    }

    void Shutdown() {
#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
        return;
#else
        if (!engine_inited_) return;

        int32_t device_id = -1;
        if (axclrtGetDevice(&device_id) == AXCL_SUCC && device_id >= 0) {
            runtime_device_id_ = device_id;
            (void)axclrtSetDevice(runtime_device_id_);
        }

        axclrtContext ctx = nullptr;
        const axclError ctx_ret = axclrtGetCurrentContext(&ctx);
        if (ctx_ret == AXCL_SUCC && ctx != nullptr) {
            (void)axclrtSetCurrentContext(ctx);
            (void)axclrtEngineFinalize();
            engine_inited_ = false;
            return;
        }

        // Fallback: create a temporary context to satisfy finalize.
        if (runtime_device_id_ >= 0) {
            if (axclrtCreateContext(&tmp_ctx_, runtime_device_id_) == AXCL_SUCC && tmp_ctx_ != nullptr) {
                if (axclrtSetCurrentContext(tmp_ctx_) == AXCL_SUCC) {
                    (void)axclrtEngineFinalize();
                }
                (void)axclrtDestroyContext(tmp_ctx_);
                tmp_ctx_ = nullptr;
            }
        }

        engine_inited_ = false;
#endif
    }

private:
#if !defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
    bool engine_inited_{false};
    int runtime_device_id_{-1};
    axclrtContext tmp_ctx_{nullptr};
#endif
};

}  // namespace deepsort::npu

#endif  // DEEPSORT_HAVE_AXCL
