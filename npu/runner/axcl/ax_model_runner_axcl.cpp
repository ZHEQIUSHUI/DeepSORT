#include "ax_model_runner_axcl.hpp"

#include "npu/runner/logging.hpp"

#include <axcl.h>

#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace {

#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
struct AxclGlobal {
    int refcnt = 0;
    bool engine_inited = false;
    bool axcl_inited = false;
};

AxclGlobal& Global() {
    static AxclGlobal g{};
    return g;
}

int AxclAcquire() {
    auto& g = Global();
    if (g.refcnt == 0) {
        const axclError ret = axclInit(nullptr);
        if (ret != AXCL_SUCC) {
            ALOGE("axclInit failed ret=0x%x", ret);
            return static_cast<int>(ret);
        }
        g.axcl_inited = true;
    }
    g.refcnt++;
    return 0;
}

int AxclEnsureEngineInited() {
    auto& g = Global();
    if (g.engine_inited) return 0;
    const axclError eng = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (eng != AXCL_SUCC) {
        ALOGE("axclrtEngineInit failed ret=0x%x", eng);
        return static_cast<int>(eng);
    }
    g.engine_inited = true;
    return 0;
}

// Returns true if this was the last active user and AXCL can be finalized.
bool AxclReleaseEngineMaybeFinalize(axclrtContext ctx, int runtime_device_id) {
    auto& g = Global();
    if (g.refcnt <= 0) return false;
    g.refcnt--;
    if (g.refcnt != 0) {
        return false;
    }
    if (g.engine_inited) {
        // axclrtEngineFinalize requires a bound current context; otherwise AXCL logs
        // "thread hasn't binded any context yet" (AXCL_ERR_CONTEXT_NO_BIND_CONTEXT).
        // Some applications alternate between multiple contexts (e.g. video SDK + NPU),
        // so be defensive here and create a temporary context if the provided one cannot be bound.
        if (runtime_device_id >= 0) {
            (void)axclrtSetDevice(runtime_device_id);
        }
        axclError set_ctx_ret = AXCL_FAIL;
        if (ctx) {
            set_ctx_ret = axclrtSetCurrentContext(ctx);
        }
        axclrtContext tmp_ctx = nullptr;
        if (set_ctx_ret != AXCL_SUCC && runtime_device_id >= 0) {
            if (axclrtCreateContext(&tmp_ctx, runtime_device_id) == AXCL_SUCC && tmp_ctx != nullptr) {
                if (axclrtSetCurrentContext(tmp_ctx) != AXCL_SUCC) {
                    (void)axclrtDestroyContext(tmp_ctx);
                    tmp_ctx = nullptr;
                }
            }
        }

        if (set_ctx_ret == AXCL_SUCC || tmp_ctx != nullptr) {
            (void)axclrtEngineFinalize();
        }

        if (tmp_ctx != nullptr) {
            (void)axclrtDestroyContext(tmp_ctx);
        }
        g.engine_inited = false;
    }
    return true;
}

void AxclFinalizeIfNeeded() {
    auto& g = Global();
    if (!g.axcl_inited) return;
    (void)axclFinalize();
    g.axcl_inited = false;
}
#else
int AxclAcquire() { return 0; }
int AxclEnsureEngineInited() { return 0; }
bool AxclReleaseEngineMaybeFinalize(axclrtContext, int) { return false; }
void AxclFinalizeIfNeeded() {}
#endif

}  // namespace

typedef struct
{
    int nIndex;
    int nSize;
    void *pBuf;
    void *pVirAddr;

    std::string Name;

    axclrtEngineIODims dims;
} AXCL_IO_BUF_T;

typedef struct
{
    uint32_t nInputSize;
    uint32_t nOutputSize;
    AXCL_IO_BUF_T *pInputs;
    AXCL_IO_BUF_T *pOutputs;
} AXCL_IO_DATA_T;

static void free_io_index(AXCL_IO_BUF_T *pBuf, size_t index)
{
    for (size_t i = 0; i < index; ++i)
    {
        (void)axclrtFree(pBuf[i].pBuf);
        if (pBuf[i].pVirAddr) std::free(pBuf[i].pVirAddr);
    }
}

static void free_io(AXCL_IO_DATA_T *io_data)
{
    for (size_t j = 0; j < io_data->nInputSize; ++j)
    {
        (void)axclrtFree(io_data->pInputs[j].pBuf);
        if (io_data->pInputs[j].pVirAddr) std::free(io_data->pInputs[j].pVirAddr);
    }
    for (size_t j = 0; j < io_data->nOutputSize; ++j)
    {
        (void)axclrtFree(io_data->pOutputs[j].pBuf);
        if (io_data->pOutputs[j].pVirAddr) std::free(io_data->pOutputs[j].pVirAddr);
    }
    delete[] io_data->pInputs;
    delete[] io_data->pOutputs;
}

static inline int prepare_io(int grpid,
                             axclrtEngineIOInfo io_info,
                             axclrtEngineIO io,
                             AXCL_IO_DATA_T *io_data)
{
    std::memset(io_data, 0, sizeof(AXCL_IO_DATA_T));

    const auto inputNum = axclrtEngineGetNumInputs(io_info);
    const auto outputNum = axclrtEngineGetNumOutputs(io_info);
    io_data->nInputSize = inputNum;
    io_data->nOutputSize = outputNum;
    io_data->pInputs = new AXCL_IO_BUF_T[inputNum];
    io_data->pOutputs = new AXCL_IO_BUF_T[outputNum];

    // 1. alloc inputs
    for (uint32_t i = 0; i < inputNum; i++)
    {
        const auto bufSize = axclrtEngineGetInputSizeByIndex(io_info, grpid, i);
        void *devPtr = nullptr;
        const axclError ret = axclrtMalloc(&devPtr, bufSize, AXCL_MEM_MALLOC_HUGE_FIRST);
        if (ret != AXCL_SUCC || devPtr == nullptr)
        {
            free_io_index(io_data->pInputs, i);
            ALOGE("axclrtMalloc input(index=%u size=%lu) failed ret=0x%x", i, bufSize, ret);
            return -1;
        }
        // zero init on device
        std::vector<char> tmp(bufSize, 0);
        (void)axclrtMemcpy(devPtr, tmp.data(), bufSize, AXCL_MEMCPY_HOST_TO_DEVICE);

        axclrtEngineIODims dims{};
        const axclError dim_ret = axclrtEngineGetInputDims(io_info, grpid, i, &dims);
        if (dim_ret != AXCL_SUCC)
        {
            free_io_index(io_data->pInputs, i);
            ALOGE("axclrtEngineGetInputDims(index=%u) failed ret=0x%x", i, dim_ret);
            return -1;
        }

        io_data->pInputs[i].nIndex = static_cast<int>(i);
        io_data->pInputs[i].nSize = static_cast<int>(bufSize);
        io_data->pInputs[i].pBuf = devPtr;
        io_data->pInputs[i].dims = dims;
        io_data->pInputs[i].Name = axclrtEngineGetInputNameByIndex(io_info, i);
        io_data->pInputs[i].pVirAddr = std::malloc(bufSize);
        if (!io_data->pInputs[i].pVirAddr) {
            free_io_index(io_data->pInputs, i + 1);
            ALOGE("malloc host input buffer failed size=%lu", bufSize);
            return -1;
        }
        std::memset(io_data->pInputs[i].pVirAddr, 0, bufSize);

        const axclError set_ret = axclrtEngineSetInputBufferByIndex(io, i, devPtr, bufSize);
        if (set_ret != AXCL_SUCC)
        {
            free_io_index(io_data->pInputs, i + 1);
            ALOGE("axclrtEngineSetInputBufferByIndex(index=%u size=%lu) failed ret=0x%x", i, bufSize, set_ret);
            return -1;
        }
    }

    // 2. alloc outputs
    for (uint32_t i = 0; i < outputNum; i++)
    {
        const auto bufSize = axclrtEngineGetOutputSizeByIndex(io_info, grpid, i);
        void *devPtr = nullptr;
        const axclError ret = axclrtMalloc(&devPtr, bufSize, AXCL_MEM_MALLOC_HUGE_FIRST);
        if (ret != AXCL_SUCC || devPtr == nullptr)
        {
            free_io_index(io_data->pOutputs, i);
            ALOGE("axclrtMalloc output(index=%u size=%lu) failed ret=0x%x", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        (void)axclrtMemcpy(devPtr, tmp.data(), bufSize, AXCL_MEMCPY_HOST_TO_DEVICE);

        axclrtEngineIODims dims{};
        const axclError dim_ret = axclrtEngineGetOutputDims(io_info, grpid, i, &dims);
        if (dim_ret != AXCL_SUCC)
        {
            free_io_index(io_data->pOutputs, i);
            ALOGE("axclrtEngineGetOutputDims(index=%u) failed ret=0x%x", i, dim_ret);
            return -1;
        }

        io_data->pOutputs[i].nIndex = static_cast<int>(i);
        io_data->pOutputs[i].nSize = static_cast<int>(bufSize);
        io_data->pOutputs[i].pBuf = devPtr;
        io_data->pOutputs[i].dims = dims;
        io_data->pOutputs[i].Name = axclrtEngineGetOutputNameByIndex(io_info, i);
        io_data->pOutputs[i].pVirAddr = std::malloc(bufSize);
        if (!io_data->pOutputs[i].pVirAddr) {
            free_io_index(io_data->pOutputs, i + 1);
            ALOGE("malloc host output buffer failed size=%lu", bufSize);
            return -1;
        }
        std::memset(io_data->pOutputs[i].pVirAddr, 0, bufSize);

        const axclError set_ret = axclrtEngineSetOutputBufferByIndex(io, i, devPtr, bufSize);
        if (set_ret != AXCL_SUCC)
        {
            free_io_index(io_data->pOutputs, i + 1);
            ALOGE("axclrtEngineSetOutputBufferByIndex(index=%u size=%lu) failed ret=0x%x", i, bufSize, set_ret);
            return -1;
        }
    }

    return 0;
}

struct ax_joint_runner_axcl_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = nullptr;
    std::vector<axclrtEngineIO> ios;
    std::vector<AXCL_IO_DATA_T> io_datas;

    axclrtContext rt_ctx = nullptr;
    int runtime_id = -1;
    bool own_rt_ctx = false;
};

int ax_runner_axcl::sub_init()
{
    int ret = axclrtEngineCreateContext(m_handle->handle, &m_handle->context);
    if (ret != AXCL_SUCC)
    {
        ALOGE("axclrtEngineCreateContext failed ret=0x%x", ret);
        return ret;
    }

    ret = axclrtEngineGetIOInfo(m_handle->handle, &m_handle->io_info);
    if (ret != AXCL_SUCC)
    {
        ALOGE("axclrtEngineGetIOInfo failed ret=0x%x", ret);
        return ret;
    }

    ret = axclrtEngineGetShapeGroupsCount(m_handle->io_info, &group_count);
    if (ret != AXCL_SUCC)
    {
        ALOGE("axclrtEngineGetShapeGroupsCount failed ret=0x%x", ret);
        return ret;
    }

    m_handle->ios.resize(group_count);
    m_handle->io_datas.resize(group_count);
    mgroup_input_tensors.resize(group_count);
    mgroup_output_tensors.resize(group_count);

    for (int grpid = 0; grpid < group_count; grpid++)
    {
        ret = axclrtEngineCreateIO(m_handle->io_info, &m_handle->ios[grpid]);
        if (ret != AXCL_SUCC)
        {
            ALOGE("axclrtEngineCreateIO failed grpid=%d ret=0x%x", grpid, ret);
            return ret;
        }

        ret = prepare_io(grpid, m_handle->io_info, m_handle->ios[grpid], &m_handle->io_datas[grpid]);
        if (ret != 0)
        {
            ALOGE("prepare_io failed grpid=%d", grpid);
            return ret;
        }
    }

    for (int grpid = 0; grpid < group_count; grpid++)
    {
        auto &io_data = m_handle->io_datas[grpid];
        for (uint32_t i = 0; i < io_data.nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = io_data.pOutputs[i].Name;
            tensor.nSize = io_data.pOutputs[i].nSize;
            for (int32_t j = 0; j < io_data.pOutputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(static_cast<unsigned int>(io_data.pOutputs[i].dims.dims[j]));
            }
            tensor.phyAddr = (unsigned long)io_data.pOutputs[i].pBuf;
            tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
            mgroup_output_tensors[grpid].push_back(tensor);
        }

        for (uint32_t i = 0; i < io_data.nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = io_data.pInputs[i].Name;
            tensor.nSize = io_data.pInputs[i].nSize;
            for (int32_t j = 0; j < io_data.pInputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(static_cast<unsigned int>(io_data.pInputs[i].dims.dims[j]));
            }
            tensor.phyAddr = (unsigned long)io_data.pInputs[i].pBuf;
            tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
            mgroup_input_tensors[grpid].push_back(tensor);
        }
    }

    moutput_tensors = mgroup_output_tensors[0];
    minput_tensors = mgroup_input_tensors[0];

    return 0;
}

int ax_runner_axcl::init(const void *model_data, unsigned int model_size, int devid)
{
    if (m_handle) {
        return -1;
    }
    m_handle = new ax_joint_runner_axcl_handle_t;
    std::memset((void *)m_handle, 0, sizeof(ax_joint_runner_axcl_handle_t));
    _devid = devid;

#if defined(DEEPSORT_AXCL_INIT_IN_RUNNER)
    const int acq = AxclAcquire();
    if (acq != 0) {
        delete m_handle;
        m_handle = nullptr;
        return acq;
    }

    axclrtDeviceList lst{};
    const axclError list_ret = axclrtGetDeviceList(&lst);
    if (list_ret != AXCL_SUCC || lst.num == 0) {
        ALOGE("axclrtGetDeviceList failed ret=0x%x num=%u", list_ret, lst.num);
        deinit();
        return -1;
    }

    if (_devid < 0) _devid = 0;
    if (_devid >= static_cast<int>(lst.num)) {
        ALOGE("invalid device index %d, total=%u", _devid, lst.num);
        deinit();
        return -1;
    }

    m_handle->runtime_id = lst.devices[_devid];
    if (axclrtSetDevice(m_handle->runtime_id) != AXCL_SUCC) {
        ALOGE("axclrtSetDevice failed runtime_id=%d", m_handle->runtime_id);
        deinit();
        return -1;
    }

    if (axclrtCreateContext(&m_handle->rt_ctx, m_handle->runtime_id) != AXCL_SUCC || m_handle->rt_ctx == nullptr) {
        ALOGE("axclrtCreateContext failed runtime_id=%d", m_handle->runtime_id);
        deinit();
        return -1;
    }
    m_handle->own_rt_ctx = true;
    if (axclrtSetCurrentContext(m_handle->rt_ctx) != AXCL_SUCC) {
        ALOGE("axclrtSetCurrentContext failed");
        deinit();
        return -1;
    }

    const int eng_ret = AxclEnsureEngineInited();
    if (eng_ret != 0) {
        deinit();
        return eng_ret;
    }
#else
    (void)_devid;
    axclrtContext current_ctx = nullptr;
    const axclError ctx_ret = axclrtGetCurrentContext(&current_ctx);
    if (ctx_ret != AXCL_SUCC || current_ctx == nullptr) {
        ALOGE("axclrtGetCurrentContext failed ret=0x%x (need app to create/bind context)", ctx_ret);
        deinit();
        return -1;
    }
    m_handle->rt_ctx = current_ctx;
    m_handle->own_rt_ctx = false;
    int32_t current_device = -1;
    if (axclrtGetDevice(&current_device) == AXCL_SUCC) {
        m_handle->runtime_id = current_device;
    }
#endif

    // Load model: axclrtEngineLoadFromMem requires device memory.
    void *devMem = nullptr;
    const axclError mret = axclrtMalloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY);
    if (mret != AXCL_SUCC || devMem == nullptr) {
        ALOGE("axclrtMalloc(model) failed ret=0x%x size=%u", mret, model_size);
        deinit();
        return -1;
    }
    const axclError cpy = axclrtMemcpy(devMem, model_data, model_size, AXCL_MEMCPY_HOST_TO_DEVICE);
    if (cpy != AXCL_SUCC) {
        ALOGE("axclrtMemcpy(H2D,model) failed ret=0x%x", cpy);
        (void)axclrtFree(devMem);
        deinit();
        return -1;
    }

    const axclError load_ret = axclrtEngineLoadFromMem(devMem, model_size, &m_handle->handle);
    (void)axclrtFree(devMem);
    if (load_ret != AXCL_SUCC) {
        ALOGE("axclrtEngineLoadFromMem failed ret=0x%x", load_ret);
        deinit();
        return -1;
    }

    return sub_init();
}

void ax_runner_axcl::deinit()
{
    if (m_handle) {
        bool need_finalize_axcl = false;
        if (m_handle->runtime_id >= 0) {
            (void)axclrtSetDevice(m_handle->runtime_id);
        }
        if (m_handle->rt_ctx) {
            (void)axclrtSetCurrentContext(m_handle->rt_ctx);
        }
        if (m_handle->handle) {
            for (int grpid = 0; grpid < group_count; grpid++) {
                if (!m_handle->io_datas.empty()) {
                    free_io(&m_handle->io_datas[grpid]);
                }
                if (!m_handle->ios.empty() && m_handle->ios[grpid]) {
                    (void)axclrtEngineDestroyIO(m_handle->ios[grpid]);
                }
            }
            if (m_handle->io_info) {
                (void)axclrtEngineDestroyIOInfo(m_handle->io_info);
                m_handle->io_info = nullptr;
            }
            (void)axclrtEngineUnload(m_handle->handle);
            m_handle->handle = 0;
        }

        need_finalize_axcl = AxclReleaseEngineMaybeFinalize(m_handle->rt_ctx, m_handle->runtime_id);

        if (m_handle->rt_ctx && m_handle->own_rt_ctx) {
            (void)axclrtDestroyContext(m_handle->rt_ctx);
        }
        m_handle->rt_ctx = nullptr;
        delete m_handle;
        m_handle = nullptr;

        if (need_finalize_axcl) {
            AxclFinalizeIfNeeded();
        }
    }

    group_count = 0;
    minput_tensors.clear();
    moutput_tensors.clear();
    map_input_tensors.clear();
    map_output_tensors.clear();
    mgroup_input_tensors.clear();
    mgroup_output_tensors.clear();
    map_group_input_tensors.clear();
    map_group_output_tensors.clear();

    // AXCL global finalize happens in deinit() when the last runner releases.
}

int ax_runner_axcl::set_affinity(int id)
{
    // axclrtEngineSetAffinity expects a bitmask-like set value.
    return axclrtEngineSetAffinity(m_handle->handle, static_cast<axclrtEngineSet>(id));
}

int ax_runner_axcl::sync_input(int idx)
{
    auto &input = get_input(idx);
    return axclrtMemcpy((void *)input.phyAddr, input.pVirAddr, input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
}

int ax_runner_axcl::sync_input(std::string name)
{
    auto &input = get_input(name);
    return axclrtMemcpy((void *)input.phyAddr, input.pVirAddr, input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
}

int ax_runner_axcl::sync_output(int idx)
{
    auto &output = get_output(idx);
    return axclrtMemcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
}

int ax_runner_axcl::sync_output(std::string name)
{
    auto &output = get_output(name);
    return axclrtMemcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
}

int ax_runner_axcl::sync_input(int grpid, int idx)
{
    auto &input = get_input(grpid, idx);
    return axclrtMemcpy((void *)input.phyAddr, input.pVirAddr, input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
}

int ax_runner_axcl::sync_input(int grpid, std::string name)
{
    auto &input = get_input(grpid, name);
    return axclrtMemcpy((void *)input.phyAddr, input.pVirAddr, input.nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
}

int ax_runner_axcl::sync_output(int grpid, int idx)
{
    auto &output = get_output(grpid, idx);
    return axclrtMemcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
}

int ax_runner_axcl::sync_output(int grpid, std::string name)
{
    auto &output = get_output(grpid, name);
    return axclrtMemcpy(output.pVirAddr, (void *)output.phyAddr, output.nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
}

int ax_runner_axcl::set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetInputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size);
}

int ax_runner_axcl::set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetOutputBufferByIndex(m_handle->ios[grpid], idx, (void *)phy_addr, size);
}

int ax_runner_axcl::set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetInputBufferByIndex(m_handle->ios[grpid], get_input(grpid, name).nIdx, (void *)phy_addr, size);
}

int ax_runner_axcl::set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    return axclrtEngineSetOutputBufferByIndex(m_handle->ios[grpid], get_output(grpid, name).nIdx, (void *)phy_addr, size);
}

int ax_runner_axcl::inference()
{
    return inference(0);
}

int ax_runner_axcl::inference(int grpid)
{
    if (!m_handle) return -1;
    if (m_handle->rt_ctx) {
        (void)axclrtSetCurrentContext(m_handle->rt_ctx);
    }

    if (_auto_sync_before_inference) {
        for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++) {
            (void)axclrtMemcpy((void *)mgroup_input_tensors[grpid][i].phyAddr,
                               mgroup_input_tensors[grpid][i].pVirAddr,
                               mgroup_input_tensors[grpid][i].nSize,
                               AXCL_MEMCPY_HOST_TO_DEVICE);
        }
    }

    const int ret = axclrtEngineExecute(m_handle->handle, m_handle->context, static_cast<uint32_t>(grpid), m_handle->ios[grpid]);
    if (ret != AXCL_SUCC)
    {
        ALOGE("axclrtEngineExecute failed ret=0x%x", ret);
        return ret;
    }

    if (_auto_sync_after_inference) {
        for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++) {
            (void)axclrtMemcpy(mgroup_output_tensors[grpid][i].pVirAddr,
                               (void *)mgroup_output_tensors[grpid][i].phyAddr,
                               mgroup_output_tensors[grpid][i].nSize,
                               AXCL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    return 0;
}
