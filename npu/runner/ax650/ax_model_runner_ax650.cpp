#include "ax_model_runner_ax650.hpp"

#include "npu/runner/logging.hpp"

// NOTE: AX650/MSP runner depends on MSP headers/libs (ax_sys_api.h, ax_engine_api.h).
// Keep this implementation guarded so x86/AXCL builds don't break.
#if defined(DEEPSORT_HAVE_AX650)

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include <cstring>
#include <mutex>

#define AX_CMM_ALIGN_SIZE 128

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) ((((x) + ((align) - 1)) / (align)) * (align))
#endif

const char *AX_CMM_SESSION_NAME = "npu";

typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

struct ax_joint_runner_ax650_handle_t
{
    AX_ENGINE_HANDLE handle{};
    AX_ENGINE_CONTEXT_T context{};
    std::vector<AX_ENGINE_IO_INFO_T *> io_info;
    std::vector<AX_ENGINE_IO_T> io_data;
};

namespace {

std::mutex g_engine_mu;
int g_engine_refcnt = 0;

int EngineAcquire() {
    std::lock_guard<std::mutex> lock(g_engine_mu);
    if (g_engine_refcnt == 0) {
        AX_ENGINE_NPU_ATTR_T npu_attr;
        std::memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        const int ret = AX_ENGINE_Init(&npu_attr);
        if (ret != 0) {
            ALOGE("AX_ENGINE_Init failed ret=%d", ret);
            return ret;
        }
    }
    ++g_engine_refcnt;
    return 0;
}

void EngineRelease() {
    std::lock_guard<std::mutex> lock(g_engine_mu);
    if (g_engine_refcnt <= 0) return;
    --g_engine_refcnt;
    if (g_engine_refcnt == 0) {
        AX_ENGINE_Deinit();
    }
}

void free_io_index(AX_ENGINE_IO_BUFFER_T *io_buf, int index)
{
    for (int i = 0; i < index; ++i)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io_buf + i;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
}

void free_io(AX_ENGINE_IO_T *io)
{
    for (size_t j = 0; j < io->nInputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pInputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    for (size_t j = 0; j < io->nOutputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pOutputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    delete[] io->pInputs;
    delete[] io->pOutputs;
}

static inline int prepare_io(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy)
{
    std::memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io_data->nInputSize = info->nInputSize;
    std::memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);

    auto ret = 0;
    for (uint i = 0; i < info->nInputSize; ++i)
    {
        auto meta = info->pInputs[i];
        auto buffer = &io_data->pInputs[i];
        buffer->nSize = meta.nSize;
        if (strategy.first == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            ALOGE("Allocate input{%d} fail ret=%d", i, ret);
            return ret;
        }
        std::memset(buffer->pVirAddr, 0, meta.nSize);
    }

    io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io_data->nOutputSize = info->nOutputSize;
    std::memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
    for (uint i = 0; i < info->nOutputSize; ++i)
    {
        auto meta = info->pOutputs[i];
        auto buffer = &io_data->pOutputs[i];
        buffer->nSize = meta.nSize;
        if (strategy.second == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0)
        {
            ALOGE("Allocate output{%d} fail ret=%d", i, ret);
            free_io_index(io_data->pInputs, io_data->nInputSize);
            free_io_index(io_data->pOutputs, i);
            return ret;
        }
        std::memset(buffer->pVirAddr, 0, meta.nSize);
    }

    return 0;
}

}  // namespace

int ax_runner_ax650::init(const void *model_data, unsigned int model_size, int devid)
{
    if (m_handle)
    {
        return -1;
    }
    m_handle = new ax_joint_runner_ax650_handle_t;
    _devid = devid;

    const int eret = EngineAcquire();
    if (eret != 0) {
        delete m_handle;
        m_handle = nullptr;
        return eret;
    }

    int ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_data, model_size);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle failed ret=%d", ret);
        delete m_handle;
        m_handle = nullptr;
        EngineRelease();
        return ret;
    }

    ret = AX_ENGINE_CreateContext(m_handle->handle);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext failed ret=%d", ret);
        AX_ENGINE_DestroyHandle(m_handle->handle);
        delete m_handle;
        m_handle = nullptr;
        EngineRelease();
        return ret;
    }
    ret = AX_ENGINE_CreateContextV2(m_handle->handle, &m_handle->context);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContextV2 failed ret=%d", ret);
        AX_ENGINE_DestroyHandle(m_handle->handle);
        delete m_handle;
        m_handle = nullptr;
        EngineRelease();
        return ret;
    }

    AX_U32 io_count = 0;
    ret = AX_ENGINE_GetGroupIOInfoCount(m_handle->handle, &io_count);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetGroupIOInfoCount failed ret=%d", ret);
        AX_ENGINE_DestroyHandle(m_handle->handle);
        delete m_handle;
        m_handle = nullptr;
        EngineRelease();
        return ret;
    }

    m_handle->io_info.resize(io_count);
    m_handle->io_data.resize(io_count);
    mgroup_input_tensors.resize(io_count);
    mgroup_output_tensors.resize(io_count);

    for (AX_U32 grpid = 0; grpid < io_count; grpid++)
    {
        AX_ENGINE_IO_INFO_T *io_info = nullptr;
        ret = AX_ENGINE_GetGroupIOInfo(m_handle->handle, grpid, &io_info);
        if (0 != ret)
        {
            ALOGE("AX_ENGINE_GetGroupIOInfo failed ret=%d", ret);
            AX_ENGINE_DestroyHandle(m_handle->handle);
            delete m_handle;
            m_handle = nullptr;
            EngineRelease();
            return ret;
        }
        m_handle->io_info[grpid] = io_info;

        ret = prepare_io(m_handle->io_info[grpid], &m_handle->io_data[grpid], std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        if (0 != ret)
        {
            ALOGE("prepare_io failed grpid=%u ret=%d", grpid, ret);
            AX_ENGINE_DestroyHandle(m_handle->handle);
            delete m_handle;
            m_handle = nullptr;
            EngineRelease();
            return ret;
        }
    }

    for (size_t grpid = 0; grpid < io_count; grpid++)
    {
        auto &io_info = m_handle->io_info[grpid];
        auto &io_data = m_handle->io_data[grpid];
        for (size_t i = 0; i < io_info->nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = std::string(io_info->pOutputs[i].pName);
            tensor.nSize = io_info->pOutputs[i].nSize;
            for (size_t j = 0; j < io_info->pOutputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(io_info->pOutputs[i].pShape[j]);
            }
            tensor.phyAddr = io_data.pOutputs[i].phyAddr;
            tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
            mgroup_output_tensors[grpid].push_back(tensor);
        }

        for (size_t i = 0; i < io_info->nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = std::string(io_info->pInputs[i].pName);
            tensor.nSize = io_info->pInputs[i].nSize;
            for (size_t j = 0; j < io_info->pInputs[i].nShapeSize; j++)
            {
                tensor.vShape.push_back(io_info->pInputs[i].pShape[j]);
            }
            tensor.phyAddr = io_data.pInputs[i].phyAddr;
            tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
            mgroup_input_tensors[grpid].push_back(tensor);
        }
    }

    moutput_tensors = mgroup_output_tensors[0];
    minput_tensors = mgroup_input_tensors[0];

    return 0;
}

void ax_runner_ax650::deinit()
{
    if (m_handle && m_handle->handle)
    {
        for (size_t i = 0; i < m_handle->io_data.size(); i++)
        {
            free_io(&m_handle->io_data[i]);
        }
        AX_ENGINE_DestroyHandle(m_handle->handle);
    }
    delete m_handle;
    m_handle = nullptr;
    EngineRelease();
}

int ax_runner_ax650::set_affinity(int id)
{
    return AX_ENGINE_SetAffinity(m_handle->handle, id);
}

int ax_runner_ax650::set_input(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    if (!m_handle) return -1;
    if (grpid < 0 || static_cast<size_t>(grpid) >= m_handle->io_data.size()) return -1;
    auto &io = m_handle->io_data[static_cast<size_t>(grpid)];
    if (idx < 0 || static_cast<AX_U32>(idx) >= io.nInputSize) return -1;
    if (phy_addr == 0 || size == 0) return -1;

    AX_S32 mem_type = 0;
    AX_VOID *vir = nullptr;
    AX_U32 block_size = 0;
    const AX_S32 ret = AX_SYS_MemGetBlockInfoByPhy(static_cast<AX_U64>(phy_addr), &mem_type, &vir, &block_size);
    (void)mem_type;
    if (ret != AX_SUCCESS || vir == nullptr) {
        ALOGE("AX_SYS_MemGetBlockInfoByPhy failed ret=%d phy=0x%llx", ret, phy_addr);
        return -1;
    }
    if (block_size < static_cast<AX_U32>(size)) {
        ALOGE("AX_SYS_MemGetBlockInfoByPhy block too small block=%u need=%lu", block_size, size);
        return -1;
    }

    io.pInputs[idx].phyAddr = static_cast<AX_U64>(phy_addr);
    io.pInputs[idx].pVirAddr = vir;
    io.pInputs[idx].nSize = static_cast<AX_U32>(size);

    if (grpid >= 0 && static_cast<size_t>(grpid) < mgroup_input_tensors.size() &&
        idx >= 0 && static_cast<size_t>(idx) < mgroup_input_tensors[static_cast<size_t>(grpid)].size()) {
        auto& tensor = mgroup_input_tensors[static_cast<size_t>(grpid)][static_cast<size_t>(idx)];
        tensor.phyAddr = static_cast<unsigned long>(phy_addr);
        tensor.pVirAddr = vir;
        tensor.nSize = static_cast<int>(size);
    }
    if (grpid == 0 && idx >= 0 && static_cast<size_t>(idx) < minput_tensors.size()) {
        auto& tensor = minput_tensors[static_cast<size_t>(idx)];
        tensor.phyAddr = static_cast<unsigned long>(phy_addr);
        tensor.pVirAddr = vir;
        tensor.nSize = static_cast<int>(size);
    }
    map_input_tensors.clear();
    map_group_input_tensors.clear();
    return 0;
}

int ax_runner_ax650::set_output(int grpid, int idx, unsigned long long int phy_addr, unsigned long size)
{
    if (!m_handle) return -1;
    if (grpid < 0 || static_cast<size_t>(grpid) >= m_handle->io_data.size()) return -1;
    auto &io = m_handle->io_data[static_cast<size_t>(grpid)];
    if (idx < 0 || static_cast<AX_U32>(idx) >= io.nOutputSize) return -1;
    if (phy_addr == 0 || size == 0) return -1;

    AX_S32 mem_type = 0;
    AX_VOID *vir = nullptr;
    AX_U32 block_size = 0;
    const AX_S32 ret = AX_SYS_MemGetBlockInfoByPhy(static_cast<AX_U64>(phy_addr), &mem_type, &vir, &block_size);
    (void)mem_type;
    if (ret != AX_SUCCESS || vir == nullptr) {
        ALOGE("AX_SYS_MemGetBlockInfoByPhy failed ret=%d phy=0x%llx", ret, phy_addr);
        return -1;
    }
    if (block_size < static_cast<AX_U32>(size)) {
        ALOGE("AX_SYS_MemGetBlockInfoByPhy block too small block=%u need=%lu", block_size, size);
        return -1;
    }

    io.pOutputs[idx].phyAddr = static_cast<AX_U64>(phy_addr);
    io.pOutputs[idx].pVirAddr = vir;
    io.pOutputs[idx].nSize = static_cast<AX_U32>(size);

    if (grpid >= 0 && static_cast<size_t>(grpid) < mgroup_output_tensors.size() &&
        idx >= 0 && static_cast<size_t>(idx) < mgroup_output_tensors[static_cast<size_t>(grpid)].size()) {
        auto& tensor = mgroup_output_tensors[static_cast<size_t>(grpid)][static_cast<size_t>(idx)];
        tensor.phyAddr = static_cast<unsigned long>(phy_addr);
        tensor.pVirAddr = vir;
        tensor.nSize = static_cast<int>(size);
    }
    if (grpid == 0 && idx >= 0 && static_cast<size_t>(idx) < moutput_tensors.size()) {
        auto& tensor = moutput_tensors[static_cast<size_t>(idx)];
        tensor.phyAddr = static_cast<unsigned long>(phy_addr);
        tensor.pVirAddr = vir;
        tensor.nSize = static_cast<int>(size);
    }
    map_output_tensors.clear();
    map_group_output_tensors.clear();
    return 0;
}

int ax_runner_ax650::set_input(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    const auto &t = get_input(grpid, name);
    return set_input(grpid, t.nIdx, phy_addr, size);
}

int ax_runner_ax650::set_output(int grpid, std::string name, unsigned long long int phy_addr, unsigned long size)
{
    const auto &t = get_output(grpid, name);
    return set_output(grpid, t.nIdx, phy_addr, size);
}

int ax_runner_ax650::sync_output(int idx)
{
    auto &tensor = get_output(idx);
    AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    return 0;
}

int ax_runner_ax650::sync_output(std::string name)
{
    auto &tensor = get_output(name);
    AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    return 0;
}

int ax_runner_ax650::sync_output(int grpid, int idx)
{
    auto &tensor = get_output(grpid, idx);
    AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    return 0;
}

int ax_runner_ax650::sync_output(int grpid, std::string name)
{
    auto &tensor = get_output(grpid, name);
    AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    return 0;
}

int ax_runner_ax650::inference()
{
    const int ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);
    for (size_t i = 0; i < get_num_outputs(); i++)
    {
        auto &tensor = get_output(i);
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}

int ax_runner_ax650::inference(int grpid)
{
    const int ret = AX_ENGINE_RunGroupIOSync(m_handle->handle, m_handle->context, grpid, &m_handle->io_data[grpid]);
    for (size_t i = 0; i < get_num_outputs(); i++)
    {
        auto &tensor = get_output(grpid, i);
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}

#else  // !DEEPSORT_HAVE_AX650

int ax_runner_ax650::init(const void *, unsigned int, int) { return -1; }
void ax_runner_ax650::deinit() {}
int ax_runner_ax650::set_affinity(int) { return -1; }
int ax_runner_ax650::set_input(int, int, unsigned long long int, unsigned long) { return -1; }
int ax_runner_ax650::set_output(int, int, unsigned long long int, unsigned long) { return -1; }
int ax_runner_ax650::set_input(int, std::string, unsigned long long int, unsigned long) { return -1; }
int ax_runner_ax650::set_output(int, std::string, unsigned long long int, unsigned long) { return -1; }
int ax_runner_ax650::sync_output(int) { return -1; }
int ax_runner_ax650::sync_output(std::string) { return -1; }
int ax_runner_ax650::sync_output(int, int) { return -1; }
int ax_runner_ax650::sync_output(int, std::string) { return -1; }
int ax_runner_ax650::inference() { return -1; }
int ax_runner_ax650::inference(int) { return -1; }

#endif  // DEEPSORT_HAVE_AX650
