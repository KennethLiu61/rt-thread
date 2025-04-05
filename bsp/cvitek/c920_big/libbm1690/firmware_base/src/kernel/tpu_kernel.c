#include <rtthread.h>
#include "tpu_kernel_internel.h"
#include "atomic_sdma_gen_cmd.h"
#include "cdma_reg_value.h"
#include "base_def.h"
#include "firmware_pmu.h"

#define M_PI        3.14159265358979323846

#define MAX_C2C_CDMA_NUM 6

#ifdef REMOVE_POLLS_IN_LLM
const char remove_polls_flag[] = "REMOVE_POLLS_IN_LLM";
#endif

int tpu_npu_num() {
    return NPU_NUM;
}
int tpu_local_mem_size_per_npu() {
    return LOCAL_MEM_SIZE;
}
int tpu_l2_sram_size() {
    return L2_SRAM_SIZE;
}
int tpu_get_ic_parallel(data_type_t dtype) {
    return get_conv_ic_parallel(PRECISION(dtype));
}
int tpu_npu_c_offset(int c, int c_stride, data_type_t dtype) {
    return (c / NPU_NUM) * c_stride * tpu_data_type_size(dtype);
}
int tpu_unified_c_offset(int c, int c_stride, data_type_t dtype) {
    return (c % NPU_NUM) * LOCAL_MEM_SIZE + tpu_npu_c_offset(c, c_stride, dtype);
}
unsigned long long tpu_l2_sram_get_start_addr() {
    return L2_SRAM_START_ADDR | (L2M_TAG << 40);
}
RTM_EXPORT(tpu_l2_sram_get_start_addr);
unsigned long long tpu_global_mem_get_start_addr() {
    return GLOBAL_MEM_START_ADDR;
}
unsigned long long tpu_local_mem_get_start_addr() {
    return LOCAL_MEM_START_ADDR;
}

unsigned long long tpu_static_mem_get_start_addr() {
    return STATIC_MEM_START_ADDR;
}

local_addr_t tpu_npu_addr(local_addr_t addr) {
    return addr & (LOCAL_MEM_SIZE - 1);
}
int tpu_npu_index(local_addr_t addr) {
    return addr / LOCAL_MEM_SIZE;
}
int tpu_channle_num_per_npu(int start_idx, int num_channels) {
    return DIV_UP(start_idx + num_channels, NPU_NUM);
}
u64 tpu_aligned_feature_size(int h, int w, data_type_t dtype) {
    return (u64)ALIGN(ALIGN((u64)h * w, tpu_eu_num(dtype)) * tpu_data_type_bits(dtype), 8) / 8;
}
int tpu_conv_kernel_size_per_oc(int ic, const dim2* ker, data_type_t dtype) {
    return ALIGN(ALIGN(ic, tpu_get_ic_parallel(dtype)) * ker->h * ker->w * tpu_data_type_bits(dtype), 8) / 8;
}
int tpu_conv_kernel_size(int oc, int ic, const dim2* ker, data_type_t dtype) {
    return ALIGN(DIV_UP(oc, tpu_npu_num()) * ALIGN(ic, tpu_get_ic_parallel(dtype)) * ker->h * ker->w * tpu_data_type_bits(dtype), 8) / 8;
}
int sfu_taylor_exp_len(data_type_t dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype) && "exp only support floating point\n");
    return (dtype == DT_FP32 ? 10 : 7);
}
void tpu_aligned_stride(
    dim4        *stride,
    int          start_idx,
    const dim4  *shape,
    data_type_t  dtype) {
    stride->w = 1;
    stride->h = shape->w;
    stride->c = shape->h * stride->h;
    stride->c = ALIGN(stride->c, tpu_eu_num(dtype));
    stride->n = DIV_UP(start_idx + shape->c, NPU_NUM) * stride->c;
}
void tpu_compact_stride(dim4 *stride, int start_idx, const dim4 *shape) {
    stride->w = 1;
    stride->h = shape->w;
    stride->c = shape->h * stride->h;
    stride->n = DIV_UP(start_idx + shape->c, NPU_NUM) * stride->c;
}

void tpu_line_aligned_stride(
    dim4        *stride,
    int          start_idx,
    const dim4  *shape,
    data_type_t  dtype) {
    stride->w = 1;
    stride->h = ALIGN(shape->w, tpu_eu_num(dtype));
    stride->c = shape->h * stride->h;
    stride->n = DIV_UP(start_idx + shape->c, NPU_NUM) * stride->c;
}
void tpu_continuous_stride(dim4 *stride, const dim4  *shape) {
    stride->w = 1;
    stride->h = shape->w;
    stride->c = shape->h * shape->w;
    stride->n = shape->c * stride->c;
}
range_t tpu_bank_range(local_addr_t addr, int size) {
    range_t res = {
        .start = tpu_bank_index(addr),
        .end = tpu_bank_index(addr + size - 1)
    };
    return res;
}

int tpu_get_local_cstride(int h, int w, bool align, data_type_t dtype){
  return align ? ALIGN(h * w, tpu_eu_num(dtype)) : (h * w);
}

int tpu_get_local_nstride(int c_stride, int c, int start_idx){
    return DIV_UP(c + start_idx, NPU_NUM) * c_stride;
}

int tpu_get_local_size(const dim4* shape, data_type_t dtype, int start_idx, bool align){
    const int cstride = tpu_get_local_cstride(shape->h, shape->w, align, dtype);
    const int nstride = tpu_get_local_nstride(cstride, shape->c, start_idx);
    const long int size = (long int)shape->n * nstride * tpu_data_type_size(dtype);
    ASSERT(size < INT32_MAX);
    return size;
}

bool tpu_range_overlapped(const range_t *r0, const range_t *r1) {
    return !(r0->end < r1->start || r0->start > r1->end);
}
bool tpu_any_range_overlapped(const range_t *ranges, int num) {
    TPUKERNEL_ASSERT(num >= 2);
    for (int i = 0; i < num; ++i)
        for (int j = i + 1; j < num; ++j)
            if (tpu_range_overlapped(ranges + i, ranges + j))
                return true;
    return false;
}
int tpu_eu_num(data_type_t dtype) {
    if (dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32 || dtype == DT_TF32)
        return EU_NUM_32BIT;
    else if (dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_INT16 ||
             dtype == DT_UINT16)
        return EU_NUM_16BIT;
    else if (dtype == DT_INT8 || dtype == DT_UINT8
             || dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2)
        return EU_NUM_8BIT;
    else if (dtype == DT_INT4 || dtype == DT_UINT4)
        return EU_NUM_4BIT;
    else
        TPUKERNEL_ASSERT(0);
    return 0;
}
int tpu_bank_num() {
    return LOCAL_MEM_BANKS;
}

int tpu_gdma_shape_limit(int dim){
    const int max_shape[]= {GDMA_MAX_N, GDMA_MAX_C, GDMA_MAX_H, GDMA_MAX_W};
    return (dim>=0 && dim<4)? max_shape[dim]:0;
}

int tpu_gdma_move_max_wstride_byte_len() {
    return MAX_WSTRIDE_BYTE_LEN;
}

system_addr_t tpu_global_mem_real_addr(system_addr_t addr) {
    const uint32_t tag = (addr >> MAX_GMEM_BIT) & TAG_MASK;
    const uint64_t offset = addr & (MAX_GMEM_SIZE - 1);
    const reg_id_t reg_id = GDMA_ID_CFG_BASE_DDR0;
    uint64_t tag_addr = (tag + (reg_id.where / 32)) * 4;
    u32 base_addr = READ_REG(GDMA_ENGINE_MAIN_CTRL + tag_addr);
    uint64_t _base_addr = (uint64_t)base_addr << 8;
    uint64_t fixed_addr = _base_addr + offset;
    return fixed_addr;
}

void *tpu_global_mem_addr(global_addr_t addr) {
#ifdef USING_FAKE_DDR_MODE
    addr += PLD_BASE_ADDR;
#endif
    system_addr_t fixed_addr = tpu_global_mem_real_addr(addr);
    if((fixed_addr >= L2_SRAM_START_ADDR) && (fixed_addr < L2_SRAM_START_ADDR + L2_SRAM_SIZE)) {
        return GET_L2_SRAM_ADDR(fixed_addr);
    }
    return GET_GLOBAL_ADDR(fixed_addr);
}
void *tpu_local_mem_addr(int start_idx, local_addr_t addr) {
    TPUKERNEL_ASSERT(start_idx >= 0 && start_idx < NPU_NUM);
    TPUKERNEL_ASSERT(addr < LOCAL_MEM_SIZE);
    return GET_LOCAL_ADDR(start_idx, addr);
}
void *tpu_local_mem_addr_unified(local_addr_t addr) {
    TPUKERNEL_ASSERT(addr < NPU_NUM * LOCAL_MEM_SIZE);
    int start_idx = tpu_npu_index(addr);
    addr = addr % LOCAL_MEM_SIZE;
    return GET_LOCAL_ADDR(start_idx, addr);
}
void *tpu_l2_sram_addr(l2_sram_addr_t addr) {
    const uint32_t tag = (addr >> MAX_GMEM_BIT) & TAG_MASK;
    const uint64_t offset = addr & (MAX_GMEM_SIZE - 1);
    const reg_id_t reg_id = GDMA_ID_CFG_BASE_DDR0;
    uint64_t tag_addr = (tag + (reg_id.where / 32)) * 4;
    u32 base_addr = READ_REG(GDMA_ENGINE_MAIN_CTRL + tag_addr);
    uint64_t _base_addr = (uint64_t)base_addr << 8;
    uint64_t fixed_addr = _base_addr + offset;
    TPUKERNEL_ASSERT((fixed_addr >= L2_SRAM_START_ADDR) && (fixed_addr < L2_SRAM_START_ADDR + L2_SRAM_SIZE));
    return GET_L2_SRAM_ADDR(fixed_addr);
}
int tpu_bank_index(local_addr_t addr) {
    return (addr % LOCAL_MEM_SIZE) / LOCAL_BANK_SIZE;
}
void tpu_set_id_node(void *node) {
    id_node = *(CMD_ID_NODE *)node;
}
void tpu_get_id_node(void *node) {
    *(CMD_ID_NODE *)node = id_node;
}

void tpu_set_parallel_id_node(void *bd_node, void* gdma_node) {
    CMD_ID_NODE* bd_ptr = (CMD_ID_NODE*)bd_node;
    CMD_ID_NODE* gdma_ptr = (CMD_ID_NODE*)gdma_node;
    id_node.in_parallel_state = true;
    if(bd_ptr){
        TPUKERNEL_ASSERT(bd_ptr->in_parallel_state);
        bdc_id_node = *bd_ptr;
    }
    if(gdma_ptr){
        TPUKERNEL_ASSERT(gdma_ptr->in_parallel_state);
        gdma_id_node = *gdma_ptr;
    }
    if(bd_ptr&& gdma_ptr) {
        TPUKERNEL_ASSERT(bd_ptr->bd_cmd_id>=gdma_ptr->bd_cmd_id && gdma_ptr->gdma_cmd_id>=bd_ptr->gdma_cmd_id);
    }
}

void tpu_get_parallel_id_node(void *bd_node, void* gdma_node) {
    CMD_ID_NODE* bd_ptr = (CMD_ID_NODE*)bd_node;
    CMD_ID_NODE* gdma_ptr = (CMD_ID_NODE*)gdma_node;
    if(bd_ptr){
        *bd_ptr = bdc_id_node;
    }
    if(gdma_ptr){
        *gdma_ptr = gdma_id_node;
    }
}

void tpu_enable_check_id_node() {
    check_id_node = true;
}
void tpu_disable_check_id_node() {
    check_id_node = false;
}

#ifdef USING_CMODEL
THREAD int override_cmdid = 0;

void tpu_set_override_cmdid(int v)
{
    override_cmdid = v;
}

void tpu_reset_cmd_id()
{
    memset(&id_node, 0, sizeof(id_node));
}
#endif

__attribute__((weak)) int has_defered_c2c();
__attribute__((weak)) void execute_defered_c2c();

THREAD int no_recursive_polls_please;

void tpu_initialize(void) {
#ifdef USING_CMODEL
    if (override_cmdid) return;
#endif

    int is_recursive = no_recursive_polls_please;
    if (!is_recursive)
    {
        no_recursive_polls_please = 1;
        tpu_poll_descriptor();
    }

#ifdef REMOVE_POLLS_IN_LLM

    if (!is_recursive)
    {
        tpu_poll();
    }

#elif !USING_APTP_PARALLEL
    resync_cmd_id(&id_node);
#endif

    if (!is_recursive)
        no_recursive_polls_please = 0;
}
RTM_EXPORT(tpu_initialize);

void tpu_nop()
{
    atomic_bd_nop_gen_cmd(&id_node);
    atomic_gdma_nop_gen_cmd(&id_node, MASTER_THREAD);
    atomic_sdma_nop_gen_cmd(&id_node, DEFAULT_SDMA_PORT);
}

int tpu_cdma_get_port(int self, int peer, int direction) {
#ifdef USE_DEBUG_CDMA_PORT
    return DEBUG_CDMA_PORT;
#else
    return get_c2c_port(self, peer, direction);
#endif
}

void tpu_cdma_all_port_nop() {
    int used_cdma_port = tpu_cdma_get_used_port_num();
    if(used_cdma_port < 1) return;
    int *cdma_ports = tpu_cdma_get_used_ports();
    for (int i = 0; i < used_cdma_port; i++) {
        int port = cdma_ports[i];
        if (port != -1) {
            tpu_cdma_nop(port);
        }
    }
}

void tpu_cdma_initialize() {
#ifdef USING_CMODEL
    if(override_cmdid) return;
#endif
    tpu_poll_descriptor();

#ifdef REMOVE_POLLS_IN_LLM
    tpu_poll();
#endif
#ifdef USE_DEBUG_CDMA_PORT
    tpu_cdma_port_initialize(DEBUG_CDMA_PORT);
    tpu_cdma_nop(DEBUG_CDMA_PORT);
#else
    int used_port_num = tpu_cdma_get_used_port_num();
    int *cdma_ports = tpu_cdma_get_used_ports();
    for (int i = 0; i < used_port_num; i++) {
        int port = cdma_ports[i];
        if (port != -1) {
            tpu_cdma_port_initialize(port);
            tpu_cdma_nop(port);
        }
    }
#endif
}

void tpu_enable_pmu(){
  enable_tpu_perf_monitor();
}

void tpu_disable_pmu(){
  disable_tpu_perf_monitor();
}

void tpu_cdma_port_initialize(int port) {
    TPUKERNEL_ASSERT(port >= 0 && port < MAX_CDMA_NUM);
    resync_cdma_port_subsys_cmd_id(&id_node, port);
}

void tpu_vsdma_port_initialize(int port) {
    TPUKERNEL_ASSERT(port >= 0 && port < MAX_CDMA_NUM);
    resync_vsdma_port_subsys_cmd_id(&id_node, port);
}

void tpu_vsdma_initialize() {
#ifdef USING_CMODEL
    if(override_cmdid) return;
#endif

#if defined(C2C_USE_DESCRIPTOR)
    tpu_initialize();
#else

    for (int i = 0; i < 4; i++) {
        tpu_vsdma_port_initialize(i);
    }
#endif
}

int tpu_bdc_busy() {
    return check_engine_busy_internal(BDC_NODE, ENGINE_BD);
}

int tpu_gdma_busy() {
    return check_engine_busy_internal(GDMA_NODE, ENGINE_GDMA);
}

int tpu_sdma_busy() {
    return check_engine_busy_internal(&id_node, ENGINE_SDMA);
}

int tpu_hau_sort_busy() {
    return check_engine_busy_internal(&id_node, ENGINE_HAU);
}

int tpu_gdma_cmd_overflow() {
    // GDMA_CMD_ID_BITS_WIDTH is 20
    return (GDMA_NODE->gdma_cmd_id > (1<<20));
}

int tpu_sdma_cmd_overflow() {
  return (id_node.sdma_cmd_id > (1<<20));
}

int tpu_hau_sort_cmd_overflow() {
    // A temporary limit for PowerStress test, may no limit
    return (id_node.hau_cmd_id > (1<<20));
}

void tpu_poll() {
#ifdef USING_CMODEL
    if (override_cmdid) return;
#endif

    if (!no_recursive_polls_please)
    {
        no_recursive_polls_please = 1;
        tpu_poll_descriptor();
        no_recursive_polls_please = 0;
    }

#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "tpu_poll GDMA %d, TIU %d, SDMA %d...\n",
        id_node.gdma_cmd_id,
        id_node.bd_cmd_id,
        id_node.sdma_cmd_id);
#endif

#if !USING_APTP_PARALLEL
    TPUKERNEL_ASSERT(!id_node.in_parallel_state);

    if (id_node.gdma_cmd_id ||
        id_node.bd_cmd_id ||
        id_node.sdma_cmd_id ||
        id_node.hau_cmd_id)
    {
        poll_all_engine_done(&id_node);
#ifdef DUMP_CACHED_CMD
        TP_DEBUG(
            "tpu_poll done GDMA %d, TIU %d, SDMA %d.\n",
            id_node.gdma_cmd_id,
            id_node.bd_cmd_id,
            id_node.sdma_cmd_id);
#endif
    }

    resync_cmd_id(&id_node);
#endif
}
RTM_EXPORT(tpu_poll);

void __attribute__((weak)) poll_cur_task_enable(void);
void __attribute__((weak)) write_response(int result);
void tpu_poll_empty() {
    TPUKERNEL_ASSERT(!id_node.in_parallel_state);
    poll_all_engine_done(&id_node);
#if USING_APTP_PARALLEL
    write_response(0);
#endif
}

void tpu_hau_poll() {
    poll_hau_engine_done(&id_node);
}

bool tpu_is_cdma_busy()
{
    if (has_defered_c2c())
        return true;

    if (id_node.cdma_cmd_id[DEBUG_CDMA_PORT])
        return true;

    int *cdma_ports = tpu_cdma_get_used_ports();
    if (!cdma_ports)
        return false;

    for (int i = 0; i < MAX_C2C_CDMA_NUM; i++)
    {
        if (id_node.cdma_cmd_id[i])
            return true;
    }

    return false;
}

THREAD int cdma_defered_msg_id;

void tpu_set_cdma_defered_barrier_msg_id(int msg_id)
{
    cdma_defered_msg_id = msg_id;
}

void tpu_cdma_poll() {
    int is_recursive = no_recursive_polls_please;
    if (!is_recursive && has_defered_c2c())
    {
        no_recursive_polls_please = 1;
#ifdef DUMP_CACHED_CMD
        TP_DEBUG("tpu_initialize execute defered c2c\n");
#endif
        execute_defered_c2c();
    }

#ifdef USE_DEBUG_CDMA_PORT
    if (id_node.cdma_cmd_id[DEBUG_CDMA_PORT])
    {
#ifdef DUMP_CACHED_CMD
        TP_DEBUG("tpu_cdma_poll debug port %d, cmdid %d...\n",
            DEBUG_CDMA_PORT, id_node.cdma_cmd_id[DEBUG_CDMA_PORT]);
#endif
        tpu_cdma_nop(DEBUG_CDMA_PORT);
        tpu_cdma_port_poll(DEBUG_CDMA_PORT);
        id_node.cdma_cmd_id[DEBUG_CDMA_PORT] = 0;
#ifdef DUMP_CACHED_CMD
        TP_DEBUG("tpu_cdma_poll debug port %d done, cmdid %d.\n",
            DEBUG_CDMA_PORT, id_node.cdma_cmd_id[DEBUG_CDMA_PORT]);
#endif
    }

#else // USE_DEBUG_CDMA_PORT

    int *cdma_ports = tpu_cdma_get_used_ports();
    int used_port_num = tpu_cdma_get_used_port_num();
    if (!cdma_ports)
        goto end;
    for (int i = 0; i < used_port_num; i++) {
        int port = cdma_ports[i];
        if (port < 0) continue;
#ifdef DUMP_CACHED_CMD
        TP_DEBUG(
            "tpu_cdma_poll port %d, cmdid %d...\n",
            port, id_node.cdma_cmd_id[port]);
#endif
        if (id_node.cdma_cmd_id[port]) {
            tpu_cdma_nop(port);
            tpu_cdma_port_poll(port);
        }
#ifdef DUMP_CACHED_CMD
        TP_DEBUG(
            "tpu_cdma_poll port %d done, cmdid %d.\n",
            port, id_node.cdma_cmd_id[port]);
#endif
    }

end:

#endif // USE_DEBUG_CDMA_PORT

    if (!is_recursive)
        no_recursive_polls_please = 0;
}

void tpu_cdma_perf_poll(int *chips, int *ports, int* actions, u64 *info_addr) {
#if DUMP_CACHED_CMD
    TP_DEBUG(
        "[begin] tpu_cdma_poll ports[0] %d, cmdid0 %d, ports[1] %d, cmdid1 %d,...\n",
        ports[0], id_node.cdma_cmd_id[ports[0]], ports[1], id_node.cdma_cmd_id[ports[1]]);
#endif

    if (id_node.cdma_cmd_id[ports[0]] && id_node.cdma_cmd_id[ports[1]]) {
        tpu_cdma_nop(ports[0]);
        tpu_cdma_nop(ports[1]);
        tpu_cdma_perf_port_poll(chips, ports, actions, info_addr);
        id_node.cdma_cmd_id[ports[0]] = 0;
        id_node.cdma_cmd_id[ports[1]] = 0;
    }

#if DUMP_CACHED_CMD
    TP_DEBUG(
        "[done] tpu_cdma_poll ports[0] %d, cmdid0 %d, ports[1] %d, cmdid1 %d,...\n",
        ports[0], id_node.cdma_cmd_id[ports[0]], ports[1], id_node.cdma_cmd_id[ports[1]]);
#endif
}

void tpu_cdma_port_poll(int port) {
    TPUKERNEL_ASSERT(port >= 0 && port < MAX_CDMA_NUM);
    if(id_node.cdma_cmd_id[port] == 0)
        return;
    poll_cdma_engine_done(port, &id_node);
    id_node.cdma_cmd_id[port] = 0;
}

void tpu_cdma_perf_port_poll(int *chips, int *ports, int* actions, u64 *info_addr) {
    TPUKERNEL_ASSERT(ports[0] >= 0 && ports[0] < MAX_CDMA_NUM);
    TPUKERNEL_ASSERT(ports[1] >= 0 && ports[1] < MAX_CDMA_NUM);
    if(id_node.cdma_cmd_id[ports[0]] == 0 || id_node.cdma_cmd_id[ports[1]] == 0)
        return;
    poll_cdma_perf_engine_done(chips, ports, actions, info_addr, &id_node);
}
void tpu_vsdma_port_poll(int port) {
    TPUKERNEL_ASSERT(port >= 0 && port < MAX_TPU_CORE_NUM);
    if (id_node.vsdma_cmd_id[port] == 0)
       return;
    poll_vsdma_engine_done(port, &id_node);
}

void tpu_vsdma_poll() {
#if defined(C2C_USE_DESCRIPTOR)
    tpu_poll();
#else
    for (int i = 0; i < 4; i++) {
        tpu_vsdma_port_poll(i);
    }
#endif
}

void tpu_parallel_start() {
    TPUKERNEL_ASSERT(!id_node.in_parallel_state);
    cmd_id_divide(&id_node, &bdc_id_node, &gdma_id_node);
}
RTM_EXPORT(tpu_parallel_start);

void tpu_parallel_end() {
    TPUKERNEL_ASSERT(id_node.in_parallel_state);
#ifdef USING_SGDNN_BOTH_MESG_TEST
    tpu_sync_core();
#endif
    cmd_id_merge(&id_node, &bdc_id_node, &gdma_id_node);

    if (!check_id_node)
        return;
    if (id_node.gdma_cmd_id > CMD_ID_OVERFLOW_VALUE ||
            id_node.bd_cmd_id > CMD_ID_OVERFLOW_VALUE) {
        poll_all_engine_done(&id_node);
        resync_cmd_id(&id_node);
    }
}
RTM_EXPORT(tpu_parallel_end);
bool tpu_is_parallel_state() {
    return id_node.in_parallel_state;
}
void tpu_sync_start() {
    id_node.in_sync_state = bdc_id_node.in_sync_state
                          = gdma_id_node.in_sync_state
                          = true;
}
void tpu_sync_end() {
    id_node.in_sync_state = bdc_id_node.in_sync_state
                          = gdma_id_node.in_sync_state
                          = false;
}
bool tpu_is_sync_state() {
    return id_node.in_sync_state;
}
bool tpu_is_data_type_signed(data_type_t dtype) {
    return SIGN(dtype);
}

void tpu_bdc_set_base_msg_id(int base_msg_id) {
    atomic_set_base_msg_id(base_msg_id, ENGINE_BD);
}
void tpu_gdma_set_base_msg_id(int base_msg_id) {
    atomic_set_base_msg_id(base_msg_id, ENGINE_GDMA);
}
void tpu_hau_set_base_msg_id(int base_msg_id) {
    atomic_set_base_msg_id(base_msg_id, ENGINE_HAU);
}

void tpu_sdma_set_base_msg_id(int base_msg_id) {
    atomic_set_base_msg_id(base_msg_id, ENGINE_SDMA);
}
void tpu_cdma_set_base_msg_id(int base_msg_id) {
    atomic_set_base_msg_id(base_msg_id, ENGINE_CDMA);
}

int tpu_get_dma_dtype(data_type_t dtype) {
    switch (dtype)
    {
    case DT_INT8:
    case DT_UINT8:
        return GDMA_INT8;
    case DT_INT16:
    case DT_UINT16:
        return GDMA_INT16;
    case DT_FP16:
        return GDMA_FP16;
    case DT_BFP16:
        return GDMA_BF16;
    case DT_INT32:
    case DT_UINT32:
        return GDMA_INT32;
    case DT_FP8E4M3:
        return GDMA_FP8_E4M3;
    case DT_FP8E5M2:
        return GDMA_FP8_E5M2;
    case DT_FP32:
    case DT_TF32:
        return GDMA_FP32;
    case DT_FP20:
        return GDMA_FP20;
    default:
        ASSERT(0);
        return -1;
    }
}

void tpu_gdma_cpy_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    dim4 dst_stride_aligned, src_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_aligned_stride(&dst_stride_aligned, 0, shape, dtype);
        dst_stride_ptr = &dst_stride_aligned;
    }
    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_continuous, shape);
        src_stride_ptr = &src_stride_continuous;
    }
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 src_offset_n = (u64)nidx * src_stride_ptr->n * dsize;
        u64 src_offset_c = (u64)cidx * src_stride_ptr->c * dsize;
        u64 src_offset_h = (u64)hidx * src_stride_ptr->h * dsize;
        u64 src_offset_w = (u64)widx * src_stride_ptr->w * dsize;
        u64 src_offset = src_offset_n + src_offset_c + src_offset_h + src_offset_w;
        int dst_offset_n = nidx * dst_stride_ptr->n * dsize;
        int dst_offset_c = tpu_unified_c_offset(cidx, dst_stride_ptr->c, dtype);
        int dst_offset_h = hidx * dst_stride_ptr->h * dsize;
        int dst_offset_w = widx * dst_stride_ptr->w * dsize;
        int dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        tensor_stride_move_gen_cmd(
            tpu_npu_addr(dst_addr + dst_offset),
            tpu_npu_index(dst_addr + dst_offset),
            src_addr + src_offset,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            GDMA_S2L,
            false,
            GDMA_NODE);
        CHECK_GDMA_OVERFLOW;
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}
RTM_EXPORT(tpu_gdma_cpy_S2L);
void tpu_gdma_cpy_nc_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        tensor_align_move_gen_cmd(
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            src_addr,
            dst_shape->c,
            dst_shape->n,
            dst_shape->h,
            dst_shape->w,
            tpu_get_dma_dtype(dtype),
            GDMA_S2L,
            true,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(dst_addr),
                dst_shape,
                dtype);
            dst_stride_ptr = &local_stride;
        }
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->c, .c = dst_shape->n,
                .h = dst_shape->h, .w = dst_shape->w
            };
            tpu_continuous_stride(&global_stride, &shape);
            src_stride_ptr = &global_stride;
        }
        tensor_stride_move_gen_cmd(
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            src_addr,
            dst_shape->c,
            dst_shape->n,
            dst_shape->h,
            dst_shape->w,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            GDMA_S2L,
            true,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
RTM_EXPORT(tpu_gdma_cpy_nc_trans_S2L);
void tpu_gdma_cpy_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (src_stride == NULL) {
        tpu_aligned_stride(&src_stride_aligned, 0, shape, dtype);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    tpu_mem_check_tensor(dst_addr, shape, dst_stride_ptr, dtype);
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        int src_offset_n = nidx * src_stride_ptr->n * dsize;
        int src_offset_c = tpu_unified_c_offset(cidx, src_stride_ptr->c, dtype);
        int src_offset_h = hidx * src_stride_ptr->h * dsize;
        int src_offset_w = widx * src_stride_ptr->w * dsize;
        int src_offset = src_offset_n + src_offset_c + src_offset_h + src_offset_w;
        tensor_stride_move_gen_cmd(
            tpu_npu_addr(src_addr + src_offset),
            tpu_npu_index(src_addr + src_offset),
            dst_addr + dst_offset,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            GDMA_L2S,
            false,
            GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}
RTM_EXPORT(tpu_gdma_cpy_L2S);
void tpu_gdma_cpy_nc_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    tpu_mem_check_tensor(dst_addr, dst_shape, dst_stride, dtype);
    if (dst_stride == NULL && src_stride == NULL)
        tensor_align_move_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr,
            dst_shape->c,
            dst_shape->n,
            dst_shape->h,
            dst_shape->w,
            tpu_get_dma_dtype(dtype),
            GDMA_L2S,
            true,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_continuous_stride(&global_stride, dst_shape);
            dst_stride_ptr = &global_stride;
        }
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->c, .c = dst_shape->n,
                .h = dst_shape->h, .w = dst_shape->w
            };
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(src_addr),
                &shape,
                dtype);
            src_stride_ptr = &local_stride;
        }
        tensor_stride_move_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr,
            dst_shape->c,
            dst_shape->n,
            dst_shape->h,
            dst_shape->w,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            GDMA_L2S,
            true,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
RTM_EXPORT(tpu_gdma_cpy_nc_trans_L2S);
void tpu_gdma_cpy_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t    dtype) {
    dim4 dst_stride_align, src_stride_align;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_aligned_stride(
            &dst_stride_align,
            tpu_npu_index(dst_addr),
            shape,
            dtype);
        dst_stride_ptr = &dst_stride_align;
    }
    if (src_stride == NULL) {
        tpu_aligned_stride(
            &src_stride_align,
            tpu_npu_index(src_addr),
            shape,
            dtype);
        src_stride_ptr = &src_stride_align;
    }
    tensor_general_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_L2L,
        false,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_cpy_nc_trans_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *dst_shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    dim4 dst_stride_align, src_stride_align;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_aligned_stride(
            &dst_stride_align,
            tpu_npu_index(dst_addr),
            dst_shape,
            dtype);
        dst_stride_ptr = &dst_stride_align;
    }
    if (src_stride == NULL) {
        dim4 shape = {
            .n = dst_shape->c, .c = dst_shape->n,
            .h = dst_shape->h, .w = dst_shape->w
        };
        tpu_aligned_stride(
            &src_stride_align,
            tpu_npu_index(src_addr),
            &shape,
            dtype);
        src_stride_ptr = &src_stride_align;
    }
    tensor_general_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        dst_shape->c,
        dst_shape->n,
        dst_shape->h,
        dst_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_L2L,
        true,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_cpy_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    dim4 dst_stride_continuous, src_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_continuous, shape);
        src_stride_ptr = &src_stride_continuous;
    }
    tpu_mem_check_tensor(dst_addr, shape, dst_stride_ptr, dtype);
    tpu_mem_check_tensor(src_addr, shape, src_stride_ptr, dtype);
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 src_offset_n = (u64)nidx * src_stride_ptr->n * dsize;
        u64 src_offset_c = (u64)cidx * src_stride_ptr->c * dsize;
        u64 src_offset_h = (u64)hidx * src_stride_ptr->h * dsize;
        u64 src_offset_w = (u64)widx * src_stride_ptr->w * dsize;
        u64 src_offset = src_offset_n + src_offset_c + src_offset_h + src_offset_w;
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        tensor_general_move_gen_cmd(
            src_addr + src_offset,
            NO_USE,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            dst_addr + dst_offset,
            NO_USE,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            GDMA_S2S,
            false,
            GDMA_NODE);
        CHECK_GDMA_OVERFLOW;
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}
void tpu_gdma_cpy_nc_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    dim4 dst_stride_continuous, src_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, dst_shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    if (src_stride == NULL) {
        dim4 shape = {
            .n = dst_shape->c, .c = dst_shape->n,
            .h = dst_shape->h, .w = dst_shape->w
        };
        tpu_continuous_stride(&src_stride_continuous, &shape);
        src_stride_ptr = &src_stride_continuous;
    }
    tpu_mem_check_tensor(dst_addr, dst_shape, dst_stride_ptr, dtype);
    tensor_general_move_gen_cmd(
        src_addr,
        NO_USE,
        dst_shape->c,
        dst_shape->n,
        dst_shape->h,
        dst_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr,
        NO_USE,
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_S2S,
        true,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_compact_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    dtype) {
    tensor_compact_move_gen_cmd(
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        src_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        tpu_get_dma_dtype(dtype),
        GDMA_S2L,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_compact_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    data_type_t    dtype) {
    tensor_compact_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        tpu_get_dma_dtype(dtype),
        GDMA_L2S,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_compact_nc_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    data_type_t    dtype) {
    tensor_compact_move_gen_cmd(
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        src_addr,
        dst_shape->c,
        dst_shape->n,
        dst_shape->h,
        dst_shape->w,
        tpu_get_dma_dtype(dtype),
        GDMA_S2L,
        true,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_compact_nc_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    data_type_t    dtype) {
    tensor_compact_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        dst_addr,
        dst_shape->c,
        dst_shape->n,
        dst_shape->h,
        dst_shape->w,
        tpu_get_dma_dtype(dtype),
        GDMA_L2S,
        true,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_cpy_cw_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        general_cwtrans_gen_cmd(
            src_addr,
            NO_USE,
            tpu_npu_addr(dst_addr),
            tpu_npu_index(dst_addr),
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            0, 0, 0, 0, 0, 0,
            false,
            GDMA_S2L,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(dst_addr),
                dst_shape,
                dtype);
            dst_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->n, .c = dst_shape->w,
                .h = dst_shape->h, .w = dst_shape->c
            };
            tpu_continuous_stride(&global_stride, &shape);
            src_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        general_cwtrans_gen_cmd(
            src_addr,
            NO_USE,
            tpu_npu_addr(dst_addr),
            tpu_npu_index(dst_addr),
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            true,
            GDMA_S2L,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_cpy_cw_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        general_cwtrans_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr,
            NO_USE,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            false,
            GDMA_L2S,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_continuous_stride(&global_stride, dst_shape);
            dst_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->n, .c = dst_shape->w,
                .h = dst_shape->h, .w = dst_shape->c
            };
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(src_addr),
                &shape,
                dtype);
            src_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        general_cwtrans_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr,
            NO_USE,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            true,
            GDMA_L2S,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_cpy_cw_trans_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *dst_shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        general_cwtrans_gen_cmd(
            tpu_npu_addr(src_addr),
            tpu_npu_index(src_addr),
            tpu_npu_addr(dst_addr),
            tpu_npu_index(dst_addr),
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            0, 0, 0, 0, 0, 0,
            false,
            GDMA_L2L,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 dst_stride_align, src_stride_align;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_aligned_stride(
                &dst_stride_align,
                tpu_npu_index(dst_addr),
                dst_shape,
                dtype);
            dst_stride_ptr = &dst_stride_align;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->n, .c = dst_shape->w,
                .h = dst_shape->h, .w = dst_shape->c
            };
            tpu_aligned_stride(
                &src_stride_align,
                tpu_npu_index(src_addr),
                &shape,
                dtype);
            src_stride_ptr = &src_stride_align;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        general_cwtrans_gen_cmd(
            tpu_npu_addr(src_addr),
            tpu_npu_index(src_addr),
            tpu_npu_addr(dst_addr),
            tpu_npu_index(dst_addr),
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            true,
            GDMA_L2L,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_cpy_cw_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        general_cwtrans_gen_cmd(
            src_addr,
            NO_USE,
            dst_addr,
            NO_USE,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            false,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 dst_stride_continuous, src_stride_continuous;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_continuous_stride(&dst_stride_continuous, dst_shape);
            dst_stride_ptr = &dst_stride_continuous;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->n, .c = dst_shape->w,
                .h = dst_shape->h, .w = dst_shape->c
            };
            tpu_continuous_stride(&src_stride_continuous, &shape);
            src_stride_ptr = &src_stride_continuous;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        general_cwtrans_gen_cmd(
            src_addr,
            NO_USE,
            dst_addr,
            NO_USE,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            true,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}

unsigned int tpu_gdma_get_filter_num() {
  unsigned int filter_res_num = get_gdma_filter_res_num_gen_cmd(GDMA_NODE);
  return filter_res_num;
}

void tpu_gdma_mask_select_L2S(
    global_addr_t  dst_addr,
    local_addr_t   src_addr,
    addr_t         mask_addr,
    int            mask_in_lmem, // 1: mask is in local_mem, 0: mask is in global_mem
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype
 ) {
    tensor_general_move_with_mask_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        mask_in_lmem ? (mask_addr % LOCAL_MEM_SIZE) : mask_addr,
        mask_in_lmem ? tpu_npu_index(mask_addr) : NO_USE,
        mask_in_lmem,
        dst_addr,
        tpu_get_dma_dtype(data_dtype),
        tpu_get_dma_dtype(mask_dtype),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        GDMA_L2S,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_mask_select_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    addr_t         mask_addr,
    int            mask_in_lmem, // 1: mask is in local_mem, 0: mask is in global_mem
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype
 ) {
    tensor_general_move_with_mask_gen_cmd(
        src_addr,
        NO_USE,
        mask_in_lmem ? (mask_addr % LOCAL_MEM_SIZE) : mask_addr,
        mask_in_lmem ? tpu_npu_index(mask_addr) : NO_USE,
        mask_in_lmem,
        dst_addr,
        tpu_get_dma_dtype(data_dtype),
        tpu_get_dma_dtype(mask_dtype),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        GDMA_S2S,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

unsigned int tpu_gdma_mask_select_S2S_with_ret(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    addr_t         mask_addr,
    int            mask_in_lmem,
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype
 ) {
    tensor_general_move_with_mask_gen_cmd(
        src_addr,
        NO_USE,
        mask_in_lmem ? (mask_addr % LOCAL_MEM_SIZE) : mask_addr,
        mask_in_lmem ? tpu_npu_index(mask_addr) : NO_USE,
        mask_in_lmem,
        dst_addr,
        tpu_get_dma_dtype(data_dtype),
        tpu_get_dma_dtype(mask_dtype),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        GDMA_S2S,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
    return tpu_gdma_get_filter_num();
}

void tpu_gdma_nonzero_L2S(
    global_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx) {
  tensor_move_nonzero_gen_cmd(
    src_addr % LOCAL_MEM_SIZE,
    tpu_npu_index(src_addr),
    dst_addr,
    tpu_get_dma_dtype(data_dtype),
    GDMA_INT32,
    shape->n,
    shape->c,
    shape->h,
    shape->w,
    base_idx,
    GDMA_L2S,
    MASTER_THREAD,
    GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_nonzero_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx) {
  tensor_move_nonzero_gen_cmd(
    src_addr,
    0,
    dst_addr,
    tpu_get_dma_dtype(data_dtype),
    GDMA_INT32,
    shape->n,
    shape->c,
    shape->h,
    shape->w,
    base_idx,
    GDMA_S2S,
    MASTER_THREAD,
    GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_set_C_system(
    system_addr_t  dst_addr,
    scalar_t       C,
    const dim4    *shape,
    const dim4    *dst_stride,
    data_type_t    dtype) {
    dim4 dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        fill_constant_gen_global_cmd_stride(
            dst_addr + dst_offset,
            &C,
            tpu_get_dma_dtype(dtype),
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            true,
            MASTER_THREAD,
            GDMA_NODE);
        CHECK_GDMA_OVERFLOW;
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}
void tpu_gdma_set_C_local(
    local_addr_t  dst_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    data_type_t   dtype) {
    if (dst_stride == NULL) {
        dim4 stride;
        tpu_aligned_stride(
            &stride,
            0,
            shape,
            dtype);
        fill_constant_gen_local_cmd_stride(
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            &C,
            tpu_get_dma_dtype(dtype),
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            stride.n,
            stride.c,
            stride.h,
            stride.w,
            true,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    } else
        fill_constant_gen_local_cmd_stride(
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            &C,
            tpu_get_dma_dtype(dtype),
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            dst_stride->n,
            dst_stride->c,
            dst_stride->h,
            dst_stride->w,
            true,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_matrix_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            rows,
    int            cols,
    int            cols_per_channel,
    int            row_stride,
    data_type_t    dtype) {
    tpu_mem_check_matrix(src_addr, rows, cols, row_stride, dtype);
    general_matrix_move_gen_cmd(
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        src_addr,
        cols_per_channel,
        rows,
        cols,
        row_stride,
        tpu_get_dma_dtype(dtype),
        GDMA_S2L,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_matrix_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            rows,
    int            cols,
    int            cols_per_channel,
    int            row_stride,
    data_type_t    dtype) {
    tpu_mem_check_matrix(dst_addr, rows, cols, row_stride, dtype);
    general_matrix_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        dst_addr,
        cols_per_channel,
        rows,
        cols,
        row_stride,
        tpu_get_dma_dtype(dtype),
        GDMA_L2S,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_matrix_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            src_rows,
    int            src_cols,
    int            dst_cols_per_channel,
    int            src_row_stride,
    data_type_t    dtype) {
    tpu_mem_check_matrix(src_addr, src_rows, src_cols, src_row_stride, dtype);
    general_matrix_move_gen_cmd(
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        src_addr,
        dst_cols_per_channel,
        src_rows,
        src_cols,
        src_row_stride,
        tpu_get_dma_dtype(dtype),
        GDMA_S2L,
        true,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_matrix_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            src_rows,
    int            src_cols,
    int            src_cols_per_channel,
    int            dst_row_stride,
    data_type_t    dtype) {
    tpu_mem_check_matrix(dst_addr, src_cols, src_rows, dst_row_stride, dtype);
    general_matrix_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        dst_addr,
        src_cols_per_channel,
        src_cols,
        src_rows,
        dst_row_stride,
        tpu_get_dma_dtype(dtype),
        GDMA_L2S,
        true,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_vector_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            len,
    int            len_per_channel,
    data_type_t    dtype) {
    tpu_gdma_matrix_S2L(
        dst_addr,
        src_addr,
        1,
        len,
        len_per_channel,
        NO_USE,
        dtype);
}
void tpu_gdma_vector_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            len,
    int            len_per_channel,
    data_type_t    dtype) {
    tpu_gdma_matrix_L2S(
        dst_addr,
        src_addr,
        1,
        len,
        len_per_channel,
        NO_USE,
        dtype);
}
void tpu_gdma_channel_bcast_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        tensor_broadcast_move_gen_cmd(
            src_addr,
            NO_USE,
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            shape->n,
            shape->h,
            shape->w,
            shape->c,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            GDMA_S2L,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(dst_addr),
                shape,
                dtype);
            dst_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 src_shape = {
                .n = shape->n, .c = 1, .h = shape->h, .w = shape->w
            };
            tpu_continuous_stride(&global_stride, &src_shape);
            src_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        tensor_broadcast_move_gen_cmd(
            src_addr,
            NO_USE,
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            shape->n,
            shape->h,
            shape->w,
            shape->c,
            src_stride_ptr->n,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            true,
            GDMA_S2L,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_channel_bcast_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    if (dst_stride == NULL && src_stride == NULL)
        tensor_broadcast_move_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            shape->n,
            shape->h,
            shape->w,
            shape->c,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            GDMA_L2L,
            GDMA_NODE);
    else {
        dim4 dst_stride_align, src_stride_align;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_aligned_stride(
                &dst_stride_align,
                tpu_npu_index(dst_addr),
                shape,
                dtype);
            dst_stride_ptr = &dst_stride_align;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 src_shape = {
                .n = shape->n, .c = 1, .h = shape->h, .w = shape->w
            };
            tpu_aligned_stride(
                &src_stride_align,
                tpu_npu_index(src_addr),
                &src_shape,
                dtype);
            src_stride_ptr = &src_stride_align;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        tensor_broadcast_move_gen_cmd(
            src_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(src_addr),
            dst_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(dst_addr),
            shape->n,
            shape->h,
            shape->w,
            shape->c,
            src_stride_ptr->n,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            true,
            GDMA_L2L,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_gather_S2L(
    local_addr_t   output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2L,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(output_addr),
                shape,
                dtype);
            output_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&global_stride, &param_shape);
            param_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2L,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_gather_L2S(
    system_addr_t  output_addr,
    local_addr_t   param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_gather_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_L2S,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&global_stride, shape);
            output_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(param_addr),
                &param_shape,
                dtype);
            param_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_gather_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, // start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_L2S,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_gather_L2L(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    addr_t        index_addr,
    bool          index_is_local,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    const dim4   *output_stride,
    const dim4   *param_stride,
    const dim4   *index_stride,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_gather_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_L2L,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_align, param_stride_align, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_aligned_stride(
                &output_stride_align,
                tpu_npu_index(output_addr),
                shape,
                dtype);
            output_stride_ptr = &output_stride_align;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_aligned_stride(
                &param_stride_align,
                tpu_npu_index(param_addr),
                &param_shape,
                dtype);
            param_stride_ptr = &param_stride_align;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_gather_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_L2L,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_gather_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_gather_S2S_ext(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    u32            start_pos,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            start_pos, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_gather_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            start_pos, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2S,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_scatter_S2L(
    local_addr_t   output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2L,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(output_addr),
                shape,
                dtype);
            output_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&global_stride, &param_shape);
            param_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2L,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_scatter_L2S(
    system_addr_t  output_addr,
    local_addr_t   param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_scatter_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_L2S,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 local_stride, global_stride, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&global_stride, shape);
            output_stride_ptr = &global_stride;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_aligned_stride(
                &local_stride,
                tpu_npu_index(param_addr),
                &param_shape,
                dtype);
            param_stride_ptr = &local_stride;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_scatter_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_L2S,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_scatter_L2L(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    addr_t        index_addr,
    bool          index_is_local,
    const dim4   *shape,
    int           param_h,
    const dim4   *output_stride,
    const dim4   *param_stride,
    const dim4   *index_stride,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_scatter_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_L2L,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_align, param_stride_align, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_aligned_stride(
                &output_stride_align,
                tpu_npu_index(output_addr),
                shape,
                dtype);
            output_stride_ptr = &output_stride_align;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_aligned_stride(
                &param_stride_align,
                tpu_npu_index(param_addr),
                &param_shape,
                dtype);
            param_stride_ptr = &param_stride_align;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_scatter_gen_cmd(
            param_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(param_addr),
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr % LOCAL_MEM_SIZE,
            tpu_npu_index(output_addr),
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_L2L,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_scatter_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2S,
            false,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2S,
            NO_USE,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_h_scatter_S2S_ext(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    u32            start_pos,
    int            inplace_add,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            start_pos, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            GDMA_S2S,
            inplace_add ? true : false,
            MASTER_THREAD,
            GDMA_NODE);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = 1
            };
            if (index_is_local)
                tpu_aligned_stride(
                    &idx_stride,
                    tpu_npu_index(index_addr),
                    &index_shape,
                    DT_UINT32);
            else
                tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        tensor_gdma_scatter_gen_cmd(
            param_addr,
            NO_USE,
            index_is_local ? (index_addr % LOCAL_MEM_SIZE) : index_addr,
            index_is_local ? tpu_npu_index(index_addr) : NO_USE,
            index_is_local,
            output_addr,
            NO_USE,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            start_pos, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            GDMA_S2S,
            inplace_add ? true : false,
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_system_cpy(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    unsigned int   count,
    data_type_t    dtype) {
    general_gdma_gen_cmd(
        src_addr,
        dst_addr,
        tpu_get_dma_dtype(dtype),
        count,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_system_bcast(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    unsigned int   count,
    int            dst_channel,
    data_type_t    dtype) {
    general_broadcast_gen_cmd(
        src_addr,
        dst_addr,
        0,
        tpu_get_dma_dtype(dtype),
        count,
        dst_channel,
        false,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_system_set(
    system_addr_t  dst_addr,
    scalar_t       C,
    unsigned int   count,
    data_type_t    dtype) {
    general_gdma_gen_cmd(
        C.u32,
        dst_addr,
        tpu_get_dma_dtype(dtype),
        count,
        true,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
#define TPU_BDC_CMP_SELECT(name, op)                                           \
void tpu_bdc_##name##_select(                                                  \
    local_addr_t       dst_addr,                                               \
    const variable_t  *src0,                                                   \
    const variable_t  *src1,                                                   \
    const variable_t  *src2,                                                   \
    const variable_t  *src3,                                                   \
    const dim4        *shape,                                                  \
    data_type_t        src0_src1_dtype,                                        \
    data_type_t        dst_dtype) {                                            \
    TPUKERNEL_ASSERT(src2->type != VECTOR && src3->type != VECTOR);            \
    atomic_fused_cmp_gen_cmd(                                                  \
        VALUE_OR_ADDR(src0),                                                   \
        VALUE_OR_ADDR(src1),                                                   \
        VALUE_OR_ADDR(src2),                                                   \
        VALUE_OR_ADDR(src3),                                                   \
        dst_addr,                                                              \
        NO_USE,                                                                \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        src0->type == SCALAR,                                                  \
        src1->type == SCALAR,                                                  \
        src2->type == SCALAR,                                                  \
        src3->type == SCALAR,                                                  \
        tpu_is_data_type_fp(src0_src1_dtype) ? FP8TYPE(src0_src1_dtype) : SIGN(src0_src1_dtype), \
        NO_USE,                                                                \
        NO_USE,                                                                \
        src0->type == VECTOR ? 3 : 0,                                          \
        src1->type == VECTOR ? 3 : 0,                                          \
        PRECISION(src0_src1_dtype),                                            \
        PRECISION(dst_dtype),                                                  \
        NO_USE,                                                                \
        op,                                                                    \
        MASTER_THREAD,                                                         \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_CMP_SELECT(greater, CMP_SG)
TPU_BDC_CMP_SELECT(less, CMP_SL)
TPU_BDC_CMP_SELECT(equal, CMP_SE)

void tpu_bdc_srch_bin_select(
    local_addr_t       dst_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const dim4        *shape,
    int                side,
    int                bin_w,
    data_type_t        src0_src1_dtype,
    data_type_t        dst_dtype) {
    atomic_fused_cmp_gen_cmd(
        VALUE_OR_ADDR(src0),
        VALUE_OR_ADDR(src1),
        NO_USE,
        NO_USE,
        dst_addr,
        NO_USE,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src0->type == SCALAR,
        NO_USE,
        NO_USE,
        NO_USE,
        tpu_is_data_type_fp(src0_src1_dtype) ? FP8TYPE(src0_src1_dtype) : SIGN(src0_src1_dtype),
        side,
        bin_w,
        src0->type == VECTOR ? 3 : 0,
        NO_USE,
        PRECISION(src0_src1_dtype),
        NO_USE,
        PRECISION(dst_dtype),
        CMP_SRCH_BIN,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

#define TPU_BDC_EXTR_CMP_SELECT(name, op)                                      \
void tpu_bdc_##name##_select(                                                  \
    local_addr_t       dst0_addr,                                              \
    local_addr_t       dst1_addr,                                              \
    const variable_t  *src0,                                                   \
    const variable_t  *src1,                                                   \
    const variable_t  *src2,                                                   \
    const variable_t  *src3,                                                   \
    const dim4        *shape,                                                  \
    data_type_t        dst0_dtype,                                             \
    data_type_t        dst1_dtype) {                                           \
    TPUKERNEL_ASSERT(src2->type != VECTOR && src3->type != VECTOR);            \
    atomic_fused_cmp_gen_cmd(                                                  \
        VALUE_OR_ADDR(src0),                                                   \
        VALUE_OR_ADDR(src1),                                                   \
        VALUE_OR_ADDR(src2),                                                   \
        VALUE_OR_ADDR(src3),                                                   \
        dst0_addr,                                                             \
        dst1_addr,                                                             \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        src0->type == SCALAR,                                                  \
        src1->type == SCALAR,                                                  \
        src2->type == SCALAR,                                                  \
        src3->type == SCALAR,                                                  \
        tpu_is_data_type_fp(dst0_dtype) ? FP8TYPE(dst0_dtype) : SIGN(dst0_dtype), \
        NO_USE,                                                                \
        NO_USE,                                                                \
        src0->type == VECTOR ? 3 : 0,                                          \
        src1->type == VECTOR ? 3 : 0,                                          \
        PRECISION(dst0_dtype),                                                 \
        PRECISION(dst1_dtype),                                                 \
        NO_USE,                                                                \
        op,                                                                    \
        MASTER_THREAD,                                                         \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_EXTR_CMP_SELECT(maximum_greater, CMP_GT_AND_SG)
TPU_BDC_EXTR_CMP_SELECT(minimum_less, CMP_LT_AND_SL)
#define TPU_BDC_BINARY(name, op)                                               \
void tpu_bdc_##name(                                                           \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src0_stride,                                                 \
    const dim4   *src1_stride,                                                 \
    data_type_t   dtype) {                                                     \
    if (src0_stride == NULL && src1_stride != NULL)                            \
        return tpu_bdc_##name(                                                 \
                   dst_addr,                                                   \
                   src1_addr,                                                  \
                   src0_addr,                                                  \
                   shape,                                                      \
                   dst_stride,                                                 \
                   src1_stride,                                                \
                   src0_stride,                                                \
                   dtype);                                                     \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3];                                                               \
    int tempValue;                                                             \
    if (IS_FLOAT(dtype)) {tempValue = FP8TYPE(dtype);} else {tempValue = SIGN(dtype);}\
    sign[0] = tempValue;                                                       \
    sign[1] = tempValue;                                                       \
    sign[2] = tempValue;                                                       \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src0_stride,                                                    \
        (int *)src1_stride,                                                    \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_BINARY(min, AR_MIN)
TPU_BDC_BINARY(max, AR_MAX)
#define TPU_BDC_BIN_BINARY(name, op)                                           \
void tpu_bdc_##name(                                                           \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src0_stride,                                                 \
    const dim4   *src1_stride,                                                 \
    data_type_t   dtype) {                                                     \
    if (src0_stride == NULL && src1_stride != NULL)                            \
        return tpu_bdc_##name(                                                 \
                   dst_addr,                                                   \
                   src1_addr,                                                  \
                   src0_addr,                                                  \
                   shape,                                                      \
                   dst_stride,                                                 \
                   src1_stride,                                                \
                   src0_stride,                                                \
                   dtype);                                                     \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    if (dtype == DT_FP32)                                                      \
        dtype = DT_INT32;                                                      \
    else if (dtype == DT_FP16 || dtype == DT_BFP16)                            \
        dtype = DT_INT16;                                                      \
    int sign[3] = {SIGN(dtype), SIGN(dtype), SIGN(dtype)};                     \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src0_stride,                                                    \
        (int *)src1_stride,                                                    \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_BIN_BINARY(and, AR_AND)
TPU_BDC_BIN_BINARY(or,  AR_OR)
TPU_BDC_BIN_BINARY(xor, AR_XOR)
#define TPU_BDC_FLOATING_POINT_BINARY(name, op)                                \
void tpu_bdc_fp_##name(                                                        \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src0_stride,                                                 \
    const dim4   *src1_stride,                                                 \
    data_type_t   dtype) {                                                     \
    if (src0_stride == NULL && src1_stride != NULL)                            \
        return tpu_bdc_fp_##name(                                              \
                   dst_addr,                                                   \
                   src1_addr,                                                  \
                   src0_addr,                                                  \
                   shape,                                                      \
                   dst_stride,                                                 \
                   src1_stride,                                                \
                   src0_stride,                                                \
                   dtype);                                                     \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));                               \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {FP8TYPE(dtype), FP8TYPE(dtype), FP8TYPE(dtype)};                                  \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src0_stride,                                                    \
        (int *)src1_stride,                                                    \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FLOATING_POINT_BINARY(add, AR_ADD)
TPU_BDC_FLOATING_POINT_BINARY(mul, AR_MUL)
TPU_BDC_FLOATING_POINT_BINARY(diff_abs, AR_FSUBABS)
void tpu_bdc_fp_sub(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {FP8TYPE(dtype), FP8TYPE(dtype), FP8TYPE(dtype)};
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};
    atomic_tensor_arithmetic_gen_cmd(
        src0_addr,
        src1_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_SUB,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_sub);
void tpu_bdc_fp32_mac(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride) {
    if (src0_stride == NULL && src1_stride != NULL)
        return tpu_bdc_fp32_mac(
                   dst_addr,
                   src1_addr,
                   src0_addr,
                   shape,
                   dst_stride,
                   src1_stride,
                   src0_stride);
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(DT_FP32), SIGN(DT_FP32), SIGN(DT_FP32)};
    int prec[3] = {PRECISION(DT_FP32), PRECISION(DT_FP32), PRECISION(DT_FP32)};
    atomic_tensor_arithmetic_gen_cmd(
        src0_addr,
        src1_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_BINARY_C(name, op)                                             \
void tpu_bdc_##name##_C(                                                       \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src_stride,                                                  \
    data_type_t   dtype) {                                                     \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3];                                                               \
    int tempValue;                                                             \
    if (IS_FLOAT(dtype)) {tempValue = FP8TYPE(dtype);} else {tempValue = SIGN(dtype);}\
    sign[0] = tempValue;                                                       \
    sign[1] = tempValue;                                                       \
    sign[2] = tempValue;                                                       \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src_addr,                                                              \
        C.u32,                                                                 \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src_stride,                                                     \
        NULL,                                                                  \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        true,                                                                  \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_BINARY_C(min, AR_MIN)
TPU_BDC_BINARY_C(max, AR_MAX)
#define TPU_BDC_BIN_BINARY_C(name, op)                                         \
void tpu_bdc_##name##_C(                                                       \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src_stride,                                                  \
    data_type_t   dtype) {                                                     \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    if (dtype == DT_FP32)                                                      \
        dtype = DT_INT32;                                                      \
    else if (dtype == DT_FP16 || dtype == DT_BFP16)                            \
        dtype = DT_INT16;                                                      \
    else if (dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2)                       \
        dtype = DT_INT8;                                                       \
    int sign[3] = {SIGN(dtype), SIGN(dtype), SIGN(dtype)};                     \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src_addr,                                                              \
        C.u32,                                                                 \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src_stride,                                                     \
        NULL,                                                                  \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        true,                                                                  \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_BIN_BINARY_C(and, AR_AND)
TPU_BDC_BIN_BINARY_C(or,  AR_OR)
TPU_BDC_BIN_BINARY_C(xor, AR_XOR)
#define TPU_BDC_FLOATING_POINT_BINARY_C(name, op)                              \
void tpu_bdc_fp_##name##_C(                                                    \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src_stride,                                                  \
    data_type_t   dtype) {                                                     \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));                               \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {FP8TYPE(dtype), FP8TYPE(dtype), FP8TYPE(dtype)};            \
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};      \
    atomic_tensor_arithmetic_gen_cmd(                                          \
        src_addr,                                                              \
        C.u32,                                                                 \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src_stride,                                                     \
        NULL,                                                                  \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        true,                                                                  \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FLOATING_POINT_BINARY_C(add, AR_ADD)
TPU_BDC_FLOATING_POINT_BINARY_C(sub, AR_SUB)
TPU_BDC_FLOATING_POINT_BINARY_C(mul, AR_MUL)
TPU_BDC_FLOATING_POINT_BINARY_C(diff_abs, AR_FSUBABS)
RTM_EXPORT(tpu_bdc_fp_add_C);
void tpu_bdc_fp32_mac_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride) {
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(DT_FP32), SIGN(DT_FP32), SIGN(DT_FP32)};
    int prec[3] = {PRECISION(DT_FP32), PRECISION(DT_FP32), PRECISION(DT_FP32)};
    scalar_t C_ = {.f32 = C};
    atomic_tensor_arithmetic_gen_cmd(
        src_addr,
        C_.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        NULL,
        (int *)dst_stride,
        false,
        true,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_C_sub(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    int short_str[3] = {
        NO_USE,
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {FP8TYPE(dtype), FP8TYPE(dtype), FP8TYPE(dtype)};
    int prec[3] = {PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)};
    atomic_tensor_arithmetic_gen_cmd(
        C.u32,
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        (int *)src_stride,
        (int *)dst_stride,
        true,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_SUB,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_cpy(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    dim4 dst_stride_aligned, src_stride_aligned;
    if (dst_stride == NULL) {
        tpu_aligned_stride(&dst_stride_aligned, 0, shape, dtype);
    } else {
        memcpy(&dst_stride_aligned, dst_stride, sizeof(dim4));
    }
    if (src_stride == NULL) {
        tpu_aligned_stride(&src_stride_aligned, 0, shape, dtype);
    } else {
        memcpy(&src_stride_aligned, src_stride, sizeof(dim4));
    }

    const int BD_MAX_NSTRIDE = (1 << 18)-1;
    if (shape->n == 1 && (
        dst_stride_aligned.n > BD_MAX_NSTRIDE ||
        src_stride_aligned.n > BD_MAX_NSTRIDE)) {
        // set nstride to 0 due the limit of BD
        dst_stride_aligned.n = 0;
        src_stride_aligned.n = 0;
    }
    int short_str[2] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_single_opd_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)&src_stride_aligned,
        (int *)&dst_stride_aligned,
        false,
        short_str,
        SIGN(dtype),
        PRECISION(dtype),
        AR_COPY,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_cpy);
void tpu_bdc_cpy_cross_npu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    atomic_lane_copy_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_not(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    int short_str[2] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_single_opd_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        (int *)dst_stride,
        false,
        short_str,
        SIGN(dtype),
        PRECISION(dtype),
        AR_NOT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_set_C(
    local_addr_t  dst_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    data_type_t   dtype) {
    int short_str[2] = {
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_single_opd_gen_cmd(
        C.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        (int *)dst_stride,
        true,
        short_str,
        SIGN(dtype),
        PRECISION(dtype),
        AR_COPY,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_set_C);
void tpu_bdc_cast(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode) {
    TPUKERNEL_ASSERT(
        mode == RM_HALF_TO_EVEN ||
        mode == RM_HALF_AWAY_FROM_ZERO ||
        mode == RM_TOWARDS_ZERO ||
        mode == RM_UP ||
        mode == RM_DOWN);
    int short_str[2] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    if (dst_dtype != src_dtype) {
        int sign[2];
        sign[0] = tpu_is_data_type_fp8(src_dtype) ? FP8TYPE(src_dtype) : SIGN(src_dtype);
        sign[1] = tpu_is_data_type_fp8(dst_dtype) ? FP8TYPE(dst_dtype) : SIGN(dst_dtype);
        int prec[2] = {PRECISION(src_dtype), PRECISION(dst_dtype)};
        // fp convert will do saturation
        int satu_mode = tpu_is_data_type_fp(src_dtype) && tpu_is_data_type_fp(dst_dtype) &&
                            (tpu_data_type_size(src_dtype) > tpu_data_type_size(dst_dtype));
        if (!satu_mode && src_dtype == DT_FP8E5M2) {
            satu_mode = (dst_dtype == DT_FP8E4M3);
        }
        atomic_tensor_arithmetic_dtype_convert_gen_cmd(
            src_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            (int *)dst_stride,
            false,
            short_str,
            sign,
            satu_mode,
            (PREC *)prec,
            mode,
            MASTER_THREAD,
            BDC_NODE);
    } else {
        atomic_tensor_arithmetic_single_opd_gen_cmd(
            src_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            (int *)dst_stride,
            false,
            short_str,
            SIGN(dst_dtype),
            PRECISION(dst_dtype),
            AR_COPY,
            MASTER_THREAD,
            BDC_NODE);
    }
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_cast);
void tpu_bdc_abs(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    int short_str[2] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    if (SIGN(dtype))
        atomic_tensor_arithmetic_single_opd_gen_cmd(
            src_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            (int *)dst_stride,
            false,
            short_str,
            SIGN(dtype),
            PRECISION(dtype),
            AR_ABS,
            MASTER_THREAD,
            BDC_NODE);
    else
        atomic_tensor_arithmetic_single_opd_gen_cmd(
            src_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            (int *)dst_stride,
            false,
            short_str,
            SIGN(dtype),
            PRECISION(dtype),
            AR_COPY,
            MASTER_THREAD,
            BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_round(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dtype,
    rounding_mode_t  mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    TPUKERNEL_ASSERT(
        mode == RM_HALF_TO_EVEN || mode == RM_HALF_AWAY_FROM_ZERO ||
        mode == RM_TOWARDS_ZERO || mode == RM_DOWN || mode == RM_UP);
    int short_str[2] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[2] = {SIGN(dtype), SIGN(dtype)};
    int prec[2] = {PRECISION(dtype), PRECISION(dtype)};
    atomic_tensor_arithmetic_dtype_convert_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        (int *)dst_stride,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        (ROUND_MODE)mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_floor(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    tpu_bdc_fp_round(
        dst_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        dtype,
        RM_DOWN);
}
RTM_EXPORT(tpu_bdc_fp_floor);
void tpu_bdc_fp_ceil(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    tpu_bdc_fp_round(
        dst_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        dtype,
        RM_UP);
}
#define TPU_BDC_FIXED_POINT_BINARY(name, op)                                   \
void tpu_bdc_int_##name(                                                       \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src0_addr,                                                \
    local_addr_t     src1_addr,                                                \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src0_stride,                                              \
    const dim4      *src1_stride,                                              \
    data_type_t      dst_dtype,                                                \
    data_type_t      src0_dtype,                                               \
    data_type_t      src1_dtype,                                               \
    char             shift,                                                    \
    rounding_mode_t  rounding_mode,                                            \
    bool             saturation) {                                             \
    if (src0_stride == NULL && src1_stride != NULL)                            \
        return tpu_bdc_int_##name(                                             \
                   dst_addr,                                                   \
                   src1_addr,                                                  \
                   src0_addr,                                                  \
                   shape,                                                      \
                   dst_stride,                                                 \
                   src1_stride,                                                \
                   src0_stride,                                                \
                   dst_dtype,                                                  \
                   src1_dtype,                                                 \
                   src0_dtype,                                                 \
                   shift,                                                      \
                   rounding_mode,                                              \
                   saturation);                                                \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src0_dtype));                         \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src1_dtype));                         \
    TPUKERNEL_ASSERT(SIGN(dst_dtype) == (SIGN(src0_dtype) | SIGN(src1_dtype))); \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {SIGN(src0_dtype), SIGN(src1_dtype), SIGN(dst_dtype)};                        \
    if (shift == 0) {                                                          \
        int prec[3] = {                                                        \
            PRECISION(src0_dtype), PRECISION(src1_dtype), PRECISION(dst_dtype) \
        };                                                                     \
        atomic_tensor_arithmetic_gen_cmd(                                      \
                src0_addr,                                                     \
                src1_addr,                                                     \
                dst_addr,                                                      \
                shape->n,                                                      \
                shape->c,                                                      \
                shape->h,                                                      \
                shape->w,                                                      \
                (int *)src0_stride,                                            \
                (int *)src1_stride,                                            \
                (int *)dst_stride,                                             \
                false,                                                         \
                false,                                                         \
                short_str,                                                     \
                sign,                                                          \
                0,                                                             \
                (PREC *)prec,                                                  \
                op,                                                            \
                MASTER_THREAD,                                              \
                BDC_NODE);                                                     \
    } else {                                                                   \
        int prec[4] = {                                                        \
            PRECISION(src0_dtype), PRECISION(src1_dtype),                      \
            PRECISION(DT_INT8), PRECISION(dst_dtype)                           \
        };                                                                     \
        scalar_t scalar = {.s8 = shift};                                       \
        atomic_tensor_arithmetic_ternary_gen_cmd(                              \
                src0_addr,                                                     \
                src1_addr,                                                     \
                scalar.u32,                                                    \
                dst_addr,                                                      \
                shape->n,                                                      \
                shape->c,                                                      \
                shape->h,                                                      \
                shape->w,                                                      \
                (int *)src0_stride,                                            \
                (int *)src1_stride,                                            \
                (int *)dst_stride,                                             \
                false,                                                         \
                false,                                                         \
                true,                                                          \
                short_str,                                                     \
                sign,                                                          \
                0,                                                             \
                (PREC *)prec,                                                  \
                op,                                                            \
                (ROUND_MODE)rounding_mode,                                     \
                MASTER_THREAD,                                              \
                BDC_NODE);                                                     \
    }                                                                          \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FIXED_POINT_BINARY(add, saturation ? AR_ADD_SATU : AR_ADD)
TPU_BDC_FIXED_POINT_BINARY(mul, saturation ? AR_MUL_SATU : AR_MUL)
void tpu_bdc_int_sub(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src0_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src1_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(src0_dtype), SIGN(src1_dtype), SIGN(dst_dtype)};
    if (shift == 0) {
        int prec[3] = {
            PRECISION(src0_dtype), PRECISION(src1_dtype), PRECISION(dst_dtype)
        };
        atomic_tensor_arithmetic_gen_cmd(
            src0_addr,
            src1_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src0_stride,
            (int *)src1_stride,
            (int *)dst_stride,
            false,
            false,
            short_str,
            sign,
            0,
            (PREC *)prec,
            saturation ? AR_SUB_SATU : AR_SUB,
            MASTER_THREAD,
            BDC_NODE);
    } else {
        int prec[4] = {
            PRECISION(src0_dtype), PRECISION(src1_dtype),
            PRECISION(DT_INT8), PRECISION(dst_dtype)
        };
        scalar_t scalar = {.s8 = shift};
        atomic_tensor_arithmetic_ternary_gen_cmd(
            src0_addr,
            src1_addr,
            scalar.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src0_stride,
            (int *)src1_stride,
            (int *)dst_stride,
            false,
            false,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            saturation ? AR_SUB_SATU : AR_SUB,
            (ROUND_MODE)rounding_mode,
            MASTER_THREAD,
            BDC_NODE);
    }
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_square(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation) {
    tpu_bdc_int_mul(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        src_stride,
        dst_dtype,
        src_dtype,
        src_dtype,
        shift,
        rounding_mode,
        saturation);
}
void tpu_bdc_int8_mac(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    unsigned char    lshift,
    unsigned char    rshift,
    rounding_mode_t  rounding_mode) {
    if (src0_stride == NULL && src1_stride != NULL)
        return tpu_bdc_int8_mac(
                   dst_addr,
                   src1_addr,
                   src0_addr,
                   shape,
                   dst_stride,
                   src1_stride,
                   src0_stride,
                   src1_dtype,
                   src0_dtype,
                   lshift,
                   rshift,
                   rounding_mode);
    TPUKERNEL_ASSERT(src0_dtype == DT_INT8 || src0_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(src1_dtype == DT_INT8 || src1_dtype == DT_UINT8);
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(src0_dtype), SIGN(src1_dtype), SIGN(src1_dtype)};
    int prec[4] = {
        PRECISION(DT_INT8), PRECISION(DT_INT8), NO_USE, PRECISION(DT_INT16)
    };
    unsigned int shift = lshift | (rshift << 8);
    atomic_tensor_arithmetic_ternary_gen_cmd(
        src0_addr,
        src1_addr,
        shift,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        true,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_MAC,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_mac_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    unsigned char    lshift,
    unsigned char    rshift,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(src_dtype == DT_INT8 || src_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(C_dtype == DT_INT8 || C_dtype == DT_UINT8);
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(src_dtype), SIGN(C_dtype), SIGN(C_dtype)};
    int prec[4] = {
        PRECISION(DT_INT8), PRECISION(DT_INT8), NO_USE, PRECISION(DT_INT16)
    };
    unsigned int shift = lshift | (rshift << 8);
    atomic_tensor_arithmetic_ternary_gen_cmd(
        src_addr,
        C.u32,
        shift,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        NULL,
        (int *)dst_stride,
        false,
        true,
        true,
        short_str,
        sign,
        0,
        (PREC *)prec,
        AR_MAC,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_FIXED_POINT_BINARY_C(name, op, sign_check)                     \
void tpu_bdc_int_##name##_C(                                                   \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src_addr,                                                 \
    scalar_t         C,                                                        \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src_stride,                                               \
    data_type_t      dst_dtype,                                                \
    data_type_t      src_dtype,                                                \
    data_type_t      C_dtype,                                                  \
    char             shift,                                                    \
    rounding_mode_t  rounding_mode,                                            \
    bool             saturation) {                                             \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));                            \
    TPUKERNEL_ASSERT(sign_check);                                               \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {SIGN(src_dtype), SIGN(C_dtype), SIGN(dst_dtype)};                            \
    if (shift == 0) {                                                          \
        int prec[3] = {                                                        \
            PRECISION(src_dtype), PRECISION(C_dtype), PRECISION(dst_dtype)     \
        };                                                                     \
        atomic_tensor_arithmetic_gen_cmd(                                      \
            src_addr,                                                          \
            C.u32,                                                             \
            dst_addr,                                                          \
            shape->n,                                                          \
            shape->c,                                                          \
            shape->h,                                                          \
            shape->w,                                                          \
            (int *)src_stride,                                                 \
            NULL,                                                              \
            (int *)dst_stride,                                                 \
            false,                                                             \
            true,                                                              \
            short_str,                                                         \
            sign,                                                              \
            0,                                                                 \
            (PREC *)prec,                                                      \
            op,                                                                \
            MASTER_THREAD,                                                  \
            BDC_NODE);                                                         \
    } else {                                                                   \
        int prec[4] = {                                                        \
            PRECISION(src_dtype), PRECISION(C_dtype),                          \
            PRECISION(DT_INT8), PRECISION(dst_dtype)                           \
        };                                                                     \
        scalar_t scalar = {.s8 = shift};                                       \
        atomic_tensor_arithmetic_ternary_gen_cmd(                              \
            src_addr,                                                          \
            C.u32,                                                             \
            scalar.u32,                                                        \
            dst_addr,                                                          \
            shape->n,                                                          \
            shape->c,                                                          \
            shape->h,                                                          \
            shape->w,                                                          \
            (int *)src_stride,                                                 \
            NULL,                                                              \
            (int *)dst_stride,                                                 \
            false,                                                             \
            true,                                                              \
            true,                                                              \
            short_str,                                                         \
            sign,                                                              \
            0,                                                                 \
            (PREC *)prec,                                                      \
            op,                                                                \
            (ROUND_MODE)rounding_mode,                                         \
            MASTER_THREAD,                                                  \
            BDC_NODE);                                                         \
    }                                                                          \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FIXED_POINT_BINARY_C(
    add, saturation ? AR_ADD_SATU : AR_ADD,
    SIGN(dst_dtype) == (SIGN(src_dtype) | SIGN(C_dtype)))
TPU_BDC_FIXED_POINT_BINARY_C(
    sub, saturation ? AR_SUB_SATU : AR_SUB, SIGN(dst_dtype))
TPU_BDC_FIXED_POINT_BINARY_C(
    mul, saturation ? AR_MUL_SATU : AR_MUL,
    SIGN(dst_dtype) == (SIGN(src_dtype) | SIGN(C_dtype)))
void tpu_bdc_int_C_sub(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype));
    int short_str[3] = {
        NO_USE,
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(C_dtype), SIGN(src_dtype), SIGN(dst_dtype)};
    if (shift == 0) {
        int prec[3] = {
            PRECISION(C_dtype), PRECISION(src_dtype), PRECISION(dst_dtype)
        };
        atomic_tensor_arithmetic_gen_cmd(
            C.u32,
            src_addr,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            NULL,
            (int *)src_stride,
            (int *)dst_stride,
            true,
            false,
            short_str,
            sign,
            0,
            (PREC *)prec,
            saturation ? AR_SUB_SATU : AR_SUB,
            MASTER_THREAD,
            BDC_NODE);
    } else {
        int prec[4] = {
            PRECISION(C_dtype), PRECISION(src_dtype),
            PRECISION(DT_INT8), PRECISION(dst_dtype)
        };
        scalar_t scalar = {.s8 = shift};
        atomic_tensor_arithmetic_ternary_gen_cmd(
            C.u32,
            src_addr,
            scalar.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            NULL,
            (int *)src_stride,
            (int *)dst_stride,
            true,
            false,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            saturation ? AR_SUB_SATU : AR_SUB,
            (ROUND_MODE)rounding_mode,
            MASTER_THREAD,
            BDC_NODE);
    }
    CHECK_BDC_OVERFLOW;
}

#define TPU_BDC_FP8_BINARY_C(name, op)                     \
void tpu_bdc_fp8_##name##_C(                                                   \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src_addr,                                                 \
    scalar_t         C,                                                        \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src_stride,                                               \
    data_type_t      dst_dtype,                                                \
    data_type_t      src_dtype,                                                \
    data_type_t      C_dtype,                                                  \
    int             satu_mode) {                                             \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp8(src_dtype));                         \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(C_dtype));                            \
    TPUKERNEL_ASSERT(tpu_data_type_size(C_dtype) <= tpu_data_type_size(dst_dtype)); \
    TPUKERNEL_ASSERT(tpu_data_type_size(src_dtype) <= tpu_data_type_size(C_dtype)); \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {FP8TYPE(src_dtype), FP8TYPE(C_dtype), FP8TYPE(dst_dtype)};  \
    int prec[3] = {                                                        \
        PRECISION(src_dtype), PRECISION(C_dtype), PRECISION(dst_dtype)     \
    };                                                                     \
    atomic_tensor_arithmetic_gen_cmd(                                      \
        src_addr,                                                          \
        C.u32,                                                             \
        dst_addr,                                                          \
        shape->n,                                                          \
        shape->c,                                                          \
        shape->h,                                                          \
        shape->w,                                                          \
        (int *)src_stride,                                                 \
        NULL,                                                              \
        (int *)dst_stride,                                                 \
        false,                                                             \
        true,                                                              \
        short_str,                                                         \
        sign,                                                              \
        satu_mode,                                                         \
        (PREC *)prec,                                                      \
        op,                                                                \
        MASTER_THREAD,                                                     \
        BDC_NODE);                                                         \
    CHECK_BDC_OVERFLOW;                                                    \
}
TPU_BDC_FP8_BINARY_C(add, AR_ADD)
TPU_BDC_FP8_BINARY_C(mul, AR_MUL)
void tpu_bdc_fp8_C_sub(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    int              satu_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp8(src_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(C_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dst_dtype));
    TPUKERNEL_ASSERT(tpu_data_type_size(C_dtype) <= tpu_data_type_size(dst_dtype));
    TPUKERNEL_ASSERT(tpu_data_type_size(src_dtype) <= tpu_data_type_size(C_dtype));
    int short_str[3] = {
        NO_USE,
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {FP8TYPE(C_dtype), FP8TYPE(src_dtype), FP8TYPE(dst_dtype)};
    int prec[3] = {
        PRECISION(C_dtype), PRECISION(src_dtype), PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_gen_cmd(
        C.u32,
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        (int *)src_stride,
        (int *)dst_stride,
        true,
        false,
        short_str,
        sign,
        satu_mode,
        (PREC *)prec,
        AR_SUB,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_int_min_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t     dtype,
    char            shift,
    rounding_mode_t rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(dtype), SIGN(dtype), SIGN(dtype)};
    if (shift == 0) {
        int prec[3] = {
            PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)
        };
        atomic_tensor_arithmetic_gen_cmd(
            src_addr,
            C.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            NULL,
            (int *)dst_stride,
            false,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            AR_MIN,
            MASTER_THREAD,
            BDC_NODE);
    } else {
        int prec[4] = {
            PRECISION(dtype), PRECISION(dtype),
            PRECISION(DT_INT8), PRECISION(dtype)
        };
        scalar_t scalar = {.s8 = shift};
        atomic_tensor_arithmetic_ternary_gen_cmd(
            src_addr,
            C.u32,
            scalar.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            NULL,
            (int *)dst_stride,
            false,
            true,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            AR_MIN,
            (ROUND_MODE)rounding_mode,
            MASTER_THREAD,
            BDC_NODE);
    }
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_max_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t     dtype,
    char            shift,
    rounding_mode_t rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(dtype), SIGN(dtype), SIGN(dtype)};
    if (shift == 0) {
        int prec[3] = {
            PRECISION(dtype), PRECISION(dtype), PRECISION(dtype)
        };
        atomic_tensor_arithmetic_gen_cmd(
            src_addr,
            C.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            NULL,
            (int *)dst_stride,
            false,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            AR_MAX,
            MASTER_THREAD,
            BDC_NODE);
    } else {
        int prec[4] = {
            PRECISION(dtype), PRECISION(dtype),
            PRECISION(DT_INT8), PRECISION(dtype)
        };
        scalar_t scalar = {.s8 = shift};
        atomic_tensor_arithmetic_ternary_gen_cmd(
            src_addr,
            C.u32,
            scalar.u32,
            dst_addr,
            shape->n,
            shape->c,
            shape->h,
            shape->w,
            (int *)src_stride,
            NULL,
            (int *)dst_stride,
            false,
            true,
            true,
            short_str,
            sign,
            0,
            (PREC *)prec,
            AR_MAX,
            (ROUND_MODE)rounding_mode,
            MASTER_THREAD,
            BDC_NODE);
    }
    CHECK_BDC_OVERFLOW;
}

#define TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY(name, op)                 \
void tpu_bdc_int_pcs_##name(                                                   \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src0_addr,                                                \
    local_addr_t     src1_addr,                                                \
    local_addr_t     shift_addr,                                               \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src0_stride,                                              \
    const dim4      *src1_stride,                                              \
    data_type_t      dst_dtype,                                                \
    data_type_t      src0_dtype,                                               \
    data_type_t      src1_dtype,                                               \
    rounding_mode_t  rounding_mode,                                            \
    bool             saturation) {                                             \
    if (src0_stride == NULL && src1_stride != NULL)                            \
        return tpu_bdc_int_pcs_##name(                                         \
                   dst_addr,                                                   \
                   src1_addr,                                                  \
                   src0_addr,                                                  \
                   shift_addr,                                                 \
                   shape,                                                      \
                   dst_stride,                                                 \
                   src1_stride,                                                \
                   src0_stride,                                                \
                   dst_dtype,                                                  \
                   src1_dtype,                                                 \
                   src0_dtype,                                                 \
                   rounding_mode,                                              \
                   saturation);                                                \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src0_dtype));                         \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src1_dtype));                         \
    TPUKERNEL_ASSERT(SIGN(dst_dtype) == (SIGN(src0_dtype) | SIGN(src1_dtype))); \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {SIGN(src0_dtype), SIGN(src1_dtype), SIGN(dst_dtype)};                        \
    int prec[4] = {                                                            \
        PRECISION(src0_dtype), PRECISION(src1_dtype),                          \
        PRECISION(DT_INT8), PRECISION(dst_dtype)                               \
    };                                                                         \
    atomic_tensor_arithmetic_ternary_gen_cmd(                                  \
        src0_addr,                                                             \
        src1_addr,                                                             \
        shift_addr,                                                            \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src0_stride,                                                    \
        (int *)src1_stride,                                                    \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        false,                                                                 \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        (ROUND_MODE)rounding_mode,                                             \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY(
    add, saturation ? AR_ADD_SATU : AR_ADD)
TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY(
    mul, saturation ? AR_MUL_SATU : AR_MUL)
void tpu_bdc_int_pcs_sub(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src0_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src1_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(src0_dtype), SIGN(src1_dtype), SIGN(dst_dtype)};
    int prec[4] = {
        PRECISION(src0_dtype), PRECISION(src1_dtype),
        PRECISION(DT_INT8), PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_ternary_gen_cmd(
        src0_addr,
        src1_addr,
        shift_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        saturation ? AR_SUB_SATU : AR_SUB,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY_C(name, op, sign_check)   \
void tpu_bdc_int_pcs_##name##_C(                                               \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src_addr,                                                 \
    local_addr_t     shift_addr,                                               \
    scalar_t         C,                                                        \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src_stride,                                               \
    data_type_t      dst_dtype,                                                \
    data_type_t      src_dtype,                                                \
    data_type_t      C_dtype,                                                  \
    rounding_mode_t  rounding_mode,                                            \
    bool             saturation) {                                             \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));                            \
    TPUKERNEL_ASSERT(sign_check);                                               \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3] = {SIGN(src_dtype), SIGN(C_dtype), SIGN(dst_dtype)};                            \
    int prec[4] = {                                                            \
        PRECISION(src_dtype), PRECISION(C_dtype),                              \
        PRECISION(DT_INT8), PRECISION(dst_dtype)                               \
    };                                                                         \
    atomic_tensor_arithmetic_ternary_gen_cmd(                                  \
        src_addr,                                                              \
        C.u32,                                                                 \
        shift_addr,                                                            \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src_stride,                                                     \
        NULL,                                                                  \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        true,                                                                  \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        0,                                                                     \
        (PREC *)prec,                                                          \
        op,                                                                    \
        (ROUND_MODE)rounding_mode,                                             \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY_C(
    add, saturation ? AR_ADD_SATU : AR_ADD,
    SIGN(dst_dtype) == (SIGN(src_dtype) | SIGN(C_dtype)))
TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY_C(
    sub, saturation ? AR_SUB_SATU : AR_SUB, SIGN(dst_dtype))
TPU_BDC_FIXED_POINT_PER_CHANNEL_SHIFT_BINARY_C(
    mul, saturation ? AR_MUL_SATU : AR_MUL,
    SIGN(dst_dtype) == (SIGN(src_dtype) | SIGN(C_dtype)))
void tpu_bdc_int_pcs_C_sub(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype));
    int short_str[3] = {
        NO_USE,
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3] = {SIGN(C_dtype), SIGN(src_dtype), SIGN(dst_dtype)};
    int prec[4] = {
        PRECISION(C_dtype), PRECISION(src_dtype),
        PRECISION(DT_INT8), PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_ternary_gen_cmd(
        C.u32,
        src_addr,
        shift_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        (int *)src_stride,
        (int *)dst_stride,
        true,
        false,
        false,
        short_str,
        sign,
        0,
        (PREC *)prec,
        saturation ? AR_SUB_SATU : AR_SUB,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_arithmetic_shift(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    const dim4      *shift_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      shift_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_signed_int(shift_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype) == SIGN(src_dtype));
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(shift_dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(shift_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[2] = {SIGN(src_dtype), 1};
    int prec[3] = {
        PRECISION(src_dtype),
        PRECISION(shift_dtype),
        PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_with_round_gen_cmd(
        src_addr,
        shift_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        (int *)shift_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        sign,
        (PREC *)prec,
        AR_ARITH_SHIFT,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_arithmetic_shift_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    char             C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    TPUKERNEL_ASSERT(SIGN(dst_dtype) == SIGN(src_dtype));
    TPUKERNEL_ASSERT(C <= 32 && C >= -32);
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[2] = {SIGN(src_dtype), 1};
    int prec[3] = {
        PRECISION(src_dtype),
        PRECISION(DT_INT8),
        PRECISION(dst_dtype)
    };
    scalar_t C_ = {.s8 = C};
    atomic_tensor_arithmetic_with_round_gen_cmd(
        src_addr,
        C_.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        NULL,
        (int *)dst_stride,
        false,
        true,
        short_str,
        sign,
        (PREC *)prec,
        AR_ARITH_SHIFT,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_logical_shift(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    const dim4      *shift_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      shift_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(src_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_signed_int(shift_dtype));
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(shift_dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(shift_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[2] = {0, 1};
    int prec[3] = {
        PRECISION(src_dtype),
        PRECISION(shift_dtype),
        PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_with_round_gen_cmd(
        src_addr,
        shift_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        (int *)shift_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        sign,
        (PREC *)prec,
        AR_LOGIC_SHIFT,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_logical_shift_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    char             C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(src_dtype));
    TPUKERNEL_ASSERT(C <= 32 && C >= -32);
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[2] = {0, 1};
    int prec[3] = {
        PRECISION(src_dtype),
        PRECISION(DT_INT8),
        PRECISION(dst_dtype)
    };
    scalar_t C_ = {.s8 = C};
    atomic_tensor_arithmetic_with_round_gen_cmd(
        src_addr,
        C_.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        NULL,
        (int *)dst_stride,
        false,
        true,
        short_str,
        sign,
        (PREC *)prec,
        AR_LOGIC_SHIFT,
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_CMP(name, op)                                                  \
void tpu_bdc_##name(                                                           \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    scalar_t      true_val,                                                    \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src0_stride,                                                 \
    const dim4   *src1_stride,                                                 \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(dst_dtype));                    \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src0_stride),                                          \
        ALIGNED_OR_USER(src1_stride),                                          \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3];                                                               \
    int tempValue1, tempValue2;                                                \
    if (IS_FLOAT(src_dtype)) {tempValue1 = FP8TYPE(src_dtype);tempValue2 = FP8TYPE(dst_dtype);} else {tempValue1 = SIGN(src_dtype);tempValue2 = SIGN(dst_dtype);}\
    sign[0] = tempValue1;                                                       \
    sign[1] = tempValue1;                                                       \
    sign[2] = tempValue2;                                                       \
    int prec[2] = {                                                            \
        PRECISION(src_dtype), PRECISION(dst_dtype)                             \
    };                                                                         \
    atomic_tensor_arithmetic_select_gen_cmd(                                   \
        src0_addr,                                                             \
        src1_addr,                                                             \
        true_val.u32,                                                          \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src0_stride,                                                    \
        (int *)src1_stride,                                                    \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        false,                                                                 \
        short_str,                                                             \
        sign,                                                                  \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_CMP(greater, AR_SG)
TPU_BDC_CMP(less,    AR_SL)
void tpu_bdc_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype) {
    if (src0_stride == NULL && src1_stride != NULL)
        return tpu_bdc_equal(
                   dst_addr,
                   src1_addr,
                   src0_addr,
                   true_val,
                   shape,
                   dst_stride,
                   src1_stride,
                   src0_stride,
                   dst_dtype,
                   src_dtype);
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(dst_dtype));
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    int sign[3];
    int tempValue1, tempValue2;
    if (IS_FLOAT(src_dtype)) {tempValue1 = FP8TYPE(src_dtype);tempValue2 = FP8TYPE(dst_dtype);} else {tempValue1 = SIGN(src_dtype);tempValue2 = SIGN(dst_dtype);}\
    sign[0] = tempValue1;
    sign[1] = tempValue1;
    sign[2] = tempValue2;
    int prec[2] = {
        PRECISION(src_dtype), PRECISION(dst_dtype)
    };
    atomic_tensor_arithmetic_select_gen_cmd(
        src0_addr,
        src1_addr,
        true_val.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        sign,
        (PREC *)prec,
        AR_SE,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_CMP_EXT(name, origin)                                          \
void tpu_bdc_##name(                                                           \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    scalar_t      true_val,                                                    \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src0_stride,                                                 \
    const dim4   *src1_stride,                                                 \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    tpu_bdc_##origin(                                                          \
        dst_addr,                                                              \
        src0_addr,                                                             \
        src1_addr,                                                             \
        true_val,                                                              \
        shape,                                                                 \
        dst_stride,                                                            \
        src0_stride,                                                           \
        src1_stride,                                                           \
        dst_dtype,                                                             \
        src_dtype);                                                            \
    tpu_bdc_xor_C(                                                             \
        dst_addr,                                                              \
        dst_addr,                                                              \
        true_val,                                                              \
        shape,                                                                 \
        dst_stride,                                                            \
        dst_stride,                                                            \
        dst_dtype);                                                            \
}
TPU_BDC_CMP_EXT(greater_equal, less)
TPU_BDC_CMP_EXT(less_equal, greater)
TPU_BDC_CMP_EXT(not_equal, equal)
#define TPU_BDC_CMP_C(name, op)                                                \
void tpu_bdc_##name##_C(                                                       \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    scalar_t      true_val,                                                    \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src_stride,                                                  \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(dst_dtype));                    \
    int short_str[3] = {                                                       \
        ALIGNED_OR_USER(src_stride),                                           \
        NO_USE,                                                                \
        ALIGNED_OR_USER(dst_stride)                                            \
    };                                                                         \
    int sign[3];                                                               \
    int tempValue1, tempValue2;                                                \
    if (IS_FLOAT(src_dtype)) {tempValue1 = FP8TYPE(src_dtype);tempValue2 = FP8TYPE(dst_dtype);} else {tempValue1 = SIGN(src_dtype);tempValue2 = SIGN(dst_dtype);}\
    sign[0] = tempValue1;                                                      \
    sign[1] = tempValue1;                                                      \
    sign[2] = tempValue2;                                                      \
    int prec[2] = {                                                            \
        PRECISION(src_dtype), PRECISION(dst_dtype)                             \
    };                                                                         \
    atomic_tensor_arithmetic_select_gen_cmd(                                   \
        src_addr,                                                              \
        C.u32,                                                                 \
        true_val.u32,                                                          \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        (int *)src_stride,                                                     \
        NULL,                                                                  \
        (int *)dst_stride,                                                     \
        false,                                                                 \
        true,                                                                  \
        short_str,                                                             \
        sign,                                                                  \
        (PREC *)prec,                                                          \
        op,                                                                    \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_CMP_C(greater, AR_SG)
TPU_BDC_CMP_C(less,    AR_SL)
TPU_BDC_CMP_C(equal,   AR_SE)
#define TPU_BDC_CMP_C_EXT(name, origin)                                        \
void tpu_bdc_##name##_C(                                                   \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    scalar_t      true_val,                                                    \
    const dim4   *shape,                                                       \
    const dim4   *dst_stride,                                                  \
    const dim4   *src_stride,                                                  \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    tpu_bdc_##origin##_C(                                                      \
        dst_addr,                                                              \
        src_addr,                                                              \
        C,                                                                     \
        true_val,                                                              \
        shape,                                                                 \
        dst_stride,                                                            \
        src_stride,                                                            \
        dst_dtype,                                                             \
        src_dtype);                                                            \
    tpu_bdc_xor_C(                                                             \
        dst_addr,                                                              \
        dst_addr,                                                              \
        true_val,                                                              \
        shape,                                                                 \
        dst_stride,                                                            \
        dst_stride,                                                            \
        dst_dtype);                                                            \
}
TPU_BDC_CMP_C_EXT(greater_equal, less)
TPU_BDC_CMP_C_EXT(less_equal, greater)
TPU_BDC_CMP_C_EXT(not_equal, equal)
void tpu_bdc_fp_bias(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_fused_linear_gen_cmd(
        src_addr,
        FP_ONE(dtype),
        bias_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        false,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype),
        LIN_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_FP_LINEAR_SQR(name, op_lin)                                    \
void tpu_bdc_fp_##name##_bias_sqr(                                             \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    local_addr_t  bias_addr,                                                   \
    const dim4   *shape,                                                       \
    data_type_t   dtype) {                                                     \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));                              \
    atomic_fused_linear_gen_cmd(                                               \
        src_addr,                                                              \
        bias_addr,                                                             \
        NO_USE,                                                                \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        false,                                                                 \
        false,                                                                 \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        FP8TYPE(dtype),                                                        \
        op_lin,                                                                \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FP_LINEAR_SQR(add, LIN_ADD_SQR)
TPU_BDC_FP_LINEAR_SQR(sub, LIN_SUB_SQR)
#define TPU_BDC_FP_LINEAR_SQR_C(name, op_lin)                                  \
void tpu_bdc_fp_##name##_C_sqr(                                                \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src_addr,                                                    \
    scalar_t      C,                                                           \
    const dim4   *shape,                                                       \
    data_type_t   dtype) {                                                     \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));                              \
    atomic_fused_linear_gen_cmd(                                               \
        src_addr,                                                              \
        C.u32,                                                                 \
        NO_USE,                                                                \
        dst_addr,                                                              \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->h,                                                              \
        shape->w,                                                              \
        true,                                                                  \
        false,                                                                 \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        FP8TYPE(dtype),                                                        \
        op_lin,                                                                \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FP_LINEAR_SQR_C(add, LIN_ADD_SQR)
TPU_BDC_FP_LINEAR_SQR_C(sub, LIN_SUB_SQR)
void tpu_bdc_fp_scale(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  scale_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_fused_linear_gen_cmd(
        src_addr,
        scale_addr,
        0,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        true,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype),
        LIN_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_scale_bias(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  scale_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_fused_linear_gen_cmd(
        src_addr,
        scale_addr,
        bias_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        false,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype),
        LIN_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_scale_bias_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      scale,
    scalar_t      bias,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_fused_linear_gen_cmd(
        src_addr,
        scale.u32,
        bias.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        true,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype),
        LIN_MAC,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_mm(
    local_addr_t output_addr,
    local_addr_t left_addr,
    local_addr_t right_addr,
    local_addr_t bias_addr,
    int left_rows,
    int left_cols,
    int right_cols,
    int left_cols_per_channel,
    int right_cols_per_channel,
    bool has_bias,
    bool result_add) {
    atomic_mm_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        false,
        !has_bias,
        result_add,
        0, //please add do_relu
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_mm_L_trans(
    local_addr_t output_addr,
    local_addr_t left_addr,
    local_addr_t right_addr,
    local_addr_t bias_addr,
    int left_rows,
    int left_cols,
    int right_cols,
    int left_cols_per_channel,
    int right_cols_per_channel,
    bool has_bias,
    bool result_add) {
    atomic_mm_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_cols,
        left_rows,
        true,
        false,
        !has_bias,
        result_add,
        0, //please add do_relu
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_mm_L_const(
    local_addr_t output_addr,
    local_addr_t right_addr,
    local_addr_t bias_addr,
    float C,
    int left_rows,
    int left_cols,
    int right_cols,
    int right_cols_per_channel,
    bool has_bias,
    bool result_add) {
    scalar_t C_ = {.f32 = C};
    atomic_mm_gen_cmd(
        C_.u32,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        16,
        DIV_UP(left_cols, 16),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        true,
        !has_bias,
        result_add,
        0, //please add do_relu
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_mm(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(left_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == PRECISION(right_dtype));
    scalar_t s = {.s8 = shift};
    if (PRECISION(left_dtype) != INT32) s.u32 = 0;
    int res_sign = SIGN(left_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        s.u32,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        true,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(left_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_mm_L_trans(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(left_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == PRECISION(right_dtype));
    scalar_t s = {.s8 = shift};
    if (PRECISION(left_dtype) != INT32) s.u32 = 0;
    int res_sign = SIGN(left_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        s.u32,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_cols,
        left_rows,
        true,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        true,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(left_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_mm_L_const(
    local_addr_t     output_addr,
    local_addr_t     right_addr,
    scalar_t         C,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              right_cols_per_channel,
    data_type_t      C_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(C_dtype) == PRECISION(right_dtype));
    scalar_t s = {.s8 = shift};
    if (PRECISION(C_dtype) != INT32) s.u32 = 0;
    int res_sign = SIGN(C_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        s.u32,
        tpu_eu_num(C_dtype),
        DIV_UP(left_cols, tpu_eu_num(C_dtype)),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        true,
        SIGN(C_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        true,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(C_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_pcs_mm(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(left_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == PRECISION(right_dtype));
    int bais_const = 0;
    if (PRECISION(left_dtype) != INT32) {
        shift_addr = 0;
        bais_const = 1;
    }
    int res_sign = SIGN(left_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        shift_addr,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        bais_const,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(left_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_pcs_mm_L_trans(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(left_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == PRECISION(right_dtype));
    int bais_const = 0;
    if (PRECISION(left_dtype) != INT32) {
        shift_addr = 0;
        bais_const = 1;
    }
    int res_sign = SIGN(left_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        shift_addr,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_cols,
        left_rows,
        true,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        bais_const,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(left_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_pcs_mm_L_const(
    local_addr_t     output_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              right_cols_per_channel,
    data_type_t      C_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(C_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(right_dtype));
    TPUKERNEL_ASSERT(PRECISION(C_dtype) == PRECISION(right_dtype));
    int bais_const = 0;
    if (PRECISION(C_dtype) != INT32) {
        shift_addr = 0;
        bais_const = 1;
    }
    int res_sign = SIGN(C_dtype) || SIGN(right_dtype);
    atomic_mm_fixed_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        shift_addr,
        tpu_eu_num(C_dtype),
        DIV_UP(left_cols, tpu_eu_num(C_dtype)),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        true,
        SIGN(C_dtype),
        SIGN(right_dtype),
        1,   //please change bias_sign
        res_sign, //please change res_sign
        bais_const,
        0, //please change add_result
        0, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        INT32, //Y_prec
        (PREC)PRECISION(C_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

static void fp8e4m3_saturate_conv_output(
    local_addr_t      addr,
    const dim4       *input_shape,
    const padding_t  *padding,
    const dim2       *dilation,
    const dim2       *kernel,
    const dim2       *stride,
    int              output_c) {
    scalar_t F8E4NAN = {.u8 = 0x7F};
    scalar_t F8E4MAX = {.u8 = 0x7E};
    scalar_t F8E4NNAN = {.u8 = 0xFF};
    scalar_t F8E4NMAX = {.u8 = 0xFE};
    const variable_t src0 = {
        .type=TENSOR,
        .context={.addr=addr}};
    const variable_t src1 = {
        .type=SCALAR,
        .context={.scalar=F8E4NAN}};
    const variable_t src2 = {
        .type=SCALAR,
        .context={.scalar=F8E4MAX}};
    const variable_t src3 = {
        .type=SCALAR,
        .context={.scalar=F8E4NNAN}};
    const variable_t src4 = {
        .type=SCALAR,
        .context={.scalar=F8E4NMAX}};

    int kh_ext = dilation->h * (kernel->h - 1) + 1;
    int kw_ext = dilation->w * (kernel->w - 1) + 1;
    int ih_ext = (input_shape->h - 1) + padding->top + padding->bottom + 1;
    int iw_ext = (input_shape->w - 1) + padding->right + padding->left + 1;
    int output_h = (ih_ext - kh_ext) / stride->h + 1;
    int output_w = (iw_ext - kw_ext) / stride->w + 1;
    const dim4 output_shape = {input_shape->n, output_c, output_h, output_w};

    tpu_bdc_equal_select(
        addr,
        &src0,
        &src1,
        &src2,
        &src0,
        &output_shape,
        DT_FP8E4M3,
        DT_FP8E4M3
        );
    tpu_bdc_equal_select(
        addr,
        &src0,
        &src3,
        &src4,
        &src0,
        &output_shape,
        DT_FP8E4M3,
        DT_FP8E4M3
        );
}

void tpu_bdc_fp_conv2d_with_rescale(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      rescale_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    bool              has_bias,
    bool              result_add,
    bool              do_relu,
    bool              do_rescale,
    bool              scale_const) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(output_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(input_dtype));
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == PRECISION(weight_dtype));
    TPUKERNEL_ASSERT(output_dtype == input_dtype || output_dtype == DT_FP32 || (input_dtype == DT_TF32 && output_dtype == DT_FP32) ||
                     (tpu_data_type_bits(input_dtype) == 8 && (output_dtype == DT_FP16 || tpu_data_type_bits(output_dtype) == 8)));
    TPUKERNEL_ASSERT(tpu_data_type_bits(input_dtype) == 8 || do_rescale == false);
    TPUKERNEL_ASSERT(bias_dtype == input_dtype || bias_dtype == DT_FP32 || (tpu_data_type_bits(input_dtype) == 8 && bias_dtype == DT_FP16));
    atomic_conv_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        0,
        do_rescale ? rescale_addr : 0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        false,
        result_add,
        NO_USE,
        (int *)input_stride,
        do_relu,
        (PREC)PRECISION(input_dtype),
        (PREC)PRECISION(output_dtype),
        (PREC)PRECISION(bias_dtype),
        FP8TYPE(input_dtype),
        FP8TYPE(weight_dtype),
        FP8TYPE(output_dtype),
        0,
        do_rescale,
        scale_const,
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
    if (output_dtype == DT_FP8E4M3) {
        fp8e4m3_saturate_conv_output(
            output_addr,
            input_shape,
            padding,
            dilation,
            kernel,
            stride,
            output_c);
    }
}

void tpu_bdc_fp_conv_bw(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      gradout_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    const dim4       *output_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              result_add
){
    atomic_conv_bw_gen_cmd(
        input_addr,
        gradout_addr,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_shape->c,
        output_shape->h,
        output_shape->w,
        kernel->h,
        kernel->w,
        0,
        0,
        dilation->h,
        dilation->w,
        stride->h,
        stride->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        result_add,
        NO_USE,
        (int *)input_stride,
        PAD_CONSTANT,
        (PREC)PRECISION(input_dtype),
        (PREC)PRECISION(output_dtype),
        (FP8_TYPE)(0),
        (FP8_TYPE)(0),
        (FP8_TYPE)(0),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add) {
    TPUKERNEL_ASSERT(!tpu_is_data_type_fp8(input_dtype));
    TPUKERNEL_ASSERT(!tpu_is_data_type_fp8(output_dtype));
    tpu_bdc_fp_conv2d_with_rescale(
        output_addr,
        input_addr,
        weight_addr,
        bias_addr,
        0,
        input_shape,
        input_stride,
        output_c,
        kernel,
        padding,
        stride,
        dilation,
        output_dtype,
        input_dtype,
        input_dtype,
        DT_FP32,
        has_bias,
        result_add,
        false,
        false,
        false);
}
void tpu_bdc_fp_conv2d_rescale_pc(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      rescale_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    bool              has_bias,
    bool              result_add,
    bool              do_relu) {
    tpu_bdc_fp_conv2d_with_rescale(
        output_addr,
        input_addr,
        weight_addr,
        bias_addr,
        rescale_addr,
        input_shape,
        input_stride,
        output_c,
        kernel,
        padding,
        stride,
        dilation,
        output_dtype,
        input_dtype,
        weight_dtype,
        bias_dtype,
        has_bias,
        result_add,
        do_relu,
        true,
        false);
}
void tpu_bdc_fp_conv2d_rescale_C(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    float             rescale,
    bool              has_bias,
    bool              result_add,
    bool              do_relu) {
    tpu_bdc_fp_conv2d_with_rescale(
        output_addr,
        input_addr,
        weight_addr,
        bias_addr,
        rescale,
        input_shape,
        input_stride,
        output_c,
        kernel,
        padding,
        stride,
        dilation,
        output_dtype,
        input_dtype,
        weight_dtype,
        bias_dtype,
        has_bias,
        result_add,
        do_relu,
        true,
        true);
}
void tpu_bdc_fp_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(output_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(input_dtype));
    TPUKERNEL_ASSERT(output_dtype == input_dtype || output_dtype == DT_FP32);
    atomic_conv_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        0,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        1,
        1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        true,
        result_add,
        0,
        (int *)input_stride,
        false,
        (PREC)PRECISION(input_dtype),
        (PREC)PRECISION(output_dtype),
        FP32,
        0,
        0,
        0,
        0,
        0,
        0,
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              kernel_is_const,
    bool              has_bias) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(output_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(input_dtype));
    TPUKERNEL_ASSERT(output_dtype == input_dtype);
    atomic_depthwise_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        0,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        1,
        1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        kernel_is_const,
        !has_bias,
        true,
        0,
        true,
        false,
        false,
        true,
        PRECISION(input_dtype),
        PRECISION(input_dtype),
        FP8TYPE(input_dtype), //for fp8
        FP8TYPE(input_dtype),
        FP8TYPE(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_conv2d_kernel_const(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      bias_addr,
    scalar_t          C,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(output_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(input_dtype));
    TPUKERNEL_ASSERT(output_dtype == input_dtype || output_dtype == DT_FP32);
    atomic_conv_gen_cmd(
        input_addr,
        C.u32,
        has_bias ? bias_addr : 0,
        0,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        !has_bias,
        true,
        false,
        result_add,
        NO_USE,
        (int *)input_stride,
        false,
        (PREC)PRECISION(input_dtype),
        (PREC)PRECISION(output_dtype),
        FP32,
        0,
        0,
        0,
        0,
        0,
        0,
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_asym_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    scalar_t          kzp_val,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT32 || PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(kzp_dtype) == INT8);
    if (!SIGN(input_dtype) && !SIGN(weight_dtype) && !SIGN(kzp_dtype) &&
            kzp_val.u8 == 0)
        TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(output_dtype));
    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        NO_USE,
        pad_val.u32,
        kzp_val.u32,
        NO_USE,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        true,
        true,
        true,
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        false,
        SIGN(output_dtype),
        (int *)input_stride,
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_asym_pc_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      kzp_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT32 || PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(kzp_dtype) == INT8);
    TPUKERNEL_ASSERT(tpu_is_data_type_signed_int(output_dtype));
    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        NO_USE,
        pad_addr,
        kzp_addr,
        NO_USE,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        true,
        false,
        false,
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        false,
        SIGN(output_dtype),
        (int *)input_stride,
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_asym_quant_conv2d_kernel_const(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    scalar_t          C,
    scalar_t          kzp_val,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT32 || PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(kzp_dtype) == INT8);
    if (!SIGN(input_dtype) && !SIGN(weight_dtype) && !SIGN(kzp_dtype) &&
            kzp_val.u8 == 0)
        TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(output_dtype));
    atomic_conv_quant_gen_cmd(
        input_addr,
        C.u32,
        NO_USE,
        pad_val.u32,
        kzp_val.u32,
        NO_USE,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        true,
        true,
        true,
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        false,
        SIGN(output_dtype),
        (int *)input_stride,
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_asym_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    scalar_t          kzp_val,
    scalar_t          pad_val,
    scalar_t          insert_val,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT32 || PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(kzp_dtype) == INT8);
    if (!SIGN(input_dtype) && !SIGN(weight_dtype) && !SIGN(kzp_dtype) &&
            kzp_val.u8 == 0)
        TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(output_dtype));
    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        NO_USE,
        pad_val.u32,
        kzp_val.u32,
        NO_USE,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        1, 1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        true,
        true,
        true,
        true,
        result_add,
        insert_val.u32,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        false,
        SIGN(output_dtype),
        (int *)input_stride,
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_asym_pc_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      kzp_addr,
    local_addr_t      pad_insert_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT32 || PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(kzp_dtype) == INT8);
    TPUKERNEL_ASSERT(tpu_is_data_type_unsigned_int(output_dtype));
    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        NO_USE,
        pad_insert_addr,
        kzp_addr,
        NO_USE,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        1, 1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        true,
        false,
        false,
        true,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        false,
        SIGN(output_dtype),
        (int *)input_stride,
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_sym_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT8 || PRECISION(output_dtype) == INT16);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        NO_USE,
        NO_USE,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        true,
        false,
        false,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_pc(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    local_addr_t requant_addr,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    bool has_bias,
    bool result_relu,
    bool has_requant,
    bool sym_range)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        NO_USE,
        NO_USE,
        requant_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        true,
        false,
        false,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        has_requant, // has_requant,
        false,      // requant_is_const,
        NO_USE,     //-rshift,
        NO_USE,     // output_zp,
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_C(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    int multiplier,
    char rshift,
    short output_zp,
    bool has_bias,
    bool result_relu,
    bool has_requant,
    bool sym_range)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        NO_USE,
        NO_USE,
        (unsigned int)multiplier,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        true,
        false,
        false,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        has_requant, // has_requant,
        true,       // requant_is_const,
        rshift,
        output_zp,
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_pc_asym_pc(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    local_addr_t kzp_addr,
    local_addr_t pad_addr,
    local_addr_t requant_addr,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    bool has_requant,
    bool has_bias,
    bool result_relu,
    bool result_add,
    bool sym_range,
    rounding_mode_t rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    if(!has_requant){
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }

    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_addr,
        kzp_addr,
        requant_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        false, // pad_is_const
        false, // kzp_is_const
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        true,   // has_requant,
        false,  // requant_is_const,
        NO_USE, //-rshift,
        NO_USE, // output_zp,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_C_asym_C(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    scalar_t kzp_val,
    scalar_t pad_val,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    int multiplier,
    char rshift,
    short output_zp,
    bool has_requant,
    bool has_bias,
    bool result_relu,
    bool result_add,
    bool sym_range,
    rounding_mode_t rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    if(!has_requant){
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }

    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        kzp_val.u32,
        (unsigned int)multiplier,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true, // pad_is_const
        true, // kzp_is_const
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        has_requant, // has_requant,
        true,        // requant_is_const,
        rshift,
        output_zp,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_pc_asym_C(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    local_addr_t requant_addr,
    scalar_t kzp_val,
    scalar_t pad_val,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    bool has_requant,
    bool has_bias,
    bool result_relu,
    bool result_add,
    bool sym_range,
    rounding_mode_t rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    if(!has_requant){
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        kzp_val.u32,
        requant_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true, // pad_is_const
        true, // kzp_is_const
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        has_requant, // has_requant,
        false,       // requant_is_const,
        NO_USE,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_conv2d_requant_C_asym_pc(
    local_addr_t output_addr,
    local_addr_t input_addr,
    local_addr_t weight_addr,
    local_addr_t bias_addr,
    local_addr_t kzp_addr,
    local_addr_t pad_addr,
    const dim4 *input_shape,
    const dim4 *input_stride,
    int output_c,
    const dim2 *kernel,
    const padding_t *padding,
    const dim2 *stride,
    const dim2 *dilation,
    data_type_t output_dtype,
    data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t bias_dtype,
    int multiplier,
    char rshift,
    short output_zp,
    bool has_requant,
    bool has_bias,
    bool result_relu,
    bool result_add,
    bool sym_range,
    rounding_mode_t rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    if(!has_requant){
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }

    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8 || PRECISION(input_dtype) == INT4);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8 || PRECISION(weight_dtype) == INT4);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);

    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_addr,
        kzp_addr,
        (unsigned int)multiplier,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        false, // pad_is_const
        false, // kzp_is_const
        false,
        result_add,
        0,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        sym_range,
        true,      // has_requant,
        true,      // requant_is_const,
        rshift,    //-rshift,
        output_zp, // output_zp,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_depthwise2d_requant_pc_asym_C(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      requant_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    bool              has_bias,
    bool              has_requant,
    bool              result_relu,
    bool              sym_range,
    rounding_mode_t   rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if(!has_requant){
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        requant_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        pad_val.s32,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        has_requant,       //  do_requant
        false,             //  requant_is_const
        NO_USE,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_depthwise2d_requant_C_sym_C(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    bool              sym_range,
    rounding_mode_t   rounding_mode)
{
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if(rshift != 0){
        if(result_relu)
            TPUKERNEL_ASSERT(output_dtype == DT_UINT8);
        else
            TPUKERNEL_ASSERT(output_dtype == DT_INT8);
    } else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        pad_val.s32,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        sym_range,
        rshift != 0,       //  do_requant
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_int8_sym_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT8 || PRECISION(output_dtype) == INT16);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);
    atomic_conv_quant_gen_cmd(
        input_addr,
        weight_addr,
        NO_USE,
        NO_USE,
        NO_USE,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_c,
        kernel->h,
        kernel->w,
        1, 1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        true,
        true,
        false,
        NO_USE,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        (int *)input_stride,
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(weight_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_max_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_max_min_pooling_gen_cmd(
        input_addr,
        pad_val.u32,
        output_addr,
        0xFFFFFFFF, //index addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        NO_USE,
        FP8TYPE(dtype),
        (PREC)PRECISION(dtype),
        NO_USE, //out_index_prec
        PAD_CONSTANT,
        0, //do relu
        PD_MAX_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_max_pool2d);
void tpu_bdc_fp_min_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_max_min_pooling_gen_cmd(
        input_addr,
        pad_val.u32,
        output_addr,
        0xFFFFFFFF, //index addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        NO_USE,
        FP8TYPE(dtype),
        (PREC)PRECISION(dtype),
        NO_USE, //out_index_prec
        PAD_CONSTANT,
        0, //do relu
        PD_MIN_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_max_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val) {
    NOT_SUPPORT(__func__);
}
void tpu_bdc_int_min_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val){
    NOT_SUPPORT(__func__);
}
void tpu_bdc_int8_max_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val) {
    TPUKERNEL_ASSERT(dtype == DT_INT8 || dtype == DT_UINT8);
    atomic_max_min_pooling_gen_cmd(
        input_addr,
        pad_val.u32,
        output_addr,
        0xFFFFFFFF, //index addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        NO_USE,
        SIGN(dtype),
        PRECISION(dtype),
        NO_USE, //out_index_prec
        PAD_CONSTANT,
        0, //do relu
        PD_MAX_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_min_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val) {
    TPUKERNEL_ASSERT(dtype == DT_INT8 || dtype == DT_UINT8);
    atomic_max_min_pooling_gen_cmd(
        input_addr,
        pad_val.u32,
        output_addr,
        0xFFFFFFFF, //index addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        NO_USE,
        SIGN(dtype),
        PRECISION(dtype),
        NO_USE, //out_index_prec
        PAD_CONSTANT,
        0, //do relu
        PD_MIN_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_ins_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *ins,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    unsigned char     scale,
    unsigned char     rshift) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(SIGN(output_dtype) == SIGN(input_dtype));
    ASSERT(rshift <= 128);
    atomic_avg_pooling_fixed_gen_cmd(
        input_addr,
        0,
        output_addr,
        0,   //rq_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        ins->h,
        ins->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        (int)scale,
        NO_USE,
        SIGN(input_dtype),
        SIGN(output_dtype),
        0, //kernel sign
        0, //do relu
        1, //do rq
        1, //rq_is_const
        0, //sym_range
        (int)1, //mul
        (s8)(-rshift), //shift,
        (s16)0, //yzp
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    unsigned char     scale,
    unsigned char     rshift) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(SIGN(output_dtype) == SIGN(input_dtype));
    ASSERT(rshift <= 128);
    atomic_avg_pooling_fixed_gen_cmd(
        input_addr,
        0,
        output_addr,
        0, //rq_addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        (int)scale,
        NO_USE,
        SIGN(input_dtype),
        SIGN(output_dtype),
        0, //kernel sign
        0, //do relu
        rshift != 0, //do rq
        1, //rq_is_const
        0, //sym_range
        (int)1, //mul
        (s8)(-rshift), //shift,
        (s16)0, //yzp
        ROUND_HALF_UP,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        pad_val.s32,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_depthwise2d_kernel_const(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      bias_addr,
    scalar_t          C,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        C.u32,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        !has_bias,
        true,
        pad_val.s32,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_pc_pad_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_addr,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0, 0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        false,
        NO_USE,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_pc_pad_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *insert,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(output_dtype));
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_addr,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        1, 1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        NO_USE,
        true,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *insert,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode) {
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(input_dtype) | SIGN(weight_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(input_dtype) == INT8);
    TPUKERNEL_ASSERT(PRECISION(weight_dtype) == INT8);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);
    atomic_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        pad_val.u32,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        1 ,1,
        insert->h,
        insert->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        pad_val.s32,
        true,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        result_relu,
        false,
        rshift != 0,
        true,
        -rshift,
        NO_USE,
        rounding_mode,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_conv2d_backward(
    local_addr_t     grad_wight_local_addr,
    local_addr_t     forward_input_local_addr,
    local_addr_t     grad_output_local_addr,
    local_addr_t     pad_ins_local_addr,
    const dim4      *forwrad_input_shape,
    const dim4      *forward_output_shape,
    // int              groups, // consider groups > 1
    const dim2      *forward_kernel,
    const dim2      *backward_insert,
    const padding_t *backward_padding,
    const dim2      *backward_stride,
    const dim2      *backward_dilation,
    const dim4      *input_stride,
    int              pad_ins_is_const,
    int              insert_const_val,
    int              pad_mode,
    data_type_t      input_dtype,
    data_type_t      grad_dtype
){
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(input_dtype) && tpu_is_data_type_fp(grad_dtype));
    TPUKERNEL_ASSERT(forwrad_input_shape->n == forward_output_shape->n);
    // for backward, part of input is not used, so we need to calculate the real ih and iw
    int kh_ext = backward_stride->h * (forward_kernel->h - 1) + 1;
    int kw_ext = backward_stride->w * (forward_kernel->w - 1) + 1;
    int ih_ext = forwrad_input_shape->h + backward_padding->top + backward_padding->bottom;
    int iw_ext = forwrad_input_shape->w + backward_padding->left + backward_padding->right;
    int real_ih = forwrad_input_shape->h - (ih_ext - kh_ext) % backward_dilation->h;
    int real_iw = forwrad_input_shape->w - (iw_ext - kw_ext) % backward_dilation->w;
    atomic_conv_bw_gen_cmd(
        forward_input_local_addr, //opad0
        grad_output_local_addr, //opad1
        pad_ins_local_addr, //opad2
        grad_wight_local_addr, //result
        forwrad_input_shape->n,
        forwrad_input_shape->c,
        // forwrad_input_shape->h,
        // forwrad_input_shape->w,
        real_ih,
        real_iw,
        forward_output_shape->c,
        forward_output_shape->h,
        forward_output_shape->w,
        forward_kernel->h,
        forward_kernel->w,
        backward_insert->h,
        backward_insert->w,
        backward_dilation->h,
        backward_dilation->w,
        backward_stride->h,
        backward_stride->w,
        backward_padding->top,
        backward_padding->bottom,
        backward_padding->left,
        backward_padding->right,
        pad_ins_is_const,
        false,
        insert_const_val,
        (int *)input_stride,
        pad_mode,
        (PREC)PRECISION(input_dtype),
        FP32,
        FP8TYPE(input_dtype),
        FP8TYPE(grad_dtype),
        NO_USE,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_avg_pool2d_ex(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    const dim2       *ins,
    data_type_t       in_dtype,
    data_type_t       out_dtype,
    scalar_t          scale) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(in_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(out_dtype));
    atomic_avg_pooling_gen_cmd(
        input_addr,
        0,
        NO_USE, //rq_addr for fp8
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        ins->h,
        ins->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        scale.s32,
        NO_USE,
        0, //do_relu
        NO_USE, // do_rq for fp8
        NO_USE, // rq_is_const for fp8
        NO_USE, // re_scale for fp8
        NO_USE, // sym_range for fp8
        PRECISION(in_dtype), // input_prec for fp8
        PRECISION(out_dtype), // output_prec for fp8
        FP8TYPE(in_dtype),
        NO_USE, // kernel fp8_type
        NO_USE, // output fp8 type
        PAD_CONSTANT,
        ROUND_HALF_TO_EVEN, //cast round_mode for fp8
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}


void tpu_bdc_fp_ins_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    const dim2       *ins,
    data_type_t       dtype,
    scalar_t          scale) {
    tpu_bdc_fp_avg_pool2d_ex(
        output_addr,
        input_addr,
        input_shape,
        kernel,
        padding,
        stride,
        dilation,
        ins,
        dtype,
        dtype,
        scale);
}

void tpu_bdc_fp_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          scale) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    dim2 ins = {0, 0};
    tpu_bdc_fp_ins_avg_pool2d(output_addr,
                              input_addr,
                              input_shape,
                              kernel,
                              padding,
                              stride,
                              dilation,
                              &ins,
                              dtype,
                              scale);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_avg_pool2d);

// only per-tensor fp8 for quantize
void tpu_bdc_fp8_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       input_dtype,
    data_type_t       output_dtype,
    scalar_t          scale,
    float             re_scale
) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp8(input_dtype));
    atomic_avg_pooling_gen_cmd(
        input_addr,
        0,
        NO_USE, //rq_addr for fp8
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        true,
        scale.s32,
        NO_USE,
        0,//do_relu
        tpu_is_data_type_fp8(output_dtype), // do_rq for fp8
        tpu_is_data_type_fp8(output_dtype), // rq_is_const for fp8
        NO_USE, // sym_range for fp8
        re_scale, // re_scale for fp8
        PRECISION(input_dtype), // input_prec for fp8
        PRECISION(output_dtype), // output_prec for fp8
        FP8TYPE(input_dtype),
        FP8TYPE(input_dtype), // kernel fp8_type
        FP8TYPE(output_dtype), // output fp8 type
        PAD_CONSTANT,
        ROUND_HALF_TO_EVEN, //cast round_mode for fp8
        MASTER_THREAD,
        BDC_NODE
    );
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_ins_depthwise(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *ins,
    const dim2       *dilation,
    data_type_t       dtype,
    bool              has_bias
) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_depthwise_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        0,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        ins->h,
        ins->w,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        NO_USE,
        false,
        false,
        false,
        false,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype), // for fp8
        FP8TYPE(dtype),
        FP8TYPE(dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_depthwise2d_with_scale(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      scale_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       out_dtype,
    data_type_t       in_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    float             scale,
    bool              has_bias,
    bool              do_relu,
    bool              do_rescale,
    bool              scale_is_const) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(in_dtype));
    if (tpu_is_data_type_fp8(in_dtype)) {
      TPUKERNEL_ASSERT(tpu_is_data_type_fp8(weight_dtype));
      TPUKERNEL_ASSERT(do_rescale || bias_dtype == out_dtype);
      TPUKERNEL_ASSERT(!do_rescale || (bias_dtype == DT_FP32 && (tpu_is_data_type_fp8(out_dtype) || out_dtype == DT_FP16)));
    } else {
      TPUKERNEL_ASSERT(bias_dtype == out_dtype);
      TPUKERNEL_ASSERT(in_dtype == out_dtype || out_dtype == DT_FP32);
    }
    atomic_depthwise_gen_cmd(
        input_addr,
        weight_addr,
        has_bias ? bias_addr : 0,
        0,
        scale_is_const ? scale : scale_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        kernel->h,
        kernel->w,
        stride->h,
        stride->w,
        0,
        0,
        dilation->h,
        dilation->w,
        padding->top,
        padding->bottom,
        padding->left,
        padding->right,
        false,
        !has_bias,
        true,
        NO_USE,
        false,
        do_relu,
        do_rescale,
        scale_is_const,
        PRECISION(in_dtype),
        PRECISION(out_dtype),
        FP8TYPE(in_dtype), // for fp8
        FP8TYPE(weight_dtype),
        FP8TYPE(out_dtype),
        PAD_CONSTANT,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
    if (out_dtype == DT_FP8E4M3) {
        fp8e4m3_saturate_conv_output(
            output_addr,
            input_shape,
            padding,
            dilation,
            kernel,
            stride,
            input_shape->c);
    }
}
void tpu_bdc_fp_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    bool              has_bias) {

    tpu_bdc_fp_depthwise2d_with_scale(
        output_addr,
        input_addr,
        weight_addr,
        bias_addr,
        0,
        input_shape,
        kernel,
        padding,
        stride,
        dilation,
        dtype,
        dtype,
        dtype,
        dtype,
        0.f,
        has_bias,
        false,
        false,
        false);
}

void tpu_bdc_fp_roi_max_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_roi_max_min_pooling_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        FP8TYPE(dtype),
        PRECISION(dtype),
        0, //do_relu
        PD_ROI_MAX_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_roi_min_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_roi_max_min_pooling_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        FP8TYPE(dtype),
        PRECISION(dtype),
        0, //do_relu
        PD_ROI_MIN_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_roi_avg_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val,
    scalar_t      scale) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_roi_avg_pooling_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        NO_USE, //rq_addr for fp8
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        scale.s32,
        except_val.s32,
        NO_USE, //do_rq for fp8
        NO_USE, //rq_is_const for fp8
        NO_USE, //rq scale for fp8
        NO_USE, //sym_range for fp8
        0, //do_relu
        FP8TYPE(dtype),
        NO_USE, //kernel fp8 type
        NO_USE, //res fp8 type
        PRECISION(dtype),
        PRECISION(dtype), //out prec
        ROUND_HALF_TO_EVEN, //cast round_mode for fp8
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_roi_depthwise2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  weight_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    atomic_roi_depthwise_gen_cmd(
        input_addr,
        weight_addr,
        roi_addr,
        0,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        false,
        false,
        false,
        false,
        true,
        PRECISION(dtype),
        PRECISION(dtype),
        FP8TYPE(dtype), // for fp8
        FP8TYPE(dtype), // for fp8
        FP8TYPE(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_roi_max_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(dtype == DT_INT8 || dtype == DT_UINT8);
    atomic_roi_max_min_pooling_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        SIGN(dtype),
        PRECISION(dtype),
        0, //do_relu
        PD_ROI_MAX_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_roi_min_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(dtype == DT_INT8 || dtype == DT_UINT8);
    atomic_roi_max_min_pooling_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        SIGN(dtype),
        PRECISION(dtype),
        0, //do_relu
        PD_ROI_MIN_POOLING,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_roi_avg_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   output_dtype,
    data_type_t   input_dtype,
    scalar_t      except_val,
    scalar_t      scale) {
    TPUKERNEL_ASSERT(PRECISION(output_dtype) == INT8 ||
                    PRECISION(output_dtype) == INT16 ||
                    PRECISION(output_dtype) == INT32);
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    atomic_roi_avg_pooling_quant_gen_cmd(
        input_addr,
        roi_addr,
        output_addr,
        0, //rq_addr
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        scale.s32,
        except_val.s32,
        SIGN(input_dtype),
        SIGN(output_dtype),
        0, //kernel_sign
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        0, //do_relu
        0, //do_rq
        0, //rq_is_const
        0, //mul
        0, //shift
        0, //yzp
        0, //sym_range
        0, //round_mode
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_roi_depthwise2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  weight_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   output_dtype,
    data_type_t   input_dtype,
    data_type_t   weight_dtype,
    scalar_t      except_val) {
    TPUKERNEL_ASSERT(PRECISION(output_dtype) == INT8 ||
                    PRECISION(output_dtype) == INT16);
    TPUKERNEL_ASSERT(input_dtype == DT_INT8 || input_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(weight_dtype == DT_INT8 || weight_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(
        SIGN(output_dtype) == (SIGN(input_dtype) | SIGN(weight_dtype)));
    atomic_roi_depthwise_quant_gen_cmd(
        input_addr,
        weight_addr,
        roi_addr,
        1,
        output_addr,
        input_shape->n,
        input_shape->c,
        input_shape->h,
        input_shape->w,
        output_w,
        kernel->h,
        kernel->w,
        except_val.s32,
        false,
        false,
        SIGN(input_dtype),
        SIGN(weight_dtype),
        SIGN(output_dtype),
        false,
        false,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        NO_USE,
        PRECISION(input_dtype),
        PRECISION(output_dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_relu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    scalar_t C = {.u32 = 0};
    tpu_bdc_max_C(
        dst_addr,
        src_addr,
        C,
        shape,
        dst_stride,
        src_stride,
        dtype);
}
void tpu_bdc_prelu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      alpha,
    const dim4   *shape,
    data_type_t   dtype) {
    if (dtype == DT_UINT32 || dtype == DT_UINT16 || dtype == DT_UINT8)
        tpu_bdc_cpy(
            dst_addr,
            src_addr,
            shape,
            NULL,
            NULL,
            dtype);
    else {
        if (tpu_is_data_type_fp(dtype))
            tpu_bdc_fp_mul_C(
                dst_addr,
                src_addr,
                alpha,
                shape,
                NULL,
                NULL,
                dtype);
        else
            tpu_bdc_int_mul_C(
                dst_addr,
                src_addr,
                alpha,
                shape,
                NULL,
                NULL,
                dtype,
                dtype,
                dtype,
                0,
                NO_USE,
                false);
        variable_t dst = {.type = TENSOR, .context = {.addr = dst_addr}};
        variable_t src = {.type = TENSOR, .context = {.addr = src_addr}};
        variable_t C = {.type = SCALAR, .context = {.scalar = {.u32 = 0}}};
        tpu_bdc_greater_select(
            dst_addr,
            &src,
            &C,
            &src,
            &dst,
            shape,
            dtype,
            dtype);
    }
}
void tpu_bdc_fp32_elu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    float         alpha,
    const dim4   *shape) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    tpu_bdc_fp32_expm1(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    if (alpha != 1.f) {
        scalar_t C = {.f32 = alpha};
        tpu_bdc_fp_mul_C(
            dst_addr,
            dst_addr,
            C,
            shape,
            NULL,
            NULL,
            DT_FP32);
    }
    variable_t dst = {.type = TENSOR, .context = {.addr = dst_addr}};
    variable_t src = {.type = TENSOR, .context = {.addr = src_addr}};
    variable_t Z = {.type = SCALAR, .context = {.scalar = {.u32 = 0}}};
    tpu_bdc_greater_select(
        dst_addr,
        &src,
        &Z,
        &src,
        &dst,
        shape,
        DT_FP32,
        DT_FP32);
}

static void map_fp32_inf_to_zero(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape) {
    const scalar_t C_inf = {.u32 = 0x7f800000};
    const scalar_t C_0 = {0};
    const variable_t lhs = {.type = TENSOR, .context = {.addr = src_addr}};
    const variable_t rhs = {.type = SCALAR, .context = {.scalar = C_inf}};
    const variable_t tbrn = {.type = SCALAR, .context = {.scalar = C_0}};
    const variable_t fbrn = {.type = TENSOR, .context = {.addr = src_addr}};
    tpu_bdc_equal_select(
        dst_addr,
        &lhs,
        &rhs,
        &tbrn,
        &fbrn,
        shape,
        DT_UINT32,
        DT_UINT32);
}
void tpu_bdc_fp_tunable_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype,
    int           num_iter) {
    TPUKERNEL_ASSERT(IS_FLOAT(dtype));
    atomic_sfu_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        num_iter,
        SFU_RSQ,
        NO_USE,
        PRECISION(dtype),
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_tunable_rsqrt);

void tpu_bdc_fp32_tunable_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    int           num_iter) {
    tpu_bdc_fp_tunable_rsqrt(
        dst_addr,
        src_addr,
        shape,
        DT_FP32,
        num_iter);
    CHECK_BDC_OVERFLOW;
}

// void tpu_bdc_fp_tunable_rsqrt(
//     local_addr_t  dst_addr,
//     local_addr_t  src_addr,
//     const dim4   *shape,
//     int           num_iter) {
//     _tpu_bdc_fp_tunable_rsqrt(
//         dst_addr,
//         src_addr,
//         shape,
//         num_iter);
//     handle pitfall at +inf
//     map_fp32_inf_to_zero(
//         dst_addr,
//         dst_addr,
//         shape);
// }

void tpu_bdc_fp_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    tpu_bdc_fp_tunable_rsqrt(
        dst_addr,
        src_addr,
        shape,
        dtype,
        rsqrt_iter_num);
}

void tpu_bdc_fp32_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape) {
    tpu_bdc_fp_rsqrt(
        dst_addr,
        src_addr,
        shape,
        DT_FP32);
}

void tpu_bdc_fp_tunable_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype,
    int           num_iter) {
    TPUKERNEL_ASSERT(dst_addr != src_addr);
    tpu_bdc_fp_tunable_rsqrt(
        dst_addr,
        src_addr,
        shape,
        dtype,
        num_iter);
    scalar_t zero_value = {.u32=0}; // +0
    const variable_t src = {.type = TENSOR, .context = {.addr = src_addr}};
    const variable_t zero = {.type = SCALAR, .context = {.scalar = zero_value}};
    const variable_t dst = {.type = TENSOR, .context = {.addr = dst_addr}};
    tpu_bdc_equal_select(dst_addr, &src, &zero, &zero, &dst, shape, dtype, dtype);
    scalar_t neg_zero_value = {.u32=0x80000000}; // -0
    if(dtype != DT_FP32) neg_zero_value.u32=0x8000;
    const variable_t neg_zero = {.type = SCALAR, .context = {.scalar = neg_zero_value}};
    tpu_bdc_equal_select(dst_addr, &src, &neg_zero, &zero, &dst, shape, dtype, dtype);
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}

void tpu_bdc_fp_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    tpu_bdc_fp_tunable_sqrt(
        dst_addr,
        src_addr,
        shape,
        dtype,
        rsqrt_iter_num);
}

void tpu_bdc_fp32_tunable_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    int           num_iter){
    tpu_bdc_fp_tunable_sqrt(
        dst_addr,
        src_addr,
        shape,
        DT_FP32,
        num_iter);
}

void tpu_bdc_fp32_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape) {
    tpu_bdc_fp_sqrt(
        dst_addr,
        src_addr,
        shape,
        DT_FP32);
}

void tpu_bdc_fp_square(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    tpu_bdc_fp_mul(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        src_stride,
        dtype);
}
void tpu_bdc_fp_taylor(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    int           num,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
    //TPUKERNEL_ASSERT(coeff_addr < LOCAL_MEM_SIZE);
    atomic_sfu_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        num,
        SFU_TAYLOR_4X, // ROBUST CHOICE: SFU_TAYLOR
        coeff_addr + tpu_npu_index(dst_addr) * LOCAL_MEM_SIZE,
        PRECISION(dtype),
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_taylor);
void tpu_bdc_fp32_exp(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    tpu_bdc_fp_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        DT_FP32);
#if 0
    TPUKERNEL_ASSERT(dst_addr != work0_addr && dst_addr != work1_addr &&
                    work0_addr != work1_addr);
    scalar_t C;
    // DST = MAX(SRC, -103)
    C.f32 = -103.f;
    tpu_bdc_max_C(
        dst_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = MIN(DST, 88)
    C.f32 = 88.f;
    tpu_bdc_min_C(
        work0_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = ROUND(WORK0)
    tpu_bdc_fp_round(
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        DT_FP32,
        RM_HALF_AWAY_FROM_ZERO);
    // WORK0 = WORK0 - DST
    tpu_bdc_fp_sub(
        work0_addr,
        work0_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK1 = INT(DST)
    tpu_bdc_cast(
        work1_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        DT_INT16,
        DT_FP32,
        RM_TOWARDS_ZERO);
    // DST = WORK1 - (-103)
    C.s16 = -103;
    tpu_bdc_int_sub_C(
        dst_addr,
        work1_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_INT16,
        DT_INT16,
        DT_INT16,
        0,
        NO_USE,
        false);
    // WORK1 = LOOKUP(DST, TABLE)
    tpu_bdc_table_lookup(
        work1_addr,
        dst_addr,
        table_addr,
        shape,
        88 - (-103) + 1,
        DT_FP32,
        DT_UINT16);
    // WORK0 = TAYLOR(WORK0)
    tpu_bdc_fp_taylor(
        work0_addr,
        work0_addr,
        coeff_addr,
        shape,
        sfu_taylor_exp_len,
        DT_FP32);
    // DST = WORK0 * WORK1;
    tpu_bdc_fp_mul(
        dst_addr,
        work0_addr,
        work1_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
#endif
}
void tpu_bdc_fp_exp(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(dst_addr != work0_addr && dst_addr != work1_addr &&
                     work0_addr != work1_addr);
    /** fast exp: x = N * ln2 + rem
     * 1. N = (int)(x / ln2)
     * 2. rem = x - N * ln2
     * 3. exp(x) = exp(N * ln2 + rem) = 2^N * exp(rem)
     * 4. use taylor to compute exp(rem)
     * 5. 2.f ^ N = (float)((2 + 127) << 23)
     */
    local_addr_t buffer0_addr = src_addr == work1_addr ? work0_addr : work1_addr;
    local_addr_t buffer1_addr =
        src_addr == dst_addr
            ? (buffer0_addr == work0_addr ? work1_addr : work0_addr)
            : dst_addr;
    local_addr_t buffer2_addr = buffer1_addr == dst_addr ?
            (buffer0_addr == work0_addr ?
                work1_addr : work0_addr) : dst_addr;

    /// BUFFER0 = MAX(min_C, src)
    /// process -inf
    scalar_t min_C = {.u32 = 0};
    if (dtype == DT_FP32) {
        min_C.f32 = -3.40282 * 1e35;
    } else if (dtype == DT_FP16) {
        min_C.f32 = -45403;
        min_C.f16.bits = tpu_cast(min_C, dtype, DT_FP32, RM_HALF_TO_EVEN).f16.bits;
    } else {
        min_C.bf16.bits = 0xff7f;
    }
    tpu_bdc_max_C(
        buffer2_addr,
        src_addr,
        min_C,
        shape,
        NULL,
        NULL,
        dtype);

    if (dtype == DT_FP16){
        scalar_t max_C = {.u32 = 0};
        max_C.f32 = 45403;
        max_C.f16.bits = tpu_cast(max_C, dtype, DT_FP32, RM_HALF_TO_EVEN).f16.bits;
        tpu_bdc_min_C(
            buffer1_addr,
            buffer2_addr,
            max_C,
            shape,
            NULL,
            NULL,
            dtype);
    }
    /// BUFFER0 = BUFFER2 / ln2
    scalar_t C = {.f32 = 1.4426950f};
    tpu_bdc_fp_mul_C(
        buffer0_addr,
        (dtype == DT_FP16) ? buffer1_addr : buffer2_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);

    /// BUFFER1 = floor(BUFFER0)
    tpu_bdc_fp_floor(
        buffer1_addr,
        buffer0_addr,
        shape,
        NULL,
        NULL,
        dtype);

    /// BUFFER0 = BUFFER1 * ln2
    C.f32 = 0.69314718f;
    tpu_bdc_fp_mul_C(
        buffer0_addr,
        buffer1_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);

    /// BUFFER0 = BUFFER2 - BUFFER0
    tpu_bdc_fp_sub(
        buffer0_addr,
        buffer2_addr,
        buffer0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);

    /// BUFFER2 = int(BUFFER1)
    const data_type_t mdtype = (dtype == DT_FP16 || dtype == DT_BFP16) ? DT_INT8 : DT_INT16;
    tpu_bdc_cast(
        buffer2_addr,
        buffer1_addr,
        shape,
        NULL,
        NULL,
        mdtype,
        dtype,
        RM_HALF_AWAY_FROM_ZERO);

    /// BUFFER1 = min(BUFFER2, (2^8 - 1) - 127)
    /// and use (2^8 - 1) - 127 - 1 to avoid nan
    C.s16 = dtype == DT_FP16 ? 15 : 127;
    tpu_bdc_min_C(
        buffer1_addr,
        buffer2_addr,
        C,
        shape,
        NULL,
        NULL,
        mdtype);

    // /// BUFFER2 = max(BUFFER1, 0 - 127)
    C.s16 = -C.s16;
    tpu_bdc_max_C(
        buffer2_addr,
        buffer1_addr,
        C,
        shape,
        NULL,
        NULL,
        mdtype);

    /// BUFFER1 = (BUFFER2 + 127) << 23
    C.s16 = -C.s16;
    tpu_bdc_int_add_C(
        buffer1_addr,
        buffer2_addr,
        C,
        shape,
        NULL,
        NULL,
        dtype == DT_FP32 ? DT_INT32 : DT_INT16,
        mdtype,
        DT_INT16,
        dtype == DT_FP32 ? 23 : (dtype == DT_FP16 ? 10 : 7), // matissa
        RM_HALF_AWAY_FROM_ZERO,
        true);

    /// BUFFER2 = taylor(BUFFER0)
    tpu_bdc_fp_taylor(
        buffer2_addr,
        buffer0_addr,
        coeff_addr,
        shape,
        sfu_taylor_exp_len(dtype),
        dtype);

    /// DST = BUFFER1 * BUFFER2
    tpu_bdc_fp_mul(
        dst_addr,
        buffer2_addr,
        buffer1_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_expm1(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    tpu_bdc_fp32_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_sub_C(
        dst_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp_expm1(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    tpu_bdc_fp_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_sub_C(
        dst_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_log(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    tpu_bdc_fp_log(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        DT_FP32);
}
void tpu_bdc_fp_log(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(dst_addr != work_addr);
    bool same_addr = false;
    if (dst_addr == src_addr) {
        local_addr_t tmp = work_addr;
        work_addr = dst_addr;
        dst_addr = tmp;
        same_addr = true;
    }
    // WORK = MAX(SRC, 1.175494351e-38)
    scalar_t max_C = {.u32 = 0};
    if (dtype == DT_FP16) {
        // min normal
        max_C.f16.bits = 1 << 10;
    } else if (dtype == DT_BFP16) {
        // min normal
        max_C.bf16.bits = 1 << 7;
    } else if (dtype == DT_FP32) {
        max_C.f32 = 1.175494351e-38;
    }
    tpu_bdc_max_C(
        work_addr,
        src_addr,
        max_C,
        shape,
        NULL,
        NULL,
        dtype);
    // DST = NORMB(SRC) = exp
    data_type_t exp_dtype = dtype == DT_FP32 ? DT_INT32 : DT_INT16;
    tpu_bdc_fp_exponent_part(
        dst_addr,
        work_addr,
        shape,
        exp_dtype,
        dtype);
    // DST = -DST << frac
    scalar_t C = {.s32 = dtype == DT_FP16 ? 30 : 254};
    char frac = dtype == DT_FP32 ? 23 : (dtype == DT_FP16 ? 10 : 7);
    tpu_bdc_int_C_sub(
        dst_addr,
        dst_addr,
        tpu_cast(C, exp_dtype, DT_INT32, RM_HALF_UP),
        shape,
        NULL,
        NULL,
        exp_dtype,
        exp_dtype,
        exp_dtype,
        frac,
        RM_HALF_AWAY_FROM_ZERO,
        true);
    // DST = WORK * DST
    tpu_bdc_fp_mul(
        dst_addr,
        work_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = DST - 1
    C.f32 = 1.f;
    tpu_bdc_fp_sub_C(
        dst_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = TAYLOR(DST)
    tpu_bdc_fp_taylor(
        dst_addr,
        dst_addr,
        coeff_addr,
        shape,
        sfu_taylor_log_len(dtype),
        dtype);
    // process 0: DST = DST == 0 ? -inf : DST
    variable_t src = {.type = TENSOR, .context = {.addr = src_addr}};
    variable_t src0 = {.type = SCALAR, .context = {.scalar = {.u32 = 0}}};
    variable_t C0 = {.type = SCALAR, .context = {.scalar = {.u32 = 0xff800000}}};
    variable_t C1 = {.type = TENSOR, .context = {.addr = dst_addr}};
    tpu_bdc_equal_select(
        dst_addr,
        &src,
        &src0,
        &C0,
        &C1,
        shape,
        dtype,
        dtype);
    // WORK = NORMB(SRC)
    tpu_bdc_fp_exponent_part(
        work_addr,
        work_addr,
        shape,
        exp_dtype,
        dtype);
    // WORK = WORK - 127
    C.s32 = dtype == DT_FP16 ? 15 : 127;
    tpu_bdc_int_sub_C(
        work_addr,
        work_addr,
        tpu_cast(C, exp_dtype, DT_INT32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        exp_dtype,
        exp_dtype,
        exp_dtype,
        0,
        RM_HALF_AWAY_FROM_ZERO,
        true);
    // WORK = FP32(WORK)
    tpu_bdc_cast(
        work_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        dtype,
        exp_dtype,
        RM_TOWARDS_ZERO);
    // WORK = WORK * LOG2
    scalar_t log2 = {.f32 = 0.69314718056f};
    tpu_bdc_fp_mul_C(
        work_addr,
        work_addr,
        tpu_cast(log2, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    if (same_addr)
        // WORK = DST + WORK
        tpu_bdc_fp_add(
            work_addr,
            dst_addr,
            work_addr,
            shape,
            NULL,
            NULL,
            NULL,
            dtype);
    else
        // DST = DST + WORK
        tpu_bdc_fp_add(
            dst_addr,
            dst_addr,
            work_addr,
            shape,
            NULL,
            NULL,
            NULL,
            dtype);
}
void tpu_bdc_fp32_logx(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    float         x) {
    TPUKERNEL_ASSERT(x > 0);
    tpu_bdc_fp32_log(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape);
    scalar_t C = {.f32 = 1. / log(x)};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_log1p(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_log(
        dst_addr,
        work_addr,
        work_addr,
        coeff_addr,
        shape);
}
void tpu_bdc_table_lookup(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    int           len,
    data_type_t   dst_dtype,
    data_type_t   src_dtype) {
    //TPUKERNEL_ASSERT(table_addr < LOCAL_MEM_SIZE);
    dim4 shape_flatten = {
        .n = shape->n, .c = shape->c, .h = 1, .w = shape->h * shape->w
    };
    tpu_bdc_batch_bcast_w_gather(
        dst_addr,
        table_addr + tpu_npu_index(dst_addr) * LOCAL_MEM_SIZE,
        src_addr,
        &shape_flatten,
        len,
        dst_dtype,
        src_dtype,
        true);
}
void tpu_bdc_fp_exponent_part(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dst_dtype,
    data_type_t   src_dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(src_dtype));
    if (src_dtype == DT_FP32)
        TPUKERNEL_ASSERT(dst_dtype == DT_FP32 || dst_dtype == DT_INT32);
    else if (src_dtype == DT_FP16)
        TPUKERNEL_ASSERT(dst_dtype == DT_FP16 || dst_dtype == DT_INT16);
    else if (src_dtype == DT_BFP16)
        TPUKERNEL_ASSERT(dst_dtype == DT_BFP16 || dst_dtype == DT_INT16);
    atomic_sfu_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NO_USE,
        SFU_NORM,
        NO_USE,
        PRECISION(dst_dtype),
        PRECISION(src_dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_npu_bcast_from_static(
    local_addr_t   dst_addr,
    static_addr_t  src_addr,
    int            npu_num,
    int            len,
    data_type_t    dtype) {
    atomic_static_broad_gen_cmd(
        src_addr,
        dst_addr,
        npu_num,
        len,
        0xffffffffffffffff,
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_npu_distribute_from_static(
    local_addr_t   dst_addr,
    static_addr_t  src_addr,
    int            len,
    data_type_t    dtype) {
    //TPUKERNEL_ASSERT(dst_addr < LOCAL_MEM_SIZE);
    atomic_static_distribute_gen_cmd(
        src_addr,
        dst_addr,
        len,
        0xffffffffffffffff,
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_arithmetic_sequence_bcast(
    local_addr_t  dst_addr,
    int           npu_num,
    int           start,
    int           step,
    int           num) {
    TPUKERNEL_ASSERT(num > 0);
    dim4 shape = {.n = 1, .c = npu_num, .h = 1, .w = num};
    tpu_bdc_npu_bcast_from_static(
        dst_addr,
        SERIAL_NUMBER_OFFSET,
        shape.c,
        MIN(NPU_NUM, num),
        DT_INT32);
    int num_remained = MIN(NPU_NUM, num);
    while (num_remained < num)
    {
        scalar_t C = {.s32 = num_remained};
        shape.w = MIN(num_remained, num - num_remained);
        tpu_bdc_int_add_C(
            dst_addr + num_remained * sizeof(int),
            dst_addr,
            C,
            &shape,
            NULL,
            NULL,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            NO_USE,
            false);
        num_remained += shape.w;
    }
    shape.w = num;
    if (step != 1) {
        scalar_t C = {.s32 = step};
        tpu_bdc_int_mul_C(
            dst_addr,
            dst_addr,
            C,
            &shape,
            NULL,
            NULL,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            NO_USE,
            false);
    }
    if (start != 0) {
        scalar_t C = {.s32 = start};
        tpu_bdc_int_add_C(
            dst_addr,
            dst_addr,
            C,
            &shape,
            NULL,
            NULL,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            NO_USE,
            false);
    }
}

void tpu_bdc_arithmetic_sequence_distribute_general(
    local_addr_t  dst_addr, // 64-byte aligned
    int           start,
    int           step,
    int           num,
    int           aligned) {
    TPUKERNEL_ASSERT(num > 0);
    int secs = DIV_UP(num, NPU_NUM);
    dim4 shape = {.n = 1, .c = NPU_NUM, .h = 1, .w = 1};
    dim4 stride;
    if(aligned) {
      tpu_aligned_stride(&stride, 0, &shape, DT_INT32);
    } else {
      tpu_compact_stride(&stride, 0, &shape);
    }
    int num_remained = num;
    for (int i = 0; i < secs; ++i) {
        shape.c = MIN(num_remained, NPU_NUM);
        if (i == 0)
            tpu_bdc_npu_distribute_from_static(
                dst_addr,
                SERIAL_NUMBER_OFFSET,
                shape.c,
                DT_INT32);
        else {
            scalar_t C = {.s32 = NPU_NUM * i};
            tpu_bdc_int_add_C(
                dst_addr + i * stride.c * sizeof(int),
                dst_addr,
                C,
                &shape,
                &stride,
                &stride,
                DT_INT32,
                DT_INT32,
                DT_INT32,
                0,
                NO_USE,
                false);
        }
        num_remained -= shape.c;
    }
    TPUKERNEL_ASSERT(num_remained == 0);
    shape.c = num;
    if (step != 1) {
        scalar_t C = {.s32 = step};
        tpu_bdc_int_mul_C(
            dst_addr,
            dst_addr,
            C,
            &shape,
            &stride,
            &stride,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            NO_USE,
            false);
    }
    if (start != 0) {
        scalar_t C = {.s32 = start};
        tpu_bdc_int_add_C(
            dst_addr,
            dst_addr,
            C,
            &shape,
            &stride,
            &stride,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            NO_USE,
            false);
    }
}
void tpu_bdc_arithmetic_sequence_distribute(
    local_addr_t  dst_addr,
    int           start,
    int           step,
    int           num) {
  tpu_bdc_arithmetic_sequence_distribute_general(dst_addr, start, step, num, 0);
}

void tpu_bdc_arithmetic_sequence_distribute_aligned(
    local_addr_t  dst_addr,
    int           start,
    int           step,
    int           num) {
  tpu_bdc_arithmetic_sequence_distribute_general(dst_addr, start, step, num, 1);
}


void tpu_bdc_generate_arithmetic_sequence(
    local_addr_t dst_addr,
    int          num,
    data_type_t  dtype
    )
{
    TPUKERNEL_ASSERT(dtype == DT_UINT32 || dtype == DT_UINT16);
    const int type_size = tpu_data_type_size(dtype);
    const int secs = DIV_UP(num, NPU_NUM);
    dim4 shape = {1, 1, 1, num};
    int num_remained = num;
    for (int i = 0; i < secs; ++i)
    {
        shape.w = MIN(num_remained, NPU_NUM);
        if (i == 0)
        {
            tpu_bdc_npu_bcast_from_static(
                dst_addr,
                SERIAL_NUMBER_OFFSET,
                1,
                NPU_NUM,
                DT_INT32);
            if (dtype == DT_UINT16)
            {
                dim4 istr = {2 * NPU_NUM, 2 * NPU_NUM, 2, 2};
                dim4 ostr = {NPU_NUM, NPU_NUM, 1, 1};
                tpu_bdc_cpy(
                    dst_addr,
                    dst_addr,
                    &shape,
                    &ostr,
                    &istr,
                    dtype
                    );
            }
        }
        else
        {
            scalar_t C;
            if (dtype == DT_UINT32)
                C.u32 = NPU_NUM * i;
            else
                C.u16 = NPU_NUM * i;
            tpu_bdc_int_add_C(
                dst_addr + i * NPU_NUM * type_size,
                dst_addr,
                C,
                &shape,
                NULL,
                NULL,
                dtype,
                dtype,
                dtype,
                0,
                NO_USE,
                false);
        }
        num_remained -= shape.w;
    }
}

void tpu_bdc_arithmetic_sequence_general(
    local_addr_t  dst_addr,
    local_addr_t  buffer_addr, // size = sizeof(int32)
    int           npu_num,
    int           start,
    int           step,
    int           num)
{
    TPUKERNEL_ASSERT(tpu_npu_index(dst_addr) == 0);
    tpu_bdc_arithmetic_sequence_bcast(
        dst_addr,
        MIN(npu_num, NPU_NUM),
        start,
        step,
        num);
    tpu_bdc_arithmetic_sequence_distribute(
        buffer_addr,
        0,
        num * step,
        MIN(npu_num, NPU_NUM));
    dim4 shape = {1, MIN(NPU_NUM, npu_num), 1, num};
    dim4 src1_stride = {1, 1, 0, 0};
    tpu_bdc_int_add(
        dst_addr,
        dst_addr,
        buffer_addr,
        &shape,
        NULL,
        NULL,
        &src1_stride,
        DT_INT32,
        DT_INT32,
        DT_INT32,
        0,
        RM_HALF_AWAY_FROM_ZERO,
        true);
    for (int csecs = 1; csecs < DIV_UP(npu_num, NPU_NUM); ++csecs) {
        shape.c = MIN(NPU_NUM, npu_num - csecs * NPU_NUM);
        scalar_t C = {.s32 = csecs * step * num * NPU_NUM};
        tpu_bdc_int_add_C(
            dst_addr + ALIGN(num, tpu_eu_num(DT_INT32)) * csecs * sizeof(int),
            dst_addr,
            C,
            &shape,
            NULL,
            NULL,
            DT_INT32,
            DT_INT32,
            DT_INT32,
            0,
            RM_HALF_AWAY_FROM_ZERO,
            true);
    }
}

void tpu_bdc_load_fp32_exp_coeff(local_addr_t coeff_addr) {
    tpu_bdc_load_fp_exp_coeff(coeff_addr, DT_FP32);
}
void tpu_bdc_load_fp32_exp_table(local_addr_t table_addr) {
    //TPUKERNEL_ASSERT(table_addr < LOCAL_MEM_SIZE);
// Useless. Exp does not need table now
#if 0
    tpu_bdc_npu_bcast_from_static(
        table_addr,
        EXP_TABLE_OFFSET,
        NPU_NUM,
        192,
        DT_FP32);
#endif
}
void tpu_bdc_load_fp32_log_coeff(local_addr_t coeff_addr) {
    tpu_bdc_load_fp_log_coeff(coeff_addr, DT_FP32);
}
void tpu_bdc_load_fp32_erf_coeff(local_addr_t coeff_addr) {
    //TPUKERNEL_ASSERT(coeff_addr < LOCAL_MEM_SIZE);
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        ERF_TAYLOR_OFFSET,
        NPU_NUM,
        10,
        DT_FP32);
}
void tpu_bdc_load_fp_sin_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    //TPUKERNEL_ASSERT(coeff_addr < LOCAL_MEM_SIZE);
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        dtype == DT_FP32 ? SIN_TAYLOR_OFFSET : dtype == DT_FP16 ? SIN_FP16_TAYLOR_OFFSET : SIN_BFP16_TAYLOR_OFFSET,
        NPU_NUM,
        dtype == DT_FP32 ? 32 : 16,
        dtype);
}
void tpu_bdc_load_fp_cos_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    //TPUKERNEL_ASSERT(coeff_addr < LOCAL_MEM_SIZE);
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        dtype == DT_FP32 ? COS_TAYLOR_OFFSET : dtype == DT_FP16 ? COS_FP16_TAYLOR_OFFSET : COS_BFP16_TAYLOR_OFFSET,
        NPU_NUM,
        dtype == DT_FP32 ? 32 : 16,
        dtype);
}
void tpu_bdc_load_fp32_sin_coeff(local_addr_t coeff_addr) {
    tpu_bdc_load_fp_sin_coeff(coeff_addr, DT_FP32);
}
void tpu_bdc_load_fp32_cos_coeff(local_addr_t coeff_addr) {
    tpu_bdc_load_fp_cos_coeff(coeff_addr, DT_FP32);
}
void tpu_bdc_load_fp32_tan_coeff(local_addr_t coeff_addr) {
    //TPUKERNEL_ASSERT(coeff_addr < LOCAL_MEM_SIZE);
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        TAN_TAYLOR_OFFSET,
        NPU_NUM,
        32,
        DT_FP32);
}
void tpu_bdc_load_fp_arcsin_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    TPUKERNEL_ASSERT(dtype == DT_FP32);
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        ARCSIN_TAYLOR_OFFSET,
        NPU_NUM,
        32,
        dtype);
}
void tpu_bdc_load_fp32_arcsin_coeff(local_addr_t coeff_addr) {
    return tpu_bdc_load_fp_arcsin_coeff(coeff_addr, DT_FP32);
}
void tpu_bdc_load_fp_exp_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        dtype == DT_FP32 ? EXP_TAYLOR_OFFSET
            : (dtype == DT_FP16 ? EXP_FP16_TAYLOR_OFFSET
                : EXP_BF16_TAYLOR_OFFSET),
        NPU_NUM,
        sfu_taylor_exp_len(dtype),
        dtype);
}
RTM_EXPORT(tpu_bdc_load_fp_exp_coeff);
void tpu_bdc_load_fp_erf_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        dtype == DT_FP32 ? ERF_TAYLOR_OFFSET
            : (dtype == DT_FP16 ? ERF_FP16_TAYLOR_OFFSET
                : ERF_BF16_TAYLOR_OFFSET),
        NPU_NUM,
        10,
        dtype);
}
void tpu_bdc_load_fp_log_coeff(local_addr_t coeff_addr, data_type_t dtype) {
    tpu_bdc_npu_bcast_from_static(
        coeff_addr,
        dtype == DT_FP32 ? LOG_TAYLOR_OFFSET
            : (dtype == DT_FP16 ? LOG_FP16_TAYLOR_OFFSET
                : LOG_BF16_TAYLOR_OFFSET),
        NPU_NUM,
        sfu_taylor_log_len(dtype),
        dtype);
}
int sfu_taylor_log_len(data_type_t dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype) && "log only support floating point\n");
    return (dtype == DT_FP32 ? 16 : 8);
}
void tpu_bdc_fp32_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    // WORK0 = -SRC
    tpu_bdc_neg(
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = EXP(WORK0)
    tpu_bdc_fp32_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    // WORK1 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = 1 / WORK1
    tpu_bdc_fp32_reciprocal(
        dst_addr,
        work1_addr,
        shape,
        NULL,
        NULL);
}

void tpu_bdc_fp_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    // WORK0 = -SRC
    tpu_bdc_neg(
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        dtype);
    // DST = EXP(WORK0)
    tpu_bdc_fp_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);
    // WORK1 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, ROUND_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = 1 / WORK1
    tpu_bdc_fp_reciprocal(
        dst_addr,
        work1_addr,
        shape,
        NULL,
        NULL,
        dtype);
}

// hardsigmoid(x; alpha, beta) := min(max(alpha*x + beta, 0), 1)
void tpu_bdc_fp_hsigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    const dim4   *shape,
    float         alpha,
    float         beta,
    data_type_t   dtype) {
    const scalar_t C1 = {.f32 = alpha};
    tpu_bdc_fp_mul_C(
        dst_addr,
        src_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    const scalar_t C2 = {.f32 = beta};
    tpu_bdc_fp_add_C(
        work0_addr,
        dst_addr,
        tpu_cast(C2, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_relu(
        work0_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        dtype);
    const scalar_t C3 = {.f32 = 1.f};
    tpu_bdc_min_C(
        dst_addr,
        work0_addr,
        tpu_cast(C3, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}
// hardswish(x) := x * hardsigmoid(x; 1/6, 0.5)
void tpu_bdc_fp_hswish(
    local_addr_t dst_addr,
    local_addr_t src_addr,
    local_addr_t work0_addr,
    const dim4 *shape,
    data_type_t dtype) {
    const scalar_t C1 = {.f32 = 3.0f};
    tpu_bdc_fp_add_C(
        dst_addr,
        src_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_relu(
        work0_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        dtype);
    const scalar_t C2 = {.f32 = 6.0f};
    tpu_bdc_min_C(
        dst_addr,
        work0_addr,
        tpu_cast(C2, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_mul(
        work0_addr,
        src_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    const scalar_t C3 = {.f32 = 1.0f / 6};
    tpu_bdc_fp_mul_C(
        dst_addr,
        work0_addr,
        tpu_cast(C3, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp_isfinite(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    scalar_t C = {.u32 = (dtype == DT_FP32 ? 0x7f800000 : (dtype == DT_FP16 ? 0x7c00 : 0x7f80))};
    tpu_bdc_and_C(
        work0_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        dtype);
    scalar_t C1 = {.u32 = FP_ONE(dtype)};
    tpu_bdc_not_equal_C(
        dst_addr,
        work0_addr,
        C,
        C1,
        shape,
        NULL,
        NULL,
        dtype,
        dtype);
}
void tpu_bdc_fp32_sinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    // DST = EXP(DST)
    tpu_bdc_fp32_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);

    // WORK0 = 1 / DST
    float C1 = 1.f;
    tpu_bdc_fp32_C_div(
        work0_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL);

    // DST = DST - WORK0
    tpu_bdc_fp_sub(
        dst_addr,
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);

    // DST = DST *1/2
    scalar_t C2 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        C2,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_cosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    // DST = EXP(DST)
    tpu_bdc_fp32_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);

    // WORK0 = 1 / DST
    float C1 = 1.f;
    tpu_bdc_fp32_C_div(
        work0_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL);

    // DST = DST + WORK0
    tpu_bdc_fp_add(
        dst_addr,
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);

    // DST = DST *1/2
    scalar_t C2 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        C2,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_tanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    // DST = SRC * 2
    tpu_bdc_fp_add(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);

    // DST = EXP(DST)
    tpu_bdc_fp32_exp(
        dst_addr,
        dst_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);

    // WROK0 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work0_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);

    // WORK1 = 2 / WORK0
    tpu_bdc_fp32_C_div(
        work1_addr,
        work0_addr,
        2.f,
        shape,
        NULL,
        NULL);

    // DST = 1 - WORK1
    tpu_bdc_fp_C_sub(
        dst_addr,
        work1_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_arcsinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    // DST = SRC * SRC
    tpu_bdc_fp_mul(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = SQRT(WORK)
    tpu_bdc_fp32_sqrt(
        dst_addr,
        work_addr,
        shape);
    // DST = SRC + DST
    tpu_bdc_fp_add(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    //  DST = LN(DST)
    tpu_bdc_fp32_log(
        dst_addr,
        dst_addr,
        work_addr,
        coeff_addr,
        shape);
}
void tpu_bdc_fp32_arccosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    // DST = SRC * SRC
    tpu_bdc_fp_mul(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK = DST - 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_sub_C(
        work_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = SQRT(WORK)
    tpu_bdc_fp32_sqrt(
        dst_addr,
        work_addr,
        shape);
    // DST = SRC + DST
    tpu_bdc_fp_add(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    //  DST = LN(DST)
    tpu_bdc_fp32_log(
        dst_addr,
        dst_addr,
        work_addr,
        coeff_addr,
        shape);
}
void tpu_bdc_fp32_arctanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    // DST = SRC + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        dst_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK = 1 - SRC
    tpu_bdc_fp_C_sub(
        work_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = DST / WORK
    tpu_bdc_fp32_div(
        dst_addr,
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        NULL);
    //  DST = LN(DST)
    tpu_bdc_fp32_log(
        dst_addr,
        dst_addr,
        work_addr,
        coeff_addr,
        shape);
    //  DST = DST * 0.5
    scalar_t C1 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_softplus(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape,
    float         beta) {
    // WORK0 = BETA * SRC
    if (beta != 1.f) {
        scalar_t C = {.f32 = beta};
        tpu_bdc_fp_mul_C(
            work0_addr,
            src_addr,
            C,
            shape,
            NULL,
            NULL,
            DT_FP32);
    }
    // DST = EXP(WORK0) OR EXP(SRC)
    tpu_bdc_fp32_exp(
        dst_addr,
        beta != 1.f ? work0_addr : src_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        exp_table_addr,
        shape);
    // WORK0 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work0_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST OR WORK1 = LOG(WORK0)
    tpu_bdc_fp32_log(
        beta != 1.f ? work1_addr : dst_addr,
        work0_addr,
        work0_addr,
        log_coeff_addr,
        shape);
    if (beta != 1.f)
        tpu_bdc_fp32_div_C(
            dst_addr,
            work1_addr,
            (float)(1. / beta),
            shape,
            NULL,
            NULL);
}

void tpu_bdc_fp_softplus(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    const dim4   *shape,
    float         beta,
    data_type_t   dtype) {
    // WORK0 = BETA * SRC
    if (beta != 1.f) {
        scalar_t C = {.f32 = beta};
        tpu_bdc_fp_mul_C(
            work0_addr,
            src_addr,
            tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            shape,
            NULL,
            NULL,
            dtype);
    }
    // DST = EXP(WORK0) OR EXP(SRC)
    tpu_bdc_fp_exp(
        dst_addr,
        beta != 1.f ? work0_addr : src_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        shape,
        dtype);
    // WORK0 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work0_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    // DST OR WORK1 = LOG(WORK0)
    tpu_bdc_fp_log(
        beta != 1.f ? work1_addr : dst_addr,
        work0_addr,
        work0_addr,
        log_coeff_addr,
        shape,
        dtype);
    if (beta != 1.f){
        scalar_t C1 = {.f32 = 1. / beta};
        tpu_bdc_fp_div_C(
            dst_addr,
            work1_addr,
            tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            shape,
            NULL,
            NULL,
            dtype);
    }
}

void tpu_bdc_fp32_softsign(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape) {
    // DST = ABS(SRC)
    tpu_bdc_abs(
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        DT_FP32);

    // DST = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        dst_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);

    // DST = SRC / DST
    tpu_bdc_fp32_div(
        dst_addr,
        src_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL);
}

void tpu_bdc_neg(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    scalar_t C = {.u32 = 0};
    if (tpu_is_data_type_fp(dtype))
        tpu_bdc_fp_C_sub(
            dst_addr,
            src_addr,
            C,
            shape,
            dst_stride,
            src_stride,
            dtype);
    else
        tpu_bdc_int_C_sub(
            dst_addr,
            src_addr,
            C,
            shape,
            dst_stride,
            src_stride,
            tpu_is_data_type_signed_int(dtype) ?
            dtype : ((PRECISION(dtype) << 1) | 1),
            dtype,
            dtype,
            0,
            NO_USE,
            false);
}
void tpu_bdc_fp_tunable_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype,
    int           num_iter) {
    scalar_t C;
    C.f32 = 1.0;
    tpu_bdc_fp_tunable_C_div(
        dst_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        dst_stride,
        src_stride,
        dtype,
        num_iter);
}

void tpu_bdc_fp_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype) {
    tpu_bdc_fp_tunable_reciprocal(
        dst_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        dtype,
        div_iter_num);
}

void tpu_bdc_fp32_tunable_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter) {
    tpu_bdc_fp_tunable_reciprocal(
        dst_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        DT_FP32,
        num_iter);
}

void tpu_bdc_fp32_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride) {
    tpu_bdc_fp32_tunable_reciprocal(
        dst_addr,
        src_addr,
        shape,
        dst_stride,
        src_stride,
        div_iter_num);
}

void tpu_bdc_fp32_tunable_compensate_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    const dim4   *shape,
    const dim4   *src_stride,
    int           num_iter,
    int           num_comp) {
    TPUKERNEL_ASSERT(num_comp >= 0);
    if (num_comp > 0)
        TPUKERNEL_ASSERT(
            dst_addr != src_addr && dst_addr != work0_addr &&
            dst_addr != work1_addr && dst_addr != work2_addr &&
            src_addr != work0_addr && src_addr != work1_addr &&
            src_addr != work2_addr && work0_addr != work1_addr &&
            work0_addr != work2_addr && work1_addr != work2_addr);
    // DST = 1 / SRC
    tpu_bdc_fp32_tunable_reciprocal(
        dst_addr,
        src_addr,
        shape,
        NULL,
        src_stride,
        num_iter);
    scalar_t C = {.f32 = 1.f};
    for (int i = 0; i < num_comp; ++i) {
        // WORK0 = DST
        tpu_bdc_cpy(
            work0_addr,
            dst_addr,
            shape,
            NULL,
            NULL,
            DT_FP32);
        // WORK1 = WORK0 * SRC
        tpu_bdc_fp_mul(
            work1_addr,
            work0_addr,
            src_addr,
            shape,
            NULL,
            NULL,
            src_stride,
            DT_FP32);
        // WORK1 = 1 - WORK1
        tpu_bdc_fp_C_sub(
            work1_addr,
            work1_addr,
            C,
            shape,
            NULL,
            NULL,
            DT_FP32);
        // WORK0 = WORK0 + WORK0 * WORK1
        tpu_bdc_fp32_mac(
            work0_addr,
            work0_addr,
            work1_addr,
            shape,
            NULL,
            NULL,
            NULL);
        // WORK2 = WORK0 * SRC
        tpu_bdc_fp_mul(
            work2_addr,
            work0_addr,
            src_addr,
            shape,
            NULL,
            NULL,
            src_stride,
            DT_FP32);
        // WORK2 = 1 - WORK2
        tpu_bdc_fp_C_sub(
            work2_addr,
            work2_addr,
            C,
            shape,
            NULL,
            NULL,
            DT_FP32);
        // WORK1 = ABS(WORK1)
        tpu_bdc_abs(
            work1_addr,
            work1_addr,
            shape,
            NULL,
            NULL,
            DT_FP32);
        // WORK2 = ABS(WORK2)
        tpu_bdc_abs(
            work2_addr,
            work2_addr,
            shape,
            NULL,
            NULL,
            DT_FP32);
        // DST = WORK1 < WORK2 ? DST : WORK0
        variable_t dst = {.type = TENSOR, .context = {.addr = dst_addr}};
        variable_t work0 = {.type = TENSOR, .context = {.addr = work0_addr}};
        variable_t work1 = {.type = TENSOR, .context = {.addr = work1_addr}};
        variable_t work2 = {.type = TENSOR, .context = {.addr = work2_addr}};
        tpu_bdc_less_select(
            dst_addr,
            &work1,
            &work2,
            &dst,
            &work0,
            shape,
            DT_FP32,
            DT_FP32);
    }
}
void tpu_bdc_fp32_compensate_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    const dim4   *shape,
    const dim4   *src_stride,
    int           num_comp) {
    tpu_bdc_fp32_tunable_compensate_reciprocal(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        shape,
        src_stride,
        div_iter_num,
        num_comp);
}
void tpu_bdc_fp32_pow(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape) {
    tpu_bdc_fp32_log(
        src1_addr == dst_addr ? work1_addr :
        (src1_addr == work0_addr ? work1_addr : dst_addr),
        src0_addr,
        src1_addr == dst_addr ? work0_addr :
        (src1_addr == work0_addr ? dst_addr : work0_addr),
        log_coeff_addr,
        shape);
    tpu_bdc_fp_mul(
        work0_addr,
        src1_addr == dst_addr ? work1_addr :
        (src1_addr == work0_addr ? work1_addr : dst_addr),
        src1_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        exp_table_addr,
        shape);
}
void tpu_bdc_fp32_pow_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    float         C,
    const dim4   *shape) {
    tpu_bdc_fp32_log(
        work1_addr,
        src_addr,
        work0_addr,
        log_coeff_addr,
        shape);
    scalar_t scalar = {.f32 = C};
    tpu_bdc_fp_mul_C(
        work0_addr,
        work1_addr,
        scalar,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        exp_table_addr,
        shape);
}
void tpu_bdc_fp32_C_pow(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  exp_table_addr,
    float         C,
    const dim4   *shape) {
    scalar_t scalar = {.f32 = log(C)};
    tpu_bdc_fp_mul_C(
        work0_addr,
        src_addr,
        scalar,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        exp_table_addr,
        shape);
}
void tpu_bdc_npu_bcast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    atomic_lane_broad_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->h,
        shape->w,
        shape->c,
        0xffffffffffffffff,
        PRECISION(dtype),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_npu_bcast);
void tpu_bdc_int_pc_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    /*FIXME: chip 1684x not support uint32, maybe overflow */
    src_dtype = src_dtype == DT_UINT32 ? DT_INT32 : src_dtype;

    TPUKERNEL_ASSERT(PRECISION(dst_dtype) == INT8 ||
                    PRECISION(dst_dtype) == INT16 ||
                    PRECISION(dst_dtype) == INT4);
    TPUKERNEL_ASSERT(src_dtype == DT_INT32 ||
                     src_dtype == DT_INT16 || src_dtype == DT_UINT16 ||
                     src_dtype == DT_INT8 || src_dtype == DT_UINT8);
    atomic_rq_i32mode_gen_cmd(
        src_addr,
        quant_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    int              multiplier,
    char             shift,
    scalar_t         offset,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    /*FIXME: chip 1684x not support uint32, maybe overflow */
    //src_dtype = src_dtype == DT_UINT32 ? DT_INT32 : src_dtype;

    TPUKERNEL_ASSERT(PRECISION(dst_dtype) == INT8 ||
                    PRECISION(dst_dtype) == INT16 ||
                    PRECISION(dst_dtype) == INT4);
    TPUKERNEL_ASSERT(src_dtype == DT_INT32 || src_dtype == DT_UINT32 ||
                     src_dtype == DT_INT16 || src_dtype == DT_UINT16 ||
                     src_dtype == DT_INT8 || src_dtype == DT_UINT8);
    atomic_rq_i32mode_gen_cmd(
        src_addr,
        NO_USE,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        multiplier,
        shift,
        offset.s16,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    float            scale,
    float            offset,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  dst_rounding_mode,
    rounding_mode_t  src_rounding_mode) {
    /*FIXME: chip 1684x not support uint32, maybe overflow */
    src_dtype = src_dtype == DT_UINT32 ? DT_INT32 : src_dtype;

    TPUKERNEL_ASSERT(PRECISION(dst_dtype) == INT8 ||
                    PRECISION(dst_dtype) == INT16 ||
                    PRECISION(dst_dtype) == INT4);
    TPUKERNEL_ASSERT(src_dtype == DT_INT32 || src_dtype == DT_INT16 ||
                     src_dtype == DT_INT8  || src_dtype == DT_UINT8 ||
                     src_dtype == DT_UINT16);
    atomic_rq_f32mode_gen_cmd(
        src_addr,
        NO_USE,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        scale,
        offset,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)src_rounding_mode,
        (ROUND_MODE)dst_rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_pc_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  dst_rounding_mode,
    rounding_mode_t  src_rounding_mode) {
    /*FIXME: chip 1684x not support uint32, maybe overflow */
    src_dtype = src_dtype == DT_UINT32 ? DT_INT32 : src_dtype;

    TPUKERNEL_ASSERT(PRECISION(dst_dtype) == INT8 ||
                     PRECISION(dst_dtype) == INT16 ||
                     PRECISION(dst_dtype) == INT4);
    TPUKERNEL_ASSERT(src_dtype == DT_INT32 || src_dtype == DT_INT16 ||
                     src_dtype == DT_UINT16 || src_dtype == DT_INT8 ||
                     src_dtype == DT_UINT8);
    atomic_rq_f32mode_gen_cmd(
        src_addr,
        quant_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        NO_USE,
        NO_USE,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)src_rounding_mode,
        (ROUND_MODE)dst_rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    scalar_t         offset,
    int              multiplier,
    char             shift,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(dst_dtype == DT_INT32 ||
                     dst_dtype == DT_INT16 || dst_dtype == DT_UINT16 ||
                     dst_dtype == DT_INT8 || dst_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(PRECISION(src_dtype) == INT8 ||
                    PRECISION(src_dtype) == INT16 ||
                    PRECISION(src_dtype) == INT4);
    atomic_dq_i32mode_gen_cmd(
        src_addr,
        NO_USE,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        offset.s16,
        multiplier,
        shift,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int_pc_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(dst_dtype == DT_INT32 ||
                     dst_dtype == DT_INT16 || dst_dtype == DT_UINT16 ||
                     dst_dtype == DT_INT8 || dst_dtype == DT_UINT8);
    TPUKERNEL_ASSERT(PRECISION(src_dtype) == INT8 ||
                    PRECISION(src_dtype) == INT16 ||
                    PRECISION(src_dtype) == INT4);
    atomic_dq_i32mode_gen_cmd(
        src_addr,
        quant_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        NO_USE,
        NO_USE,
        NO_USE,
        SIGN(src_dtype),
        SIGN(dst_dtype),
        0,
        PRECISION(src_dtype),
        PRECISION(dst_dtype),
        (ROUND_MODE)rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    scalar_t         offset,
    float            scale,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(PRECISION(src_dtype) == INT8 ||
                    PRECISION(src_dtype) == INT4 ||
                    PRECISION(src_dtype) == INT16);
    atomic_dq_f32mode_gen_cmd(
        src_addr,
        NO_USE,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        true,
        scale,
        offset.s16,
        SIGN(src_dtype),
        SIGN(DT_FP32),
        PRECISION(src_dtype),
        rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp32_pc_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode) {
    TPUKERNEL_ASSERT(PRECISION(src_dtype) == INT8 ||
                    PRECISION(src_dtype) == INT4 ||
                    PRECISION(src_dtype) == INT16);
    atomic_dq_f32mode_gen_cmd(
        src_addr,
        quant_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        false,
        NO_USE,
        NO_USE,
        SIGN(src_dtype),
        SIGN(DT_FP32),
        PRECISION(src_dtype),
        rounding_mode,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_f16_group_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      src_dtype,
    data_type_t      dst_dtype,
    int              group) {
    NOT_SUPPORT(__func__);
}
void tpu_bdc_fp_tunable_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype,
    int           num_iter) {
    int short_str[3] = {
        ALIGNED_OR_USER(src0_stride),
        ALIGNED_OR_USER(src1_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_div_gen_cmd(
        src0_addr,
        src1_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src0_stride,
        (int *)src1_stride,
        (int *)dst_stride,
        false,
        false,
        short_str,
        PRECISION(dtype),
        num_iter,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_tunable_div);
void tpu_bdc_fp32_tunable_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    int           num_iter){
    tpu_bdc_fp_tunable_div(
        dst_addr,
        src0_addr,
        src1_addr,
        shape,
        dst_stride,
        src0_stride,
        src1_stride,
        DT_FP32,
        num_iter);
}
void tpu_bdc_fp_tunable_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype,
    int           num_iter) {
    int short_str[3] = {
        ALIGNED_OR_USER(src_stride),
        NO_USE,
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_div_gen_cmd(
        src_addr,
        C.u32,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        (int *)src_stride,
        NULL,
        (int *)dst_stride,
        false,
        true,
        short_str,
        PRECISION(dtype),
        num_iter,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp32_tunable_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter) {
    scalar_t C_;
    C_.f32 = C;
    tpu_bdc_fp_tunable_div_C(
        dst_addr,
        src_addr,
        C_,
        shape,
        dst_stride,
        src_stride,
        DT_FP32,
        num_iter);
}

void tpu_bdc_fp_tunable_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype,
    int           num_iter) {
    int short_str[3] = {
        NO_USE,
        ALIGNED_OR_USER(src_stride),
        ALIGNED_OR_USER(dst_stride)
    };
    atomic_tensor_arithmetic_div_gen_cmd(
        C.u32,
        src_addr,
        dst_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        (int *)src_stride,
        (int *)dst_stride,
        true,
        false,
        short_str,
        PRECISION(dtype),
        num_iter,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
RTM_EXPORT(tpu_bdc_fp_tunable_C_div);

void tpu_bdc_fp32_tunable_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter) {
    scalar_t C_;
    C_.f32 = C;
    tpu_bdc_fp_tunable_C_div(dst_addr, src_addr, C_, shape, dst_stride, src_stride, DT_FP32, num_iter);
}
void tpu_bdc_fp_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype) {
    tpu_bdc_fp_tunable_div(
        dst_addr,
        src0_addr,
        src1_addr,
        shape,
        dst_stride,
        src0_stride,
        src1_stride,
        dtype,
        div_iter_num);
}

void tpu_bdc_fp_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t  dtype) {
    tpu_bdc_fp_tunable_div_C(
        dst_addr,
        src_addr,
        C,
        shape,
        dst_stride,
        src_stride,
        dtype,
        div_iter_num);
}

void tpu_bdc_fp_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t  dtype) {
    tpu_bdc_fp_tunable_C_div(
        dst_addr,
        src_addr,
        C,
        shape,
        dst_stride,
        src_stride,
        dtype,
        div_iter_num);
}

void tpu_bdc_fp32_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride) {
    tpu_bdc_fp_div(
        dst_addr,
        src0_addr,
        src1_addr,
        shape,
        dst_stride,
        src0_stride,
        src1_stride,
        DT_FP32);
}

void tpu_bdc_fp32_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride) {
    scalar_t C_;
    C_.f32 = C;
    tpu_bdc_fp_div_C(
        dst_addr,
        src_addr,
        C_,
        shape,
        dst_stride,
        src_stride,
        DT_FP32);
}

void tpu_bdc_fp32_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride) {
    scalar_t C_;
    C_.f32 = C;
    tpu_bdc_fp_C_div(
        dst_addr,
        src_addr,
        C_,
        shape,
        dst_stride,
        src_stride,
        DT_FP32);
}

void tpu_bdc_cw_trans(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    atomic_cw_transpose_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->w,
        shape->h,
        shape->c,
        PRECISION(dtype),
        TRAN_C_W_TRANSPOSE,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_wc_trans(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    atomic_cw_transpose_gen_cmd(
        src_addr,
        dst_addr,
        shape->n,
        shape->w,
        shape->h,
        shape->c,
        PRECISION(dtype),
        TRAN_W_C_TRANSPOSE,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_sign(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    bool dtype_signed = tpu_is_data_type_signed(dtype);
    if (dtype_signed)
        TPUKERNEL_ASSERT(dst_addr != src_addr);
    variable_t src = {.type = TENSOR, .context = {.addr = src_addr}};
    variable_t dst = {.type = TENSOR, .context = {.addr = dst_addr}};
    variable_t C0 = {.type = SCALAR, .context = {.scalar = {.u32 = 0}}};
    variable_t C1 = {
        .type = SCALAR,
        .context = {
            .scalar = {.u32 = tpu_is_data_type_fp(dtype) ? FP_ONE(dtype) : 1}
        }
    };
    variable_t CN1 = {
        .type = SCALAR, .context = {
            .scalar = {
                .u32 = tpu_is_data_type_fp(dtype) ?
                FP_NEG_ONE(dtype) : 0xffffffff
            }
        }
    };
    tpu_bdc_greater_select(
        dst_addr,
        &src,
        &C0,
        &C1,
        dtype_signed ? &CN1 : &C0,
        shape,
        dtype,
        dtype);
    if (dtype_signed)
        tpu_bdc_equal_select(
            dst_addr,
            &src,
            &C0,
            &C0,
            &dst,
            shape,
            dtype,
            dtype);
}
void tpu_bdc_fp_sin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(dst_addr != work_addr);
    // WORK = SRC / 2PI
    scalar_t C = {.f32 = 1. / (M_PI * 2.)};
    tpu_bdc_fp_mul_C(
        work_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = ROUND(WORK)
    tpu_bdc_fp_round(
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        dtype,
        RM_HALF_AWAY_FROM_ZERO);
    // DST = WORK - DST
    /*
     * -x.9 ->  0.1, -x.8 ->  0.2, -x.7 ->  0.3, -x.6 ->  0.4, -x.5 ->  0.5
     * -x.4 -> -0.4, -x.3 -> -0.3, -x.2 -> -0.2, -x.1 -> -0.1, -x.0 ->  0.0
     *  x.1 ->  0.1,  x.2 ->  0.2,  x.3 ->  0.3,  x.4 ->  0.4,  x.5 -> -0.5
     *  x.6 -> -0.4,  x.7 -> -0.3,  x.8 -> -0.2,  x.9 -> -0.1,  x.0 ->  0.0
     */
    tpu_bdc_fp_sub(
        dst_addr,
        work_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK = DST * DST
    tpu_bdc_fp_mul(
        work_addr,
        dst_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK = TAYLOR(WORK)
    tpu_bdc_fp_taylor(
        work_addr,
        work_addr,
        coeff_addr,
        shape,
        sfu_taylor_sin_len,
        dtype);
    // DST = DST * WORK
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_sin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    tpu_bdc_fp_sin(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        DT_FP32);
}
void tpu_bdc_fp_cos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(dst_addr != work_addr);
    // WORK = SRC / 2PI
    scalar_t C = {.f32 = 1. / (M_PI * 2.)};
    tpu_bdc_fp_mul_C(
        work_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = ROUND(WORK)
    tpu_bdc_fp_round(
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        dtype,
        RM_HALF_AWAY_FROM_ZERO);
    // DST = WORK - DST
    /*
     * -x.9 ->  0.1, -x.8 ->  0.2, -x.7 ->  0.3, -x.6 ->  0.4, -x.5 ->  0.5
     * -x.4 -> -0.4, -x.3 -> -0.3, -x.2 -> -0.2, -x.1 -> -0.1, -x.0 ->  0.0
     *  x.1 ->  0.1,  x.2 ->  0.2,  x.3 ->  0.3,  x.4 ->  0.4,  x.5 -> -0.5
     *  x.6 -> -0.4,  x.7 -> -0.3,  x.8 -> -0.2,  x.9 -> -0.1,  x.0 ->  0.0
     */
    tpu_bdc_fp_sub(
        dst_addr,
        work_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK = DST * DST
    tpu_bdc_fp_mul(
        work_addr,
        dst_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = TAYLOR(WORK)
    tpu_bdc_fp_taylor(
        dst_addr,
        work_addr,
        coeff_addr,
        shape,
        sfu_taylor_cos_len,
        dtype);
}
void tpu_bdc_fp32_cos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    tpu_bdc_fp_cos(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        DT_FP32);
}
void tpu_bdc_fp32_tan_cot(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    bool          tan) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work_addr);
    /*
     * To reduce large loss in bound case (value in src is very close to -/+ 0.5pi),
     * assume x in range [0, pi/2), transform formula as following:
     * tan(x) = (1-tan(x-pi/4))/(1-tan(x-pi/4))
     */

    // WORK = SRC / PI
    scalar_t C = {.f32 = 1. / M_PI};
    tpu_bdc_fp_mul_C(
        work_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    if (!tan) {
        // WORK = 0.5 - WORK
        scalar_t C05 = {.f32 = 0.5f};
        tpu_bdc_fp_C_sub(
            work_addr,
            work_addr,
            C05,
            shape,
            NULL,
            NULL,
            DT_FP32);
    }
    // DST = ROUND(WORK)
    tpu_bdc_fp_round(
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        DT_FP32,
        RM_HALF_TO_EVEN);
    // DST = WORK - DST
    /*
     * -x.9 ->  0.1, -x.8 ->  0.2, -x.7 ->  0.3, -x.6 ->  0.4, -x.5 ->  0.5
     * -x.4 -> -0.4, -x.3 -> -0.3, -x.2 -> -0.2, -x.1 -> -0.1, -x.0 ->  0.0
     *  x.1 ->  0.1,  x.2 ->  0.2,  x.3 ->  0.3,  x.4 ->  0.4,  x.5 -> -0.5
     *  x.6 -> -0.4,  x.7 -> -0.3,  x.8 -> -0.2,  x.9 -> -0.1,  x.0 ->  0.0
     */
    tpu_bdc_fp_sub(
        src_addr,
        work_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);

    // WORK = ABS(SRC)
    scalar_t C_ABS_MASK = {.u32 = 0x7fffffff};
    tpu_bdc_and_C(
        work_addr,
        src_addr,
        C_ABS_MASK,
        shape,
        NULL,
        NULL,
        DT_FP32);

    //  DST = WORK - 0.25
    scalar_t C025 = {.f32 = 0.25f};
    tpu_bdc_fp_sub_C(
        dst_addr,
        work_addr,
        C025,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK = DST ^ 2
    tpu_bdc_fp_mul(
        work_addr,
        dst_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK = TAYLOR(WORK)
    tpu_bdc_fp_taylor(
        work_addr,
        work_addr,
        coeff_addr,
        shape,
        sfu_taylor_tan_len,
        DT_FP32);
    // DST = DST * WORK
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);

    // DST = (1+DST)/(1-DST) = 2/(1-DST) - 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_C_sub(
        work_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_C_div(
        dst_addr,
        work_addr,
        2,
        shape,
        NULL,
        NULL);
    tpu_bdc_fp_sub_C(
        dst_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);

    // DST = SIGN(SRC) * DST
    scalar_t C_SIGN_MASK = {.u32 = 0x80000000};
    tpu_bdc_and_C(
        src_addr,
        src_addr,
        C_SIGN_MASK,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_or(
        dst_addr,
        src_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_tan(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    tpu_bdc_fp32_tan_cot(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        true);
}
void tpu_bdc_fp32_cot(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    tpu_bdc_fp32_tan_cot(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        false);
}
void tpu_bdc_fp_arcsin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    TPUKERNEL_ASSERT(dtype == DT_FP32);
    // DST = SRC * SRC
    tpu_bdc_fp_mul(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = DST * 0.25
    scalar_t coeff = {.f32 = 0.25f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        tpu_cast(coeff, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK = 0.25 - DST
    scalar_t quarter = {.f32 = 0.25f};
    tpu_bdc_fp_C_sub(
        work_addr,
        dst_addr,
        tpu_cast(quarter, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = SQRT(WORK)
    tpu_bdc_fp_sqrt(
        dst_addr,
        work_addr,
        shape,
        dtype);
    // WORK = 0.5 - DST
    scalar_t half = {.f32 = 0.5f};
    tpu_bdc_fp_C_sub(
        work_addr,
        dst_addr,
        tpu_cast(half, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = SQRT(WORK)
    tpu_bdc_fp_sqrt(
        dst_addr,
        work_addr,
        shape,
        dtype);
    // WORK = TAYLOR(WORK)
    tpu_bdc_fp_taylor(
        work_addr,
        work_addr,
        coeff_addr,
        shape,
        sfu_taylor_arcsin_len,
        dtype);
    // WORK = WORK * DST
    tpu_bdc_fp_mul(
        work_addr,
        work_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = WORK * 2
    scalar_t C = {.f32 = 2.f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        work_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
   // WORK = SIGN(SRC)
    tpu_bdc_sign(
        work_addr,
        src_addr,
        shape,
        dtype);
    // DST = DST * WORK
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_arcsin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    return tpu_bdc_fp_arcsin(dst_addr, src_addr, work_addr, coeff_addr, shape, DT_FP32);
}
void tpu_bdc_fp_arccos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(dtype == DT_FP32);
    tpu_bdc_fp_arcsin(
        dst_addr,
        src_addr,
        work_addr,
        coeff_addr,
        shape,
        dtype);
    scalar_t C = {.f32 = M_PI * 0.5};
    tpu_bdc_fp_C_sub(
        dst_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_arccos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape) {
    return tpu_bdc_fp_arccos(dst_addr, src_addr, work_addr, coeff_addr, shape, DT_FP32);
}
void tpu_bdc_fp32_erf(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape) {
    tpu_bdc_fp_erf(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        shape,
        DT_FP32);
#if 0
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr && src_addr != work2_addr &&
                    work2_addr != dst_addr && work2_addr != work0_addr &&
                    work2_addr != work1_addr);
    // DST = |SRC|
    tpu_bdc_abs(
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK2 = 0.5 * DST
    scalar_t C05 = {.f32 = .5f};
    tpu_bdc_fp_mul_C(
        work2_addr,
        dst_addr,
        C05,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = WORK2 + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        dst_addr,
        work2_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK2 = 1 / DST
    tpu_bdc_fp32_reciprocal(
        work2_addr,
        dst_addr,
        shape,
        NULL,
        NULL);
    // DST = TAYLOR(WORK2)
    tpu_bdc_fp_taylor(
        dst_addr,
        work2_addr,
        erf_coeff_addr,
        shape,
        10,
        DT_FP32);
    // WORK0 = SRC * SRC
    tpu_bdc_fp_mul(
        work0_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = DST - WORK0
    tpu_bdc_fp_sub(
        work0_addr,
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // DST = EXP(WORK0)
    tpu_bdc_fp32_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        exp_table_addr,
        shape);
    // WORK0 = DST * WORK2
    tpu_bdc_fp_mul(
        work0_addr,
        dst_addr,
        work2_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK2 = 1 - WORK0
    tpu_bdc_fp_C_sub(
        work2_addr,
        work0_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = SIGN(SRC)
    tpu_bdc_sign(
        work0_addr,
        src_addr,
        shape,
        DT_FP32);
    // DST = WORK2 * WORK0
    tpu_bdc_fp_mul(
        dst_addr,
        work2_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
#endif
}

void tpu_bdc_fp_erf(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr && src_addr != work2_addr &&
                    work2_addr != dst_addr && work2_addr != work0_addr &&
                    work2_addr != work1_addr);
    // DST = |SRC|
    tpu_bdc_abs(
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        dtype);
    // WORK2 = 0.5 * DST
    scalar_t C05 = {.f32 = .5f};
    tpu_bdc_fp_mul_C(
        work2_addr,
        dst_addr,
        tpu_cast(C05, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = WORK2 + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        dst_addr,
        work2_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK2 = 1 / DST
    tpu_bdc_fp_reciprocal(
        work2_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        dtype);
    // DST = TAYLOR(WORK2)
    tpu_bdc_fp_taylor(
        dst_addr,
        work2_addr,
        erf_coeff_addr,
        shape,
        10,
        dtype);
    // WORK1 = SRC * SRC
    tpu_bdc_fp_mul(
        work1_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK0 = DST - WORK1
    tpu_bdc_fp_sub(
        work0_addr,
        dst_addr,
        work1_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = EXP(WORK0)
    tpu_bdc_fp_exp(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        exp_coeff_addr,
        shape,
        dtype);
    // WORK0 = DST * WORK2
    tpu_bdc_fp_mul(
        work0_addr,
        dst_addr,
        work2_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK2 = 1 - WORK0
    tpu_bdc_fp_C_sub(
        work2_addr,
        work0_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK0 = SIGN(SRC)
    tpu_bdc_sign(
        work0_addr,
        src_addr,
        shape,
        dtype);
    // DST = WORK2 * WORK0
    tpu_bdc_fp_mul(
        dst_addr,
        work2_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}

void tpu_bdc_fp32_gelu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  work3_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape) {
    tpu_bdc_fp_gelu(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        work3_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        shape,
        DT_FP32);
}
void tpu_bdc_fp_gelu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  work3_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    // WORK3 = SRC / SQRT(2)
    scalar_t C = {.f32 = 1 / sqrt(2.)};
    tpu_bdc_fp_mul_C(
        work3_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = ERF(WORK3)
    tpu_bdc_fp_erf(
        dst_addr,
        work3_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        shape,
        dtype);
    // WORK1 = DST + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = WORK1 * 0.5
    scalar_t C05 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        work1_addr,
        tpu_cast(C05, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);

    // DST *= SRC
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_gelu_fast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    // WORK0 = SRC * SRC
    tpu_bdc_fp_mul(
        work0_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK1 = WORK0 * SRC
    tpu_bdc_fp_mul(
        work1_addr,
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = WORK1 * 0.044715
    scalar_t C = {.f32 = 0.044715f};
    tpu_bdc_fp_mul_C(
        work0_addr,
        work1_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK1 = WORK0 + SRC
    tpu_bdc_fp_add(
        work1_addr,
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = WORK1 * SQRT(2 / PI)
    scalar_t Csqrt = {.f32 = sqrt(2. / M_PI)};
    tpu_bdc_fp_mul_C(
        work0_addr,
        work1_addr,
        Csqrt,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = TANH(WORK0)
    tpu_bdc_fp32_tanh(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    // WORK1 = DST + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        C1,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = WORK1 * SRC
    tpu_bdc_fp_mul(
        work0_addr,
        work1_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
    // DST = WORK0 * 0.5
    scalar_t C05 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        work0_addr,
        C05,
        shape,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp_gelu_fast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                     src_addr != work1_addr);
    // WORK0 = SRC * SRC
    tpu_bdc_fp_mul(
        work0_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK1 = WORK0 * SRC
    tpu_bdc_fp_mul(
        work1_addr,
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK0 = WORK1 * 0.044715
    scalar_t C = {.f32 = 0.044715f};
    tpu_bdc_fp_mul_C(
        work0_addr,
        work1_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK1 = WORK0 + SRC
    tpu_bdc_fp_add(
        work1_addr,
        work0_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // WORK0 = WORK1 * SQRT(2 / PI)
    scalar_t Csqrt = {.f32 = sqrt(2. / M_PI)};
    tpu_bdc_fp_mul_C(
        work0_addr,
        work1_addr,
        tpu_cast(Csqrt, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = TANH(WORK0)
    tpu_bdc_fp_tanh(
        dst_addr,
        work0_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);
    // WORK1 = DST + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK0 = WORK1 * SRC
    tpu_bdc_fp_mul(
        work0_addr,
        work1_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // DST = WORK0 * 0.5
    scalar_t C05 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        work0_addr,
        tpu_cast(C05, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_gelu_fast2(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    tpu_bdc_fp_gelu_fast2(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape,
        DT_FP32);
}
void tpu_bdc_fp_gelu_fast2(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                     src_addr != work1_addr);
    const scalar_t C = {.f32 = 1.702};
    tpu_bdc_fp_mul_C(
        dst_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_sigmoid(
        dst_addr,
        dst_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape,
        dtype);
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_erfc(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape) {
    tpu_bdc_fp_erfc(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        shape,
        DT_FP32);
#if 0
    tpu_bdc_fp32_erf(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        exp_table_addr,
        shape);
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_C_sub(
        dst_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
#endif
}
void tpu_bdc_fp_erfc(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    tpu_bdc_fp_erf(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        work2_addr,
        exp_coeff_addr,
        erf_coeff_addr,
        shape,
        dtype);
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_C_sub(
        dst_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp32_mish(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    // DST = EXP(SRC)
    tpu_bdc_fp32_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    // for not getting inf result during the next calculation WORK1 = (DST + 1) ^ 2
    scalar_t float_sqrt_max_ = {.f32 = 1.84467e+19f};//SQRT(FLT_MAX)
    scalar_t float_sqrt_min_ = {.f32 = 1.08420e-19f};//SQRT(FLT_MIN)
    variable_t exp_variable =  {.type = TENSOR, .context = {.addr = dst_addr}};
    variable_t cut_off_variable =  {.type = TENSOR, .context = {.addr = work0_addr}};
    tpu_bdc_set_C(
        work0_addr,
        float_sqrt_max_,
        shape,
        NULL,
        DT_FP32
    );
    tpu_bdc_greater_select(
        dst_addr,
        &exp_variable,
        &cut_off_variable,
        &cut_off_variable,
        &exp_variable,
        shape,
        DT_FP32,
        DT_FP32
    );
    tpu_bdc_set_C(
        work0_addr,
        float_sqrt_min_,
        shape,
        NULL,
        DT_FP32
    );
    tpu_bdc_less_select(
        dst_addr,
        &exp_variable,
        &cut_off_variable,
        &cut_off_variable,
        &exp_variable,
        shape,
        DT_FP32,
        DT_FP32
    );
    // WORK1 = (DST + 1) ^ 2
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C_sqr(
        work1_addr,
        dst_addr,
        C,
        shape,
        DT_FP32);
    // DST = WORK1 + 1
    tpu_bdc_fp_add_C(
        dst_addr,
        work1_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // WORK0 = SRC / DST
    scalar_t scale = {.f32 = -2.f};
    tpu_bdc_fp32_div(
        work0_addr,
        src_addr,
        dst_addr,
        shape,
        NULL,
        NULL,
        NULL);
    // WORK1 = WORK0*(-2)
    tpu_bdc_fp_mul_C(
        work1_addr,
        work0_addr,
        scale,
        shape,
        NULL,
        NULL,
        DT_FP32);
    // DST = SRC + WORK1
    tpu_bdc_fp_add(
        dst_addr,
        src_addr,
        work1_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_swish(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    float         beta,
    const dim4   *shape) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    scalar_t C = {.f32 = beta};
    tpu_bdc_fp_mul_C(
        dst_addr,
        src_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    tpu_bdc_fp32_sigmoid(
        dst_addr,
        dst_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
}
void tpu_bdc_fp32_silu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    tpu_bdc_fp32_sigmoid(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
}

void tpu_bdc_fp_silu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    TPUKERNEL_ASSERT(src_addr != dst_addr && src_addr != work0_addr &&
                    src_addr != work1_addr);
    tpu_bdc_fp_sigmoid(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape,
        dtype);
    tpu_bdc_fp_mul(
        dst_addr,
        dst_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}

void tpu_bdc_fp32_selu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    tpu_bdc_fp32_elu(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        1.6732632423543772848170429916717f,
        shape);
    scalar_t C = {.f32 = 1.0507009873554804934193349852946f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
}

void tpu_bdc_fp32_log_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    local_addr_t  ln_coeff_addr,
    const dim4   *shape) {
    //  DST = EXP(SRC)
    tpu_bdc_fp32_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        table_addr,
        shape);
    //  WORK1 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        C,
        shape,
        NULL,
        NULL,
        DT_FP32);
    //  WORK0 = LOG(WORK1)
    tpu_bdc_fp32_log(
        work0_addr,
        work1_addr,
        dst_addr,
        ln_coeff_addr,
        shape);
    //  DST = SRC - WORK0
    tpu_bdc_fp_sub(
        dst_addr,
        src_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
}

void tpu_bdc_random_gen_init(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    int             need_store_state,
    int             jump_cnt,
    int             c_offset,
    const dim4     *shape,
    data_type_t     dtype) {
    atomic_random_gen_init_seed_gen_cmd(
        res_addr, shape->n, shape->c, shape->h, shape->w, NULL,
        0, PRECISION(dtype), jump_cnt, c_offset, store_state_addr,
        need_store_state, MASTER_THREAD, BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_random_gen(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    int             need_store_state,
    const dim4     *shape,
    data_type_t     dtype) {
    atomic_random_gen_gen_cmd(
        res_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        0,
        PRECISION(dtype),
        0,
        store_state_addr,
        need_store_state,
        PRNG,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_random_gen_load_state(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    local_addr_t    load_state_addr,
    int             need_store_state,
    const dim4     *shape,
    data_type_t     dtype) {
    atomic_random_gen_gen_cmd(
        res_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NULL,
        0,
        PRECISION(dtype),
        load_state_addr,
        store_state_addr,
        need_store_state,
        PRNG_WITH_LOADED_STATES,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

#define TPU_BDC_VC_BINARY(name, op)                                            \
void tpu_bdc_vc_##name(                                                        \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dtype) {                                                     \
    atomic_vector_correlation_gen_cmd(                                         \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        op,                                                                    \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        NO_USE,                                                                \
        NO_USE,                                                                \
        IS_FLOAT(dtype) ? FP8TYPE(dtype) : SIGN(dtype),                        \
        IS_FLOAT(dtype) ? FP8TYPE(dtype) : SIGN(dtype),                        \
        IS_FLOAT(dtype) ? FP8TYPE(dtype) : SIGN(dtype),                        \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_VC_BINARY(min, AR_MIN)
TPU_BDC_VC_BINARY(max, AR_MAX)
#define TPU_BDC_VC_BIN_BINARY(name, op)                                        \
void tpu_bdc_vc_##name(                                                        \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dtype) {                                                     \
    if (dtype == DT_FP32)                                                      \
        dtype = DT_INT32;                                                      \
    else if (dtype == DT_FP16 || dtype == DT_BFP16)                            \
        dtype = DT_INT16;                                                      \
    atomic_vector_correlation_gen_cmd(                                         \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        op,                                                                    \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        NO_USE,                                                                \
        NO_USE,                                                                \
        SIGN(dtype),                                                           \
        SIGN(dtype),                                                           \
        SIGN(dtype),                                                           \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_VC_BIN_BINARY(xor, AR_XOR)
TPU_BDC_VC_BIN_BINARY(and, AR_AND)
TPU_BDC_VC_BIN_BINARY(or,  AR_OR)
#define TPU_BDC_VC_CMP(name, op)                                               \
void tpu_bdc_vc_##name(                                                        \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    scalar_t      true_val,                                                    \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    TPUKERNEL_ASSERT(WIDTH(src_dtype) >= WIDTH(dst_dtype));                    \
    atomic_vector_correlation_gen_cmd(                                         \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        op,                                                                    \
        PRECISION(src_dtype),                                                  \
        PRECISION(src_dtype),                                                  \
        PRECISION(dst_dtype),                                                  \
        NO_USE,                                                                \
        true_val.u32,                                                          \
        IS_FLOAT(src_dtype) ? FP8TYPE(src_dtype) : SIGN(src_dtype),            \
        IS_FLOAT(src_dtype) ? FP8TYPE(src_dtype) : SIGN(src_dtype),            \
        IS_FLOAT(dst_dtype) ? FP8TYPE(dst_dtype) : SIGN(dst_dtype),            \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_VC_CMP(greater, AR_SG)
TPU_BDC_VC_CMP(less,    AR_SL)
TPU_BDC_VC_CMP(equal,   AR_SE)
#define TPU_BDC_VC_CMP_EXT(name, origin)                                       \
void tpu_bdc_vc_##name(                                                        \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    scalar_t      true_val,                                                    \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src_dtype) {                                                 \
    tpu_bdc_vc_##origin(                                                       \
        dst_addr,                                                              \
        src0_addr,                                                             \
        src1_addr,                                                             \
        true_val,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        dst_dtype,                                                             \
        src_dtype);                                                            \
    dim4 shape = {                                                             \
        .n = src0_len, .c = DIV_UP(src1_len, src1_len_per_channel),            \
        .h = 1, .w = src1_len_per_channel                                      \
    };                                                                         \
    tpu_bdc_xor_C(                                                             \
        dst_addr,                                                              \
        dst_addr,                                                              \
        true_val,                                                              \
        &shape,                                                                \
        NULL,                                                                  \
        NULL,                                                                  \
        dst_dtype);                                                            \
}
TPU_BDC_VC_CMP_EXT(greater_equal, less)
TPU_BDC_VC_CMP_EXT(less_equal, greater)
TPU_BDC_VC_CMP_EXT(not_equal, equal)
#define TPU_BDC_FLOATING_POINT_VC_BINARY(name, op)                             \
void tpu_bdc_fp_vc_##name(                                                     \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dtype) {                                                     \
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));                               \
    atomic_vector_correlation_gen_cmd(                                         \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        op,                                                                    \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        PRECISION(dtype),                                                      \
        NO_USE,                                                                \
        NO_USE,                                                                \
        FP8TYPE(dtype),                                                           \
        FP8TYPE(dtype),                                                           \
        FP8TYPE(dtype),                                                           \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FLOATING_POINT_VC_BINARY(add, AR_ADD)
TPU_BDC_FLOATING_POINT_VC_BINARY(sub, AR_SUB)
TPU_BDC_FLOATING_POINT_VC_BINARY(mul, AR_MUL)
void tpu_bdc_fp32_vc_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel) {
    atomic_vector_correlation_gen_cmd(
        src0_addr,
        src1_addr,
        dst_addr,
        src0_len,
        src1_len,
        src0_len_per_channel,
        src1_len_per_channel,
        AR_DIV,
        PRECISION(DT_FP32),
        PRECISION(DT_FP32),
        PRECISION(DT_FP32),
        NO_USE,
        NO_USE,
        SIGN(DT_FP32),
        SIGN(DT_FP32),
        SIGN(DT_FP32),
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_FIXED_POINT_VC_BINARY(name, op, sign_check)                    \
void tpu_bdc_int_vc_##name(                                                    \
    local_addr_t  dst_addr,                                                    \
    local_addr_t  src0_addr,                                                   \
    local_addr_t  src1_addr,                                                   \
    int           src0_len,                                                    \
    int           src1_len,                                                    \
    int           src0_len_per_channel,                                        \
    int           src1_len_per_channel,                                        \
    data_type_t   dst_dtype,                                                   \
    data_type_t   src0_dtype,                                                  \
    data_type_t   src1_dtype,                                                  \
    bool          saturation) {                                                \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));                          \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src0_dtype));                         \
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src1_dtype));                         \
    TPUKERNEL_ASSERT(sign_check);                                               \
    atomic_vector_correlation_gen_cmd(                                         \
        src0_addr,                                                             \
        src1_addr,                                                             \
        dst_addr,                                                              \
        src0_len,                                                              \
        src1_len,                                                              \
        src0_len_per_channel,                                                  \
        src1_len_per_channel,                                                  \
        op,                                                                    \
        PRECISION(src0_dtype),                                                 \
        PRECISION(src1_dtype),                                                 \
        PRECISION(dst_dtype),                                                  \
        NO_USE,                                                                \
        NO_USE,                                                                \
        SIGN(src0_dtype),                                                      \
        SIGN(src1_dtype),                                                      \
        SIGN(dst_dtype),                                                       \
        MASTER_THREAD,                                                      \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_FIXED_POINT_VC_BINARY(
    add, saturation ? AR_ADD_SATU : AR_ADD,
    SIGN(dst_dtype) == (SIGN(src0_dtype) | SIGN(src1_dtype)))
TPU_BDC_FIXED_POINT_VC_BINARY(
    sub, saturation ? AR_SUB_SATU : AR_SUB, SIGN(dst_dtype))
TPU_BDC_FIXED_POINT_VC_BINARY(
    mul, saturation ? AR_MUL_SATU : AR_MUL,
    SIGN(dst_dtype) == (SIGN(src0_dtype) | SIGN(src1_dtype)))
void tpu_hau_sort(
    system_addr_t  output_addr,
    system_addr_t  input_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    atomic_sort_gen_cmd(
        input_addr,
        NO_USE,
        output_addr,
        NO_USE,
        dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
        1,
        len,
        descended,
        false,
        NO_USE,
        K,
        &id_node);
}
void tpu_hau_sort_natural_index(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    atomic_sort_gen_cmd(
        input_addr,
        NO_USE,
        output_data_addr,
        output_idx_addr,
        dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
        1,
        len,
        descended,
        true,
        true,
        K,
        &id_node);
}

void tpu_hau_sort_general(
        system_addr_t src_data_addr,
        system_addr_t src_idx_addr,
        system_addr_t dst_data_addr,
        system_addr_t dst_idx_addr,
        int dtype_flags,   // 0:fp32 1:int32 2:uint32
        int len,
        int is_descend,
        int idx_enable,
        int idx_auto,
        int topk) {
    TPUKERNEL_ASSERT(topk > 0 && topk <= len);
    TPUKERNEL_ASSERT(
        dtype_flags == 0 || dtype_flags == 1 || dtype_flags == 2);
    atomic_sort_gen_cmd(
        src_data_addr,
        src_idx_addr,
        dst_data_addr,
        dst_idx_addr,
        dtype_flags,
        1,
        len,
        is_descend,
        idx_enable,
        idx_auto,
        topk,
        &id_node);
}

void tpu_hau_sort_specific_index(
    system_addr_t output_data_addr,
    system_addr_t output_idx_addr,
    system_addr_t input_data_addr,
    system_addr_t input_idx_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    atomic_sort_gen_cmd(
        input_data_addr,
        input_idx_addr,
        output_data_addr,
        output_idx_addr,
        dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
        1,
        len,
        descended,
        true,
        false,
        K,
        &id_node);
}
void tpu_hau_sort_2d(
    system_addr_t  output_addr,
    system_addr_t  input_addr,
    int            row_num,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    TPUKERNEL_ASSERT(!tpu_is_parallel_state());
  atomic_sort_gen_cmd(
      input_addr,
      NO_USE,
      output_addr,
      NO_USE,
      dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
      row_num,
      len,
      descended,
      false,
      false,
      K,
      &id_node);
}

void tpu_hau_sort_natural_index_2d(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_addr,
    int            row_num,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    TPUKERNEL_ASSERT(!tpu_is_parallel_state());
  atomic_sort_gen_cmd(
      input_addr,
      NO_USE,
      output_data_addr,
      output_idx_addr,
      dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
      row_num,
      len,
      descended,
      true,
      true,
      K,
      &id_node);
}

void tpu_hau_sort_specific_index_2d(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_data_addr,
    system_addr_t  input_idx_addr,
    int            row_num,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    TPUKERNEL_ASSERT(K > 0 && K <= len);
    TPUKERNEL_ASSERT(
        dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32);
    TPUKERNEL_ASSERT(!tpu_is_parallel_state());
  atomic_sort_gen_cmd(
      input_data_addr,
      input_idx_addr,
      output_data_addr,
      output_idx_addr,
      dtype == DT_FP32 ? 0 : (dtype == DT_INT32 ? 1 : 2),
      row_num,
      len,
      descended,
      true,
      false,
      K,
      &id_node);
}
void tpu_hau_hard_nms(
    system_addr_t  output_addr,
    system_addr_t  input_addr,
    int            box_num,
    int            keep_num) {
    TPUKERNEL_ASSERT_INFO(0, "unsupported func: %s for BM1686\n", __func__);
}
void tpu_hau_soft_nms(
    system_addr_t  output_addr,
    system_addr_t  iou_addr,
    system_addr_t  score_addr,
    float          threshold,
    int            box_num,
    int            keep_num) {
    TPUKERNEL_ASSERT_INFO(0, "unsupported func: %s for BM1686\n", __func__);
}
void tpu_hau_line_gather(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    scalar_t       C,
    int            line_num,
    int            line_len,
    int            index_len,
    int            start,
    int            end,
    data_type_t    dtype,
    bool           fill_const) {
    TPUKERNEL_ASSERT_INFO(0, "unsupported func: %s for BM1686\n", __func__);
}
void tpu_bdc_w_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    TPUKERNEL_ASSERT(shape->h == 1);
    atomic_pl_sgd1_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_w,
        shape->n,
        shape->c,
        shape->w,
        NO_USE,
        NO_USE,
        false,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PL_gather_d1coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_w_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          fill_const) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    TPUKERNEL_ASSERT(shape->h == 1);
    atomic_pl_sgd1_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_w,
        shape->n,
        shape->c,
        shape->w,
        fill_const,
        C.u32,
        true,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PL_gather_d1coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_w_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    TPUKERNEL_ASSERT(shape->h == 1);
    atomic_pl_sgd1_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_w,
        shape->n,
        shape->c,
        shape->w,
        NO_USE,
        NO_USE,
        false,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PL_scatter_d1coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_hw_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype) {
    atomic_pl_sgd2_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_h,
        param_w,
        shape->n,
        shape->c,
        1,
        shape->h * shape->w,
        NO_USE,
        NO_USE,
        false,
        PRECISION(dtype),
        PL_gather_d2coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_hw_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype,
    bool          fill_const) {
    atomic_pl_sgd2_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_h,
        param_w,
        shape->n,
        shape->c,
        1,
        shape->h * shape->w,
        fill_const,
        C.u32,
        true,
        PRECISION(dtype),
        PL_gather_d2coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_hw_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype) {
    atomic_pl_sgd2_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        1,
        param_h * param_w,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        NO_USE,
        NO_USE,
        false,
        PRECISION(dtype),
        PL_scatter_d2coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
#define TPU_BDC_HW_FLATTEN_BATCH_BCAST(name, index_w)                          \
void tpu_bdc_batch_bcast_w_##name(                                             \
    local_addr_t  output_addr,                                                 \
    local_addr_t  param_addr,                                                  \
    local_addr_t  index_addr,                                                  \
    const dim4   *shape,                                                       \
    int           param_w,                                                     \
    data_type_t   dtype,                                                       \
    data_type_t   index_dtype,                                                 \
    bool          is_param_repeated) {                                         \
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);     \
    TPUKERNEL_ASSERT(shape->h == 1);                                           \
    range_t ranges[3];                                                         \
    dim4 stride;                                                               \
    int start_idx = tpu_npu_index(output_addr);                                \
    tpu_aligned_stride(                                                        \
        &stride,                                                               \
        start_idx,                                                             \
        shape,                                                                 \
        dtype);                                                                \
    ranges[0] = tpu_bank_range(output_addr, shape->n * stride.n *              \
                               DSIZE(dtype));                                  \
    dim4 index_shape = {.n = shape->n, .c = shape->c, .h = 1, .w = index_w};   \
    tpu_aligned_stride(                                                        \
        &stride,                                                               \
        start_idx,                                                             \
        &index_shape,                                                          \
        index_dtype);                                                          \
    ranges[1] = tpu_bank_range(index_addr, index_shape.n * stride.n *          \
                               DSIZE(index_dtype));                            \
    dim4 param_shape = {                                                       \
        .n = 1, .c = is_param_repeated ? 1 : shape->c, .h = 1, .w = param_w    \
    };                                                                         \
    tpu_aligned_stride(                                                        \
        &stride,                                                               \
        start_idx,                                                             \
        &param_shape,                                                          \
        dtype);                                                                \
    ranges[2] = tpu_bank_range(param_addr, param_shape.n * stride.n *          \
                               DSIZE(dtype));                                  \
    bool conflicting = tpu_any_range_overlapped(ranges, 3);                    \
    atomic_pes_sg_d1hzd_gen_cmd(                                               \
        param_addr,                                                            \
        index_addr,                                                            \
        output_addr,                                                           \
        param_w,                                                               \
        shape->n,                                                              \
        shape->c,                                                              \
        shape->w,                                                              \
        is_param_repeated,                                                     \
        NO_USE,                                                                \
        NO_USE,                                                                \
        false,                                                                 \
        PRECISION(index_dtype),                                                \
        PRECISION(dtype),                                                      \
        conflicting ? PE_S_##name##_hzd : PE_S_##name##_d1coor,                \
        0,                                                                     \
        BDC_NODE);                                                             \
    CHECK_BDC_OVERFLOW;                                                        \
}
TPU_BDC_HW_FLATTEN_BATCH_BCAST(gather, shape->w)
TPU_BDC_HW_FLATTEN_BATCH_BCAST(scatter, param_w)
void tpu_bdc_batch_bcast_w_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated,
    bool          fill_const) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    TPUKERNEL_ASSERT(shape->h == 1);
    range_t ranges[3];
    dim4 stride;
    int start_idx = tpu_npu_index(output_addr);
    tpu_aligned_stride(
        &stride,
        start_idx,
        shape,
        dtype);
    ranges[0] = tpu_bank_range(output_addr, shape->n * stride.n * DSIZE(dtype));
    dim4 index_shape = {.n = shape->n, .c = shape->c, .h = 1, .w = shape->w};
    tpu_aligned_stride(
        &stride,
        start_idx,
        &index_shape,
        index_dtype);
    ranges[1] = tpu_bank_range(index_addr, index_shape.n * stride.n *
                               DSIZE(index_dtype));
    dim4 param_shape = {
        .n = 1, .c = is_param_repeated ? 1 : shape->c, .h = 1, .w = param_w
    };
    tpu_aligned_stride(
        &stride,
        start_idx,
        &param_shape,
        dtype);
    ranges[2] = tpu_bank_range(param_addr, param_shape.n * stride.n *
                               DSIZE(dtype));
    bool conflicting = tpu_any_range_overlapped(ranges, 3);
    atomic_pes_sg_d1hzd_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_w,
        shape->n,
        shape->c,
        shape->w,
        is_param_repeated,
        fill_const,
        C.u32,
        true,
        PRECISION(index_dtype),
        PRECISION(dtype),
        conflicting ? PE_S_gather_hzd : PE_S_gather_d1coor,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_batch_bcast_w_mask_select(
    local_addr_t  output_addr,
    local_addr_t  count_addr,
    local_addr_t  param_addr,
    local_addr_t  mask_addr,
    const dim4   *shape,
    data_type_t   dtype,
    data_type_t   mask_dtype,
    bool          is_param_repeated) {
    TPUKERNEL_ASSERT(mask_dtype == DT_UINT8 || mask_dtype == DT_UINT16 ||
                    mask_dtype == DT_UINT32);
    TPUKERNEL_ASSERT(shape->h == 1);
    range_t ranges[4];
    dim4 stride;
    int start_idx = tpu_npu_index(output_addr);
    tpu_aligned_stride(
        &stride,
        start_idx,
        shape,
        dtype);
    ranges[0] = tpu_bank_range(output_addr, shape->n * stride.n * DSIZE(dtype));
    dim4 count_shape = {.n = shape->n, .c = shape->c, .h = 1, .w = 1};
    tpu_aligned_stride(
        &stride,
        start_idx,
        &count_shape,
        DT_UINT16);
    ranges[1] = tpu_bank_range(count_addr, count_shape.n * stride.n *
                               DSIZE(DT_UINT16));
    tpu_aligned_stride(
        &stride,
        start_idx,
        shape,
        mask_dtype);
    ranges[2] = tpu_bank_range(mask_addr, shape->n * stride.n *
                               DSIZE(mask_dtype));
    dim4 param_shape = {
        .n = 1, .c = is_param_repeated ? 1 : shape->c, .h = 1, .w = shape->w
    };
    tpu_aligned_stride(
        &stride,
        start_idx,
        &param_shape,
        dtype);
    ranges[3] = tpu_bank_range(param_addr, param_shape.n * stride.n *
                               DSIZE(dtype));
    bool conflicting = tpu_any_range_overlapped(ranges, 4);
    atomic_pes_mask_sel_gen_cmd(
        param_addr,
        mask_addr,
        output_addr,
        count_addr,
        shape->w,
        shape->n,
        shape->c,
        is_param_repeated,
        PRECISION(mask_dtype),
        PRECISION(dtype),
        conflicting ? PE_S_mask_selhzd : PE_S_mask_select,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_batch_bcast_h_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    atomic_sgl_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_h,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        is_param_repeated,
        NO_USE,
        NO_USE,
        false,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PE_S_gather_line,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_batch_bcast_h_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated,
    bool          fill_const) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    atomic_sgl_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_h,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        is_param_repeated,
        fill_const,
        C.u32,
        true,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PE_S_gather_line,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_batch_bcast_h_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated) {
    TPUKERNEL_ASSERT(index_dtype == DT_UINT8 || index_dtype == DT_UINT16);
    atomic_sgl_gen_cmd(
        param_addr,
        index_addr,
        output_addr,
        param_h,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        is_param_repeated,
        NO_USE,
        NO_USE,
        false,
        PRECISION(index_dtype),
        PRECISION(dtype),
        PE_S_scatter_line,
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

static inline uint64_t get_numel(const dim4* shape) {
    return (uint64_t)shape->n * shape->c * shape->h * shape->w;
}
static inline bool check_same_numel(const dim4 *src_shape, const dim4 *dst_shape) {
    const uint64_t src_numel = get_numel(src_shape);
    const uint64_t dst_numel = get_numel(dst_shape);
    return src_numel == dst_numel;
}
void tpu_gdma_general_cpy_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *src_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    ASSERT(check_same_numel(src_shape, dst_shape));
    HANDLE_LOCAL_STRIDE(dst_stride_ptr, dst_stride,
        tmp_dst_stride, tpu_npu_index(dst_addr), dst_shape, dtype)
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride,
        tmp_src_stride, src_shape)
    tensor_general_move_gen_cmd(
        src_addr,
        NO_USE,
        src_shape->n,
        src_shape->c,
        src_shape->h,
        src_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        tpu_npu_addr(dst_addr),
        tpu_npu_index(dst_addr),
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_S2L,
        false,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_general_cpy_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    const dim4    *src_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    ASSERT(check_same_numel(src_shape, dst_shape));
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride,
        tmp_dst_stride, dst_shape)
    HANDLE_LOCAL_STRIDE(src_stride_ptr, src_stride,
        tmp_src_stride, tpu_npu_index(src_addr), src_shape, dtype)
    tensor_general_move_gen_cmd(
        tpu_npu_addr(src_addr),
        tpu_npu_index(src_addr),
        src_shape->n,
        src_shape->c,
        src_shape->h,
        src_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr,
        NO_USE,
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_L2S,
        false,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}
void tpu_gdma_general_cpy_L2L(
    local_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *dst_shape,
    const dim4 *src_shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype)
{
    ASSERT(check_same_numel(src_shape, dst_shape));
    dim4 dst_stride_align, src_stride_align;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_aligned_stride(
            &dst_stride_align,
            tpu_npu_index(dst_addr),
            dst_shape,
            dtype);
        dst_stride_ptr = &dst_stride_align;
    }
    if (src_stride == NULL) {
        tpu_aligned_stride(
            &src_stride_align,
            tpu_npu_index(src_addr),
            src_shape,
            dtype);
        src_stride_ptr = &src_stride_align;
    }
    tensor_general_move_gen_cmd(
        src_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(src_addr),
        src_shape->n,
        src_shape->c,
        src_shape->h,
        src_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr % LOCAL_MEM_SIZE,
        tpu_npu_index(dst_addr),
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        GDMA_L2L,
        false,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

static int __get_reverse_offset(int processed, int slice, int size, int stride, int is_reversed){
    return (is_reversed ? (size - processed - slice) : processed) * stride;
}

void tpu_gdma_reverse_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    const dim4 *src_stride_ptr,
    int reverse_axis,
    data_type_t dtype
){
    HANDLE_GLOBAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, shape)
    HANDLE_GLOBAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, shape)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);

    dim4 index = {0,0,0,0};
    dim4 slice_shape;
    dim4 src_offset;
    dim4 dst_offset;
    int typelen = tpu_data_type_size(dtype);
    while (index.n < shape->n) {
        slice_shape.n = MIN(GDMA_MAX_N, shape->n-index.n);
        src_offset.n = __get_reverse_offset(index.n, slice_shape.n, shape->n, src_stride->n, 0);
        dst_offset.n = __get_reverse_offset(index.n, slice_shape.n, shape->n, dst_stride->n, reverse_axis==0);
        while (index.c < shape->c) {
            slice_shape.c = MIN(GDMA_MAX_C, shape->c-index.c);
            src_offset.c = __get_reverse_offset(index.c, slice_shape.c, shape->c, src_stride->c, 0);
            dst_offset.c = __get_reverse_offset(index.c, slice_shape.c, shape->c, dst_stride->c, reverse_axis==1);
            while (index.h < shape->h) {
                slice_shape.h = MIN(GDMA_MAX_H, shape->h-index.h);
                src_offset.h = __get_reverse_offset(index.h, slice_shape.h, shape->h, src_stride->h, 0);
                dst_offset.h = __get_reverse_offset(index.h, slice_shape.h, shape->h, dst_stride->h, reverse_axis==2);
                while (index.w < shape->w) {
                    slice_shape.w = MIN(GDMA_MAX_W, shape->w-index.w);
                    src_offset.w = __get_reverse_offset(index.w, slice_shape.w, shape->w, src_stride->w, 0);
                    dst_offset.w = __get_reverse_offset(index.w, slice_shape.w, shape->w, dst_stride->w, reverse_axis==3);
                    int src_total_offset = src_offset.n + src_offset.c + src_offset.h + src_offset.w;
                    int dst_total_offset = dst_offset.n + dst_offset.c + dst_offset.h + dst_offset.w;
                    tensor_gdma_reverse_gen_cmd(
                        src_addr + src_total_offset * typelen,
                        dst_addr + dst_total_offset * typelen,
                        slice_shape.n, slice_shape.c, slice_shape.h, slice_shape.w,
                        src_stride->n, src_stride->c, src_stride->h,
                        dst_stride->n, dst_stride->c, dst_stride->h,
                        reverse_axis,
                        tpu_get_dma_dtype(dtype),
                        GDMA_S2S,
                        MASTER_THREAD,
                        GDMA_NODE);
                    CHECK_GDMA_OVERFLOW;
                    index.w += slice_shape.w;
                }
                index.h += slice_shape.h;
                index.w = 0;
            }
            index.c += slice_shape.c;
            index.w = 0;
            index.h = 0;
        }
        index.n += slice_shape.n;
        index.w = 0;
        index.h = 0;
        index.c = 0;
    }
}

void tpu_gdma_reverse_S2L(
    local_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    const dim4 *src_stride_ptr,
    int reverse_axis,
    data_type_t dtype
){
    HANDLE_GLOBAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, shape)
    HANDLE_LOCAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, 0, shape, dtype)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);
    tensor_gdma_reverse_gen_cmd(
        src_addr,
        dst_addr,
        shape->n, shape->c, shape->h, shape->w,
        src_stride->n, src_stride->c, src_stride->h,
        dst_stride->n, dst_stride->c, dst_stride->h,
        reverse_axis,
        tpu_get_dma_dtype(dtype),
        GDMA_S2L,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_reverse_L2S(
    system_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    const dim4 *src_stride_ptr,
    int reverse_axis,
    data_type_t dtype
){
    HANDLE_GLOBAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, shape)
    HANDLE_LOCAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, 0, shape, dtype)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);
    tensor_gdma_reverse_gen_cmd(
        src_addr,
        dst_addr,
        shape->n, shape->c, shape->h, shape->w,
        src_stride->n, src_stride->c, src_stride->h,
        dst_stride->n, dst_stride->c, dst_stride->h,
        reverse_axis,
        tpu_get_dma_dtype(dtype),
        GDMA_L2S,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_reverse_L2L(
    local_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    const dim4 *src_stride_ptr,
    int reverse_axis,
    data_type_t dtype
){
    HANDLE_LOCAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, 0, shape, dtype)
    HANDLE_LOCAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, 0, shape, dtype)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);
    tensor_gdma_reverse_gen_cmd(
        tpu_npu_addr(src_addr),
        tpu_npu_addr(dst_addr),
        shape->n, shape->c, shape->h, shape->w,
        src_stride->n, src_stride->c, src_stride->h,
        dst_stride->n, dst_stride->c, dst_stride->h,
        reverse_axis,
        tpu_get_dma_dtype(dtype),
        GDMA_L2L,
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

int tpu_gdma_compress_normal_max_bytes(const dim4 *shape, data_type_t dtype,
                                       bool zero_guard) {
  zero_guard =
      dtype == DT_FP16 ? zero_guard : (dtype == DT_BFP16 ? true : false);
  int size =
      shape->n * shape->c * shape->h * shape->w * tpu_data_type_size(dtype);
  int blk_len = tpu_data_type_bits(dtype) == 16 ? 32 : 16;
  int blk_num = (size + blk_len - 1) / blk_len;
  int kmap_sz = ALIGN(blk_num * (zero_guard ? 2 : 1), NNVLC_ALIGN_BYTES);
  int payload_sz = ALIGN(blk_num * blk_len, NNVLC_ALIGN_BYTES);
  return kmap_sz + payload_sz;
}

void tpu_gdma_compress_normal_L2S(
    global_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *src_stride_ptr,
    data_type_t dtype,
    unsigned char bias0,
    unsigned char bias1,
    bool zero_guard
){
    HANDLE_LOCAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, 0, shape, dtype)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    LOCAL_INDEX_MUST_BE_0(src_addr);
    tensor_normal_compress_gen_cmd(
        src_addr,
        dst_addr,
        shape->n, shape->c, shape->h, shape->w,
        src_stride->n, src_stride->c, src_stride->h,
        bias0, bias1,
        SIGN(dtype),
        zero_guard,
        tpu_get_dma_dtype(dtype),
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

int tpu_gdma_compress_RACU_max_meta_bytes(const dim4* shape, data_type_t dtype){
    int lane_num = DIV_UP(shape->c, NPU_NUM);
    return shape->n*lane_num*shape->h*4;
}

int tpu_gdma_compress_RACU_max_racu_bytes(const dim4* shape, data_type_t dtype, bool zero_guard){
    int lane_num = DIV_UP(shape->c, NPU_NUM);
    dim4 gcw_shape = {1,NPU_NUM, 1, shape->w};
    int gcw = tpu_gdma_compress_normal_max_bytes(&gcw_shape, dtype, zero_guard);
    return shape->n*lane_num*shape->h*gcw;
}

dim4 tpu_gdma_compress_RACU_racu_stride(const dim4* shape, data_type_t dtype, bool zero_guard){
    int lane_num = DIV_UP(shape->c, NPU_NUM);
    dim4 gcw_shape = {1, MIN(NPU_NUM, shape->c), 1, shape->w};
    int gcw = tpu_gdma_compress_normal_max_bytes(&gcw_shape, dtype, zero_guard);
    //[n,lane_num,h,gcw]
    dim4 stride;
    stride.w = 1;
    stride.h = gcw;
    stride.c = shape->h * stride.h;
    stride.n = lane_num * stride.c;
    return stride;
}

dim4 tpu_gdma_compress_RACU_meta_stride(const dim4* shape, data_type_t dtype){
    int lane_num = DIV_UP(shape->c, NPU_NUM);
    //[n,lane_num,h,1]
    dim4 stride;
    stride.h = 1;
    stride.w = 1;
    stride.c = shape->h;
    stride.n = lane_num*shape->h;
    return stride;
}

void tpu_gdma_compress_RACU_L2S(
    global_addr_t dst_racu_addr,
    global_addr_t dst_meta_addr,
    local_addr_t src_addr,
    const dim4* shape,
    const dim4* dst_racu_stride,
    const dim4* dst_meta_stride,
    const dim4* src_stride_ptr,
    data_type_t dtype,
    unsigned char bias0,
    unsigned char bias1,
    bool zero_guard
){
    HANDLE_LOCAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, 0, shape, dtype)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    LOCAL_INDEX_MUST_BE_0(src_addr);
    if (dst_racu_stride != NULL && dst_meta_stride != NULL) {
        tensor_racu_compress_gen_cmd(
            tpu_npu_addr(src_addr),
            dst_racu_addr,
            dst_meta_addr,
            shape->n, shape->c, shape->h, shape->w,
            src_stride->n, src_stride->c, src_stride->h,
            dst_racu_stride->n, dst_racu_stride->c, dst_racu_stride->h,
            dst_meta_stride->n, dst_meta_stride->c,
            bias0, bias1,
            SIGN(dtype), zero_guard, tpu_get_dma_dtype(dtype),
            MASTER_THREAD,
            GDMA_NODE);
    } else {
        dim4 racu_stride, meta_stride;
        const dim4 *dst_racu_stride_ptr = dst_racu_stride;
        const dim4 *dst_meta_stride_ptr = dst_meta_stride;
        if (dst_racu_stride == NULL) {
            racu_stride = tpu_gdma_compress_RACU_racu_stride(shape, dtype, zero_guard);
            dst_racu_stride_ptr = &racu_stride;
        }
        if (dst_meta_stride == NULL) {
            meta_stride = tpu_gdma_compress_RACU_meta_stride(shape, dtype);
            dst_meta_stride_ptr = &meta_stride;
        }
        tensor_racu_compress_gen_cmd(
            tpu_npu_addr(src_addr),
            dst_racu_addr,
            dst_meta_addr,
            shape->n, shape->c, shape->h, shape->w,
            src_stride->n, src_stride->c, src_stride->h,
            dst_racu_stride_ptr->n, dst_racu_stride_ptr->c, dst_racu_stride_ptr->h,
            dst_meta_stride_ptr->n, dst_meta_stride_ptr->c,
            bias0, bias1,
            SIGN(dtype), zero_guard, tpu_get_dma_dtype(dtype),
            MASTER_THREAD,
            GDMA_NODE);
    }
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_decompress_normal_S2L(
    local_addr_t dst_addr,
    global_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    data_type_t dtype,
    unsigned char bias0,
    unsigned char bias1,
    bool zero_guard
){
    HANDLE_LOCAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, 0, shape, dtype)
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);
    LOCAL_INDEX_MUST_BE_0(dst_addr);
    tensor_normal_decompress_gen_cmd(
        tpu_npu_addr(dst_addr),
        src_addr,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride->n, dst_stride->c, dst_stride->h,
        bias0, bias1,
        SIGN(dtype),
        zero_guard,
        tpu_get_dma_dtype(dtype),
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}

void tpu_gdma_decompress_RACU_S2L(
    local_addr_t dst_addr,
    global_addr_t src_racu_addr,
    global_addr_t src_meta_addr,
    const dim4* shape,
    const dim4* dst_stride_ptr,
    const dim4* src_racu_stride,
    const dim4* src_meta_stride,
    data_type_t dtype,
    unsigned char bias0,
    unsigned char bias1,
    bool zero_guard
){
    HANDLE_LOCAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, 0, shape, dtype)
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);
    LOCAL_INDEX_MUST_BE_0(dst_addr);
    if (src_racu_stride != NULL && src_meta_stride != NULL) {
        tensor_racu_decompress_gen_cmd(
            tpu_npu_addr(dst_addr),
            src_racu_addr,
            src_meta_addr,
            shape->n, shape->c, shape->h, shape->w,
            dst_stride->n, dst_stride->c, dst_stride->h,
            src_racu_stride->n, src_racu_stride->c, src_racu_stride->h,
            src_meta_stride->n, src_meta_stride->c,
            bias0, bias1,
            SIGN(dtype), zero_guard, tpu_get_dma_dtype(dtype),
            MASTER_THREAD,
            GDMA_NODE);
    } else {
        dim4 racu_stride, meta_stride;
        const dim4 *src_racu_stride_ptr = src_racu_stride;
        const dim4 *src_meta_stride_ptr = src_meta_stride;
        if (src_racu_stride == NULL) {
            racu_stride = tpu_gdma_compress_RACU_racu_stride(shape, dtype, zero_guard);
            src_racu_stride_ptr = &racu_stride;
        }
        if (src_meta_stride == NULL) {
            meta_stride = tpu_gdma_compress_RACU_meta_stride(shape, dtype);
            src_meta_stride_ptr = &meta_stride;
        }
        tensor_racu_decompress_gen_cmd(
            tpu_npu_addr(dst_addr),
            src_racu_addr,
            src_meta_addr,
            shape->n, shape->c, shape->h, shape->w,
            dst_stride->n, dst_stride->c, dst_stride->h,
            src_racu_stride_ptr->n, src_racu_stride_ptr->c, src_racu_stride_ptr->h,
            src_meta_stride_ptr->n, src_meta_stride_ptr->c,
            bias0, bias1,
            SIGN(dtype), zero_guard, tpu_get_dma_dtype(dtype),
            MASTER_THREAD,
            GDMA_NODE);
    }
    tensor_racu_decompress_gen_cmd(
        tpu_npu_addr(dst_addr),
        src_racu_addr,
        src_meta_addr,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride->n, dst_stride->c, dst_stride->h,
        src_racu_stride->n, src_racu_stride->c, src_racu_stride->h,
        src_meta_stride->n, src_meta_stride->c,
        bias0, bias1,
        SIGN(dtype), zero_guard, tpu_get_dma_dtype(dtype),
        MASTER_THREAD,
        GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
}


void tpu_flush_cache(system_addr_t address,
                     unsigned long long size)
{
#ifdef USING_FAKE_DDR_MODE
    if (!(address >= L2_SRAM_START_ADDR && address < (L2_SRAM_START_ADDR + L2_SRAM_SIZE)))
        address += PLD_BASE_ADDR;
#endif
    flush_cache(address, size);
}

void tpu_invalidate_cache(system_addr_t address,
                          unsigned long long size)
{
    invalidate_cache(address, size);
}
int tpu_cache_line_size() { return CACHE_LINE_SIZE; }

void tpu_sdma_cpy_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    tpu_vsdma_cpy_S2S(
    dst_addr, src_addr, shape,
    dst_stride, src_stride, dtype, DEFAULT_SDMA_PORT);
}
RTM_EXPORT(tpu_sdma_cpy_S2S);

void tpu_vsdma_cpy_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype,
    int            port_id) {
#if defined(C2C_USE_DESCRIPTOR)
    port_id = DEFAULT_SDMA_PORT;
#endif
    dim4 dst_stride_continuous, src_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_continuous, shape);
        src_stride_ptr = &src_stride_continuous;
    }
    tpu_mem_check_tensor(dst_addr, shape, dst_stride_ptr, dtype);
    tpu_mem_check_tensor(src_addr, shape, src_stride_ptr, dtype);
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 src_offset_n = (u64)nidx * src_stride_ptr->n * dsize;
        u64 src_offset_c = (u64)cidx * src_stride_ptr->c * dsize;
        u64 src_offset_h = (u64)hidx * src_stride_ptr->h * dsize;
        u64 src_offset_w = (u64)widx * src_stride_ptr->w * dsize;
        u64 src_offset = src_offset_n + src_offset_c + src_offset_h + src_offset_w;
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        sdma_tensor_general_move_gen_cmd(
            src_addr + src_offset,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            dst_addr + dst_offset,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            false,
            port_id,
            &id_node);
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}

void tpu_sdma_cpy_nc_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    tpu_vsdma_cpy_nc_trans_S2S(
    dst_addr, src_addr, dst_shape,
    dst_stride, src_stride, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_cpy_nc_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype,
    int            port_id) {
    dim4 dst_stride_continuous, src_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, dst_shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    tpu_mem_check_tensor(dst_addr, dst_shape, dst_stride_ptr, dtype);
    if (src_stride == NULL) {
        dim4 shape = {
            .n = dst_shape->c, .c = dst_shape->n,
            .h = dst_shape->h, .w = dst_shape->w
        };
        tpu_continuous_stride(&src_stride_continuous, &shape);
        src_stride_ptr = &src_stride_continuous;
    }
    sdma_tensor_general_move_gen_cmd(
        src_addr,
        dst_shape->c,
        dst_shape->n,
        dst_shape->h,
        dst_shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        src_stride_ptr->w,
        tpu_get_dma_dtype(dtype),
        dst_addr,
        dst_shape->n,
        dst_shape->c,
        dst_shape->h,
        dst_shape->w,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        dst_stride_ptr->w,
        true,
        port_id,
        &id_node);
}

void tpu_sdma_cpy_cw_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype) {
    tpu_vsdma_cpy_cw_trans_S2S(
    dst_addr, src_addr, dst_shape,
    dst_stride, src_stride, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_cpy_cw_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype,
    int            port_id) {
    tpu_mem_check_tensor(dst_addr, dst_shape, dst_stride, dtype);
    if (dst_stride == NULL && src_stride == NULL)
        sdma_general_cwtrans_gen_cmd(
            src_addr,
            dst_addr,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            false,
            port_id,
            &id_node);
    else {
        dim4 dst_stride_continuous, src_stride_continuous;
        const dim4 *dst_stride_ptr = dst_stride;
        const dim4 *src_stride_ptr = src_stride;
        if (dst_stride == NULL) {
            tpu_continuous_stride(&dst_stride_continuous, dst_shape);
            dst_stride_ptr = &dst_stride_continuous;
        } else
            TPUKERNEL_ASSERT(dst_stride->w == 1);
        if (src_stride == NULL) {
            dim4 shape = {
                .n = dst_shape->n, .c = dst_shape->w,
                .h = dst_shape->h, .w = dst_shape->c
            };
            tpu_continuous_stride(&src_stride_continuous, &shape);
            src_stride_ptr = &src_stride_continuous;
        } else
            TPUKERNEL_ASSERT(src_stride->w == 1);
        sdma_general_cwtrans_gen_cmd(
            src_addr,
            dst_addr,
            dst_shape->n,
            dst_shape->w,
            dst_shape->h,
            dst_shape->c,
            tpu_get_dma_dtype(dtype),
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            false,
            port_id,
            &id_node);
    }
}

void tpu_sdma_set_C_system(
    system_addr_t  dst_addr,
    scalar_t       C,
    const dim4    *shape,
    const dim4    *dst_stride,
    data_type_t    dtype) {
    tpu_vsdma_set_C_system(
    dst_addr, C, shape,
    dst_stride, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_set_C_system(
    system_addr_t  dst_addr,
    scalar_t       C,
    const dim4    *shape,
    const dim4    *dst_stride,
    data_type_t    dtype,
    int            port_id) {
    dim4 dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        sdma_fill_constant_gen_global_cmd_stride(
            dst_addr + dst_offset,
            &C,
            tpu_get_dma_dtype(dtype),
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            true,
            port_id,
            &id_node);
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}

void tpu_sdma_h_gather_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    tpu_vsdma_h_gather_S2S(
    output_addr, param_addr, index_addr,
    C, shape, param_h, output_stride, param_stride, index_stride,
    dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_h_gather_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype,
    int            port_id) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        sdma_tensor_gather_gen_cmd(
            param_addr,
            index_addr,
            output_addr,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            port_id,
            &id_node);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        sdma_tensor_gather_gen_cmd(
            param_addr,
            index_addr,
            output_addr,
            C.u32,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            port_id,
            &id_node);
    }
}

void tpu_sdma_h_scatter_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype) {
    tpu_vsdma_h_scatter_S2S(
    output_addr, param_addr, index_addr,
    shape, param_h, output_stride, param_stride, index_stride,
    dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_h_scatter_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype,
    int            port_id) {
    TPUKERNEL_ASSERT(shape->n == 1);
    if (output_stride == NULL && param_stride == NULL && index_stride == NULL)
        sdma_tensor_scatter_gen_cmd(
            param_addr,
            index_addr,
            output_addr,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            NO_USE,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            false,
            port_id,
            false,
            &id_node);
    else {
        dim4 output_stride_continuous, param_stride_continuous, idx_stride;
        const dim4 *output_stride_ptr = output_stride;
        const dim4 *param_stride_ptr = param_stride;
        const dim4 *index_stride_ptr = index_stride;
        if (output_stride == NULL) {
            tpu_continuous_stride(&output_stride_continuous, shape);
            output_stride_ptr = &output_stride_continuous;
        } else
            TPUKERNEL_ASSERT(output_stride->w == 1);
        if (param_stride == NULL) {
            dim4 param_shape = {
                .n = shape->n, .c = shape->c, .h = param_h, .w = shape->w
            };
            tpu_continuous_stride(&param_stride_continuous, &param_shape);
            param_stride_ptr = &param_stride_continuous;
        } else
            TPUKERNEL_ASSERT(param_stride->w == 1);
        if (index_stride == NULL) {
            dim4 index_shape = {
                .n = shape->n, .c = shape->c, .h = shape->h, .w = 1
            };
            tpu_continuous_stride(&idx_stride, &index_shape);
            index_stride_ptr = &idx_stride;
        } else
            TPUKERNEL_ASSERT(index_stride->h == 1);
        sdma_tensor_scatter_gen_cmd(
            param_addr,
            index_addr,
            output_addr,
            shape->c,
            param_h,
            shape->w,
            shape->h,
            NO_USE, //start_pos
            param_stride_ptr->c,
            param_stride_ptr->h,
            index_stride_ptr->c,
            index_stride_ptr->h,
            output_stride_ptr->c,
            output_stride_ptr->h,
            tpu_get_dma_dtype(dtype),
            false,
            false,
            true,
            port_id,
            false,
            &id_node);
    }
}

void tpu_sdma_system_cpy(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    unsigned int   count,
    data_type_t    dtype) {
    tpu_vsdma_system_cpy(
    dst_addr, src_addr, count, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_system_cpy(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    unsigned int   count,
    data_type_t    dtype,
    int            port_id) {
#if defined(C2C_USE_DESCRIPTOR)
    port_id = DEFAULT_SDMA_PORT;
#endif
    sdma_general_gen_cmd(
        src_addr,
        dst_addr,
        tpu_get_dma_dtype(dtype),
        count,
        false,
        port_id,
        &id_node);
}

void tpu_sdma_system_set(
    system_addr_t  dst_addr,
    scalar_t       C,
    unsigned int   count,
    data_type_t    dtype) {
    tpu_vsdma_system_set(
    dst_addr, C, count, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_system_set(
    system_addr_t  dst_addr,
    scalar_t       C,
    unsigned int   count,
    data_type_t    dtype,
    int            port_id) {
#if defined(C2C_USE_DESCRIPTOR)
    port_id = DEFAULT_SDMA_PORT;
#endif
    sdma_general_gen_cmd(
        C.u32,
        dst_addr,
        tpu_get_dma_dtype(dtype),
        count,
        true,
        port_id,
        &id_node);
}

void tpu_sdma_reverse_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reverse_axis,
    data_type_t dtype) {
    tpu_vsdma_reverse_S2S(
    dst_addr, src_addr, shape,
    dst_stride, src_stride,
    reverse_axis, dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_reverse_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride_ptr,
    const dim4 *src_stride_ptr,
    int reverse_axis,
    data_type_t dtype,
    int port_id) {
    HANDLE_GLOBAL_STRIDE(src_stride, src_stride_ptr, tmp_src_stride, shape)
    HANDLE_GLOBAL_STRIDE(dst_stride, dst_stride_ptr, tmp_dst_stride, shape)
    ASSERT_INFO(src_stride->w == 1, "src_stride->w=%d", src_stride->w);
    ASSERT_INFO(dst_stride->w == 1, "dst_stride->w=%d", dst_stride->w);

    dim4 index = {0,0,0,0};
    dim4 slice_shape;
    dim4 src_offset;
    dim4 dst_offset;
    int typelen = tpu_data_type_size(dtype);
    while (index.n < shape->n) {
        slice_shape.n = MIN(GDMA_MAX_N, shape->n-index.n);
        src_offset.n = __get_reverse_offset(index.n, slice_shape.n, shape->n, src_stride->n, 0);
        dst_offset.n = __get_reverse_offset(index.n, slice_shape.n, shape->n, dst_stride->n, reverse_axis==0);
        while (index.c < shape->c) {
            slice_shape.c = MIN(GDMA_MAX_C, shape->c-index.c);
            src_offset.c = __get_reverse_offset(index.c, slice_shape.c, shape->c, src_stride->c, 0);
            dst_offset.c = __get_reverse_offset(index.c, slice_shape.c, shape->c, dst_stride->c, reverse_axis==1);
            while (index.h < shape->h) {
                slice_shape.h = MIN(GDMA_MAX_H, shape->h-index.h);
                src_offset.h = __get_reverse_offset(index.h, slice_shape.h, shape->h, src_stride->h, 0);
                dst_offset.h = __get_reverse_offset(index.h, slice_shape.h, shape->h, dst_stride->h, reverse_axis==2);
                while (index.w < shape->w) {
                    slice_shape.w = MIN(GDMA_MAX_W, shape->w-index.w);
                    src_offset.w = __get_reverse_offset(index.w, slice_shape.w, shape->w, src_stride->w, 0);
                    dst_offset.w = __get_reverse_offset(index.w, slice_shape.w, shape->w, dst_stride->w, reverse_axis==3);
                    int src_total_offset = src_offset.n + src_offset.c + src_offset.h + src_offset.w;
                    int dst_total_offset = dst_offset.n + dst_offset.c + dst_offset.h + dst_offset.w;
                    sdma_tensor_reverse_gen_cmd(
                        src_addr + src_total_offset * typelen,
                        dst_addr + dst_total_offset * typelen,
                        slice_shape.n, slice_shape.c, slice_shape.h, slice_shape.w,
                        src_stride->n, src_stride->c, src_stride->h,
                        dst_stride->n, dst_stride->c, dst_stride->h,
                        reverse_axis,
                        tpu_get_dma_dtype(dtype),
                        port_id,
                        &id_node);
                    index.w += slice_shape.w;
                }
                index.h += slice_shape.h;
                index.w = 0;
            }
            index.c += slice_shape.c;
            index.w = 0;
            index.h = 0;
        }
        index.n += slice_shape.n;
        index.w = 0;
        index.h = 0;
        index.c = 0;
    }
}

void tpu_sdma_mask_select_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    global_addr_t  mask_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype) {
    tpu_vsdma_mask_select_S2S(
    dst_addr, src_addr, mask_addr,
    shape, data_dtype, mask_dtype, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_mask_select_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    global_addr_t  mask_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype,
    int            port_id) {
    sdma_tensor_general_move_with_mask_gen_cmd(
        src_addr,
        mask_addr,
        dst_addr,
        tpu_get_dma_dtype(data_dtype),
        tpu_get_dma_dtype(mask_dtype),
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        port_id,
        &id_node);
}

void tpu_sdma_nonzero_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx) {
    tpu_vsdma_nonzero_S2S(
    dst_addr, src_addr, shape,
    data_dtype, base_idx, DEFAULT_SDMA_PORT);
}

void tpu_vsdma_nonzero_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx,
    int            port_id) {
    sdma_tensor_move_nonzero_gen_cmd(
        src_addr,
        dst_addr,
        tpu_get_dma_dtype(data_dtype),
        GDMA_INT32,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        base_idx,
        port_id,
        &id_node);
}

unsigned int tpu_sdma_get_filter_num() {
  return get_sdma_filter_res_num_gen_cmd(DEFAULT_SDMA_PORT, &id_node);
}

unsigned int tpu_vsdma_get_filter_num(int port_id) {
  return get_sdma_filter_res_num_gen_cmd(port_id, &id_node);
}

void tpu_cdma_send(
    int            dst_chipid,
    int            src_chipid,
    system_addr_t  src_addr,
    int            src_n,
    int            src_c,
    int            src_h,
    int            src_w,
    int            src_n_stride,
    int            src_c_stride,
    int            src_h_stride,
    int            opcode,
    data_type_t    dtype)
{
#ifdef DISABLE_CDMA
    if (src_chipid == dst_chipid) {
        FW_ERR("%s: same chipid, dst_chipid = %d, src_addr = 0x%llx, "
               "src_n = %d, src_c = %d, src_h = %d, src_w = %d, "
               "src_n_stride = %d, src_c_stride = %d, src_h_stride = %d, "
               "opcode = %d, dtype = %d\n",
               __func__, dst_chipid, src_addr,
               src_n, src_c, src_h, src_w,
               src_n_stride, src_c_stride, src_h_stride,
               opcode, dtype);
        return;
    }
#endif

    cdma_send_cmodel_gen_cmd(
        dst_chipid,
        src_addr,
        (unsigned short)src_n,
        (unsigned short)src_c,
        (unsigned int)src_h,
        (unsigned int)src_w,
        (unsigned int)src_n_stride,
        (unsigned int)src_c_stride,
        (unsigned int)src_h_stride,
        get_cdma_format_from_precision(PRECISION(dtype)),
        1,// stride_enable
        opcode,
        0,// nchw_copy
        &id_node
#if defined(SG_TV_GEN)
        , 1,1
#endif
        );
}

void tpu_cdma_lossy_compress(
    int            dst_chipid,
    int            src_chipid,
    system_addr_t  src_addr,
    unsigned short src_n,
    unsigned short src_c,
    unsigned int   src_h,
    unsigned int   src_w,
    unsigned int   src_n_stride,
    unsigned int   src_c_stride,
    unsigned int   src_h_stride,
    int            opcode)
{
#ifdef DISABLE_CDMA
    if (src_chipid == dst_chipid) {
        FW_ERR("%s: same chipid, dst_chipid = %d, src_addr = 0x%llx, "
               "src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
               "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
               "opcode = %d\n",
               __func__, dst_chipid, src_addr,
               src_n, src_c, src_h, src_w,
               src_n_stride, src_c_stride, src_h_stride,
               opcode);
        return;
    }
#endif
    cdma_lossy_compress_gen_cmd(
        dst_chipid,
        src_addr,
        src_n,
        src_c,
        src_h,
        src_w,
        src_n_stride,
        src_c_stride,
        src_h_stride,
        CDMA_DTYPE_FP32,
        1,// stride_enable
        opcode,
        0,// nchw_copy
        &id_node
    );
}

void tpu_cdma_lossy_decompress(
    int            dst_chipid,
    int            src_chipid,
    system_addr_t  src_addr,
    unsigned short src_n,
    unsigned short src_c,
    unsigned int   src_h,
    unsigned int   src_w,
    unsigned int   src_n_stride,
    unsigned int   src_c_stride,
    unsigned int   src_h_stride,
    int            opcode)
{
#ifdef DISABLE_CDMA
    if (src_chipid == dst_chipid) {
        FW_ERR("%s: same chipid, dst_chipid = %d, src_addr = 0x%llx, "
               "src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
               "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
               "opcode = %d\n",
               __func__, dst_chipid, src_addr,
               src_n, src_c, src_h, src_w,
               src_n_stride, src_c_stride, src_h_stride,
               opcode);
        return;
    }
#endif
    cdma_lossy_decompress_gen_cmd(
        dst_chipid,
        src_addr,
        src_n,
        src_c,
        src_h,
        src_w,
        src_n_stride,
        src_c_stride,
        src_h_stride,
        CDMA_DTYPE_FP20,
        opcode,
        1,// stride_enable
        0,// nchw_copy
        &id_node
    );
}

void tpu_cdma_recv(
    int            dst_chipid,
    int            src_chipid,
    system_addr_t  dst_addr,
    int            dst_n,
    int            dst_c,
    int            dst_h,
    int            dst_w,
    int            dst_n_stride,
    int            dst_c_stride,
    int            dst_h_stride,
    system_addr_t  input_addr,
    int            opcode,
    data_type_t    dtype)
{
#ifdef DISABLE_CDMA
    if (dst_chipid == src_chipid) {
        FW_ERR("%s: same chipid, src_chipid = %d, dst_chipid = %d, dst_addr = 0x%llx, "
               "dst_n = %d, dst_c = %d, dst_h = %d, dst_w = %d, "
               "dst_n_stride = %d, dst_c_stride = %d, dst_h_stride = %d, "
               "input_addr = 0x%llx, opcode = %d, dtype = %d\n",
               __func__, src_chipid, dst_chipid, dst_addr,
               dst_n, dst_c, dst_h, dst_w,
               dst_n_stride, dst_c_stride, dst_h_stride,
               input_addr, opcode, dtype);
	return;
    }
#endif
    cdma_recv_gen_cmd(
        dst_chipid,
        src_chipid,
        dst_addr,
        (unsigned short)dst_n,
        (unsigned short)dst_c,
        (unsigned int)dst_h,
        (unsigned int)dst_w,
        (unsigned int)dst_n_stride,
        (unsigned int)dst_c_stride,
        (unsigned int)dst_h_stride,
        opcode,
        get_cdma_format_from_precision(PRECISION(dtype)),
        1,
        &id_node
#if defined(SG_TV_GEN)
        , 1,1,CDMA_SEND
#endif
        );
}

void tpu_cdma_write(
    int             src_chipid,
    int             dst_chipid,
    system_addr_t   src_addr,
    system_addr_t   dst_addr,
    unsigned short  src_n,
    unsigned short  src_c,
    unsigned int    src_h,
    unsigned int    src_w,
    unsigned int    src_n_stride,
    unsigned int    src_c_stride,
    unsigned int    src_h_stride,
    unsigned short  dst_n,
    unsigned short  dst_c,
    unsigned int    dst_h,
    unsigned int    dst_w,
    unsigned int    dst_n_stride,
    unsigned int    dst_c_stride,
    unsigned int    dst_h_stride,
    int             is_fill_const,
    int             const_val,
    data_type_t     dtype,
    int             stride_enable,
    int             nchw_copy)
{
    cdma_write_gen_cmd(
        src_chipid,
        dst_chipid,
        src_addr,
        dst_addr,
        src_n,
        src_c,
        src_h,
        src_w,
        src_n_stride,
        src_c_stride,
        src_h_stride,
        dst_n,
        dst_c,
        dst_h,
        dst_w,
        dst_n_stride,
        dst_c_stride,
        dst_h_stride,
        is_fill_const,
        const_val,
        get_cdma_format_from_precision(PRECISION(dtype)),
        stride_enable,
        nchw_copy,
        &id_node
        );
}

void tpu_gdma_lossy_compress_L2S(
    system_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT(shape != NULL);
    ASSERT(src_stride->w == 1);
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, shape)
    const int src_npu_idx = tpu_npu_index(src_addr);
    HANDLE_LOCAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, src_npu_idx, shape, DT_FP32)
    gdma_lossy_compress_gen_cmd(
        tpu_npu_addr(src_addr),
        src_npu_idx,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        dst_addr,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        GDMA_L2S,
        MASTER_THREAD,
        GDMA_NODE);
}

void tpu_gdma_lossy_compress_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT(shape != NULL);
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, shape)
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, shape)
    gdma_lossy_compress_gen_cmd(
        src_addr,
        NO_USE,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        dst_addr,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        GDMA_S2S,
        MASTER_THREAD,
        GDMA_NODE);
}

void tpu_gdma_lossy_decompress_S2L(
    local_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT(shape != NULL);
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, shape)
    const int dst_npu_idx = tpu_npu_index(dst_addr);
    HANDLE_LOCAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, dst_npu_idx, shape, DT_FP32)
    gdma_lossy_decompress_gen_cmd(
        src_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        tpu_npu_addr(dst_addr),
        dst_npu_idx,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        GDMA_S2L,
        MASTER_THREAD,
        GDMA_NODE);
}

void tpu_gdma_lossy_decompress_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(shape != NULL);
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, shape)
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, shape)
    gdma_lossy_decompress_gen_cmd(
        src_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        dst_addr,
        NO_USE,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        GDMA_S2S,
        MASTER_THREAD,
        GDMA_NODE);
}

void tpu_sdma_lossy_compress(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT(shape != NULL);
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, shape)
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, shape)
    sdma_lossy_compress_gen_cmd(
        src_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        dst_addr,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        DEFAULT_SDMA_PORT,
        &id_node);
}

void tpu_sdma_lossy_decompress(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT(shape != NULL);
    HANDLE_GLOBAL_STRIDE(dst_stride_ptr, dst_stride, tmp_dst_stride, shape)
    HANDLE_GLOBAL_STRIDE(src_stride_ptr, src_stride, tmp_src_stride, shape)
    sdma_lossy_decompress_gen_cmd(
        src_addr,
        shape->n,
        shape->c,
        shape->h,
        shape->w,
        src_stride_ptr->n,
        src_stride_ptr->c,
        src_stride_ptr->h,
        dst_addr,
        dst_stride_ptr->n,
        dst_stride_ptr->c,
        dst_stride_ptr->h,
        DEFAULT_SDMA_PORT,
        &id_node);
}

void tpu_gdma_cpy_reduce_L12L2(
    system_addr_t dst_addr,
    local_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;

    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_aligned_stride(&src_stride_aligned, 0, shape, dtype);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    const int dsize = tpu_data_type_size(dtype);
    // split N, C, H, W due the limit of GDMA
    int nidx = 0, cidx = 0, hidx = 0, widx = 0;
    while (nidx < shape->n) {
        int real_nslice = MIN(shape->n - nidx, GDMA_MAX_N);
        int real_cslice = MIN(shape->c - cidx, GDMA_MAX_C);
        int real_hslice = MIN(shape->h - hidx, GDMA_MAX_H);
        int real_wslice = MIN(shape->w - widx, GDMA_MAX_W);
        u64 dst_offset_n = (u64)nidx * dst_stride_ptr->n * dsize;
        u64 dst_offset_c = (u64)cidx * dst_stride_ptr->c * dsize;
        u64 dst_offset_h = (u64)hidx * dst_stride_ptr->h * dsize;
        u64 dst_offset_w = (u64)widx * dst_stride_ptr->w * dsize;
        u64 dst_offset = dst_offset_n + dst_offset_c + dst_offset_h + dst_offset_w;
        int src_offset_n = nidx * src_stride_ptr->n * dsize;
        int src_offset_c = tpu_unified_c_offset(cidx, src_stride_ptr->c, dtype);
        int src_offset_h = hidx * src_stride_ptr->h * dsize;
        int src_offset_w = widx * src_stride_ptr->w * dsize;
        int src_offset = src_offset_n + src_offset_c + src_offset_h + src_offset_w;
        tensor_stride_move_reduce_gen_cmd(
            tpu_npu_addr(src_addr + src_offset),
            tpu_npu_index(src_addr + src_offset),
            dst_addr + dst_offset,
            real_nslice,
            real_cslice,
            real_hslice,
            real_wslice,
            src_stride_ptr->n,
            src_stride_ptr->c,
            src_stride_ptr->h,
            src_stride_ptr->w,
            dst_stride_ptr->n,
            dst_stride_ptr->c,
            dst_stride_ptr->h,
            dst_stride_ptr->w,
            tpu_get_dma_dtype(dtype),
            GDMA_L2S,
            false,
            reduce_psum_op,
            reduce_opcode,
            MASTER_THREAD,
            GDMA_NODE);
    CHECK_GDMA_OVERFLOW;
        widx += GDMA_MAX_W;
        if (widx < shape->w) continue;
        widx = 0;
        hidx += GDMA_MAX_H;
        if (hidx < shape->h) continue;
        hidx = 0;
        cidx += GDMA_MAX_C;
        if (cidx < shape->c) continue;
        cidx = 0;
        nidx += GDMA_MAX_N;
    }
}

void tpu_gdma_cpy_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;

    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    tensor_general_move_reduce_gen_cmd(
        src_addr, 0,
        shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h, src_stride_ptr->w,
        tpu_get_dma_dtype(dtype), dst_addr, 0,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h, dst_stride_ptr->w,
        GDMA_S2S, 0, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_gdma_cpy_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    tensor_general_move_reduce_gen_cmd(
        src_addr, 0,
        shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h, src_stride_ptr->w,
        tpu_get_dma_dtype(dtype), dst_addr, 0,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h, dst_stride_ptr->w,
        GDMA_S2S, 0, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_sdma_cpy_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;

    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_tensor_reduce_gen_cmd(
        src_addr,
        shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h, src_stride_ptr->w,
        tpu_get_dma_dtype(dtype), dst_addr,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h, dst_stride_ptr->w,
        0, reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_sdma_cpy_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_tensor_reduce_gen_cmd(
        src_addr,
        shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h, src_stride_ptr->w,
        tpu_get_dma_dtype(dtype), dst_addr,
        shape->n, shape->c, shape->h, shape->w,
        dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h, dst_stride_ptr->w,
        0, reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_gdma_lossy_compress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    gdma_lossy_compress_reduce_gen_cmd(
        src_addr, 0, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        GDMA_S2S, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_gdma_lossy_compress_reduce_L12L2(
    system_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_aligned_stride(&src_stride_aligned, 0, shape, DT_FP32);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    gdma_lossy_compress_reduce_gen_cmd(
        tpu_npu_addr(src_addr), tpu_npu_addr(src_addr), shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        GDMA_L2S, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_gdma_lossy_compress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    gdma_lossy_compress_reduce_gen_cmd(
        src_addr, 0, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        GDMA_S2S, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_sdma_lossy_compress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_lossy_compress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_sdma_lossy_compress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_lossy_compress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_gdma_lossy_decompress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    gdma_lossy_decompress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, 0, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        GDMA_S2S, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_sdma_lossy_decompress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_lossy_decompress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_gdma_lossy_decompress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    gdma_lossy_decompress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, 0, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        GDMA_S2S, reduce_psum_op, reduce_opcode, MASTER_THREAD, GDMA_NODE);
}

void tpu_sdma_lossy_decompress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    dim4 src_stride_aligned, dst_stride_continuous;
    const dim4 *dst_stride_ptr = dst_stride;
    const dim4 *src_stride_ptr = src_stride;
    TPUKERNEL_ASSERT(IN_L2_SRAM(src_addr));
    TPUKERNEL_ASSERT(IN_L2_SRAM(dst_addr));
    TPUKERNEL_ASSERT(reduce_psum_op < 2); // wo, rw
    TPUKERNEL_ASSERT(reduce_opcode < 5); // nop, add, mul, max, min

    if (src_stride == NULL) {
        tpu_continuous_stride(&src_stride_aligned, shape);
        src_stride_ptr = &src_stride_aligned;
    }
    if (dst_stride == NULL) {
        tpu_continuous_stride(&dst_stride_continuous, shape);
        dst_stride_ptr = &dst_stride_continuous;
    }
    sdma_lossy_decompress_reduce_gen_cmd(
        src_addr, shape->n, shape->c, shape->h, shape->w,
        src_stride_ptr->n, src_stride_ptr->c, src_stride_ptr->h,
        dst_addr, dst_stride_ptr->n, dst_stride_ptr->c, dst_stride_ptr->h,
        reduce_psum_op, reduce_opcode, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_bd_set_random_gen_seed(u64 seed) {
    atomic_set_bd_random_gen_seed(seed);
}

void tpu_bd_rand_seed_gen() {
    atomic_bd_rand_seed_gen_cmd(MASTER_THREAD, &id_node);
}

void tpu_set_gdma_id(int gdma_id) {
    atomic_set_gdma_id_gen_cmd(gdma_id, GDMA_NODE);
    GDMA_NODE->gdma_cmd_id = gdma_id;
}
void tpu_set_bd_id(int bdc_id) {
    atomic_set_bdid_gen_cmd(bdc_id, BDC_NODE);
    BDC_NODE->bd_cmd_id = bdc_id;
}

int tpu_physical_core_id() {
    return CORE_ID;
}
int tpu_physical_core_num() {
    return MAX_TPU_CORE_NUM;
}

#ifdef USING_CMODEL
void tpu_cdma_fake_all_reduce(
    int            dst_chipid,
    u64            src_addr,
    int            dst_n,
    int            dst_c,
    int            dst_h,
    int            dst_w,
    int            dst_n_stride,
    int            dst_c_stride,
    int            dst_h_stride,
    int            opcode,
    data_type_t    dtype)
{
    cdma_fake_all_reduce_gen_cmd(
        dst_chipid,
        src_addr,
        dst_n,
        dst_c,
        dst_h,
        dst_w,
        dst_n_stride,
        dst_c_stride,
        dst_h_stride,
        opcode,
        PRECISION(dtype),
        &id_node);
}

void tpu_cdma_fake_p2p(
    int            dst_chipid,
    u64            src_addr,
    int            dst_n,
    int            dst_c,
    int            dst_h,
    int            dst_w,
    int            dst_n_stride,
    int            dst_c_stride,
    int            dst_h_stride,
    int            opcode,
    data_type_t    dtype)
{
    cdma_fake_p2p_gen_cmd(
        dst_chipid,
        src_addr,
        dst_n,
        dst_c,
        dst_h,
        dst_w,
        dst_n_stride,
        dst_c_stride,
        dst_h_stride,
        opcode,
        PRECISION(dtype),
        &id_node);
}

#endif

void tpu_bdc_fp_sinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    // DST = EXP(DST)
    tpu_bdc_fp_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);

    // WORK0 = 1 / DST
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_C_div(
        work0_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);

    // DST = DST - WORK0
    tpu_bdc_fp_sub(
        dst_addr,
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);

    // DST = DST *1/2
    scalar_t C2 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        tpu_cast(C2, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp_cosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    // DST = EXP(DST)
    tpu_bdc_fp_exp(
        dst_addr,
        src_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);

    // WORK0 = 1 / DST
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_C_div(
        work0_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);

    // DST = DST + WORK0
    tpu_bdc_fp_add(
        dst_addr,
        dst_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);

    // DST = DST *1/2
    scalar_t C2 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        tpu_cast(C2, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}
void tpu_bdc_fp_tanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype) {
    // DST = SRC * 2
    tpu_bdc_fp_add(
        dst_addr,
        src_addr,
        src_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);

    // DST = EXP(DST)
    tpu_bdc_fp_exp(
        dst_addr,
        dst_addr,
        work0_addr,
        work1_addr,
        coeff_addr,
        shape,
        dtype);

    // WROK0 = DST + 1
    scalar_t C1 = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work0_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);

    // WORK1 = 2 / WORK0
    scalar_t C2 = {.f32 = 2.f};
    tpu_bdc_fp_C_div(
        work1_addr,
        work0_addr,
        tpu_cast(C2, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);

    // DST = 1 - WORK1
    tpu_bdc_fp_C_sub(
        dst_addr,
        work1_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        shape,
        NULL,
        NULL,
        dtype);
}

RTM_EXPORT(tpu_bdc_fp_add);
RTM_EXPORT(tpu_bdc_fp_mul_C);
RTM_EXPORT(tpu_bdc_max);
RTM_EXPORT(tpu_bdc_max_C);
RTM_EXPORT(tpu_bdc_min_C);
RTM_EXPORT(tpu_bdc_int_add_C);
RTM_EXPORT(tpu_bdc_fp_mul);