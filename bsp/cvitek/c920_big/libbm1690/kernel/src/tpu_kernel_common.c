#include <rtthread.h>
#include "tpu_kernel.h"
#include <string.h>

#ifdef USING_CMODEL
// use for user custom ops
__attribute__((constructor)) void init_cmodel_firmware() {
  const char *path = getenv("TPUKERNEL_FIRMWARE_PATH");
  if (!path) return;
  #include <dlfcn.h>
  void *handle = dlopen(path, RTLD_NOW);
  if (handle) {
    TPUKERNEL_LOG("load custom cmodel firmware %s\n", path);
  } else {
    TPUKERNEL_LOG("dlerror: %s\n", dlerror());
  }
}
#endif

bool tpu_is_data_type_int4(data_type_t dtype) {
  return dtype == DT_INT4 || dtype == DT_UINT4;
}

bool tpu_is_data_type_int8(data_type_t dtype) {
  return dtype == DT_INT8 || dtype == DT_UINT8;
}

bool tpu_is_data_type_int16(data_type_t dtype) {
  return dtype == DT_INT16 || dtype == DT_UINT16;
}

bool tpu_is_data_type_int32(data_type_t dtype) {
  return dtype == DT_INT32 || dtype == DT_UINT32;
}

bool tpu_is_data_type_int(data_type_t dtype) {
    return tpu_is_data_type_int4(dtype) ||
           tpu_is_data_type_int8(dtype) ||
           tpu_is_data_type_int16(dtype) ||
           tpu_is_data_type_int32(dtype);
}

bool tpu_is_data_type_signed_int(data_type_t dtype) {
    return dtype == DT_INT32 || dtype == DT_INT16 || dtype == DT_INT8 || dtype == DT_INT4;
}
bool tpu_is_data_type_unsigned_int(data_type_t dtype) {
    return dtype == DT_UINT32 || dtype == DT_UINT16 || dtype == DT_UINT8 || dtype == DT_UINT4;
}
bool tpu_is_data_type_fp(data_type_t dtype) {
    return dtype == DT_FP32 || dtype == DT_TF32 || dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2;
}
bool tpu_is_data_type_fp8(data_type_t dtype) {
    return dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2;
}
int tpu_data_type_size(data_type_t dtype) {
    if (dtype == DT_FP32 || dtype == DT_TF32 || dtype == DT_INT32 || dtype == DT_UINT32 || dtype == DT_FP20)
        return 4;
    else if (dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_INT16 ||
             dtype == DT_UINT16)
        return 2;
    else if (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2)
        return 1;
    else if (dtype == DT_INT4 || dtype == DT_UINT4)
        TPUKERNEL_ASSERT_INFO(0, "should not use %s for INT4/UINT4, try to use 'tpu_data_type_bits'", __func__);
    else
        TPUKERNEL_ASSERT_INFO(0, "not supported dtype=%d", dtype);
    return 0;
}

int tpu_data_type_bits(data_type_t dtype) {
    if (dtype == DT_FP32 || dtype == DT_TF32 || dtype == DT_INT32 || dtype == DT_UINT32)
        return 32;
    else if (dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_INT16 ||
             dtype == DT_UINT16)
        return 16;
    else if (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2)
        return 8;
    else if (dtype == DT_INT4 || dtype == DT_UINT4)
        return 4;
    else
        TPUKERNEL_ASSERT_INFO(0, "not supported dtype=%d", dtype);
    return 0;
}

void tpu_local_shape_5d_to_4d(const dim5* shape_5d, dim4* shape_4d) {
    shape_4d->n = shape_5d->n * shape_5d->d;
    shape_4d->c = shape_5d->c;
    shape_4d->h = shape_5d->h;
    shape_4d->w = shape_5d->w;
}

void tpu_print_value(const void* data, data_type_t dtype){
  if(dtype == DT_INT32 || dtype == DT_UINT32){
    TPUKERNEL_LOG("0x%08x ", *(int*)data);
  } else if(dtype == DT_FP32) {
    TPUKERNEL_LOG("%f ", *(float*)data);
  } else if(dtype == DT_INT16 || dtype == DT_UINT16){
    TPUKERNEL_LOG("0x%04x ", *(short*)data);
  } else if(dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_UINT4 || dtype == DT_INT4){
    TPUKERNEL_LOG("0x%02x ", (*(char*)data)&0xFF);
  } else if(dtype == DT_FP8E5M2){
    scalar_t v;
    v.f8e5m2 = *(float8e5m2*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    TPUKERNEL_LOG("%f ", v.f32);
  } else if(dtype == DT_FP8E4M3){
    scalar_t v;
    v.f8e3m4 = *(float8e4m3*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    TPUKERNEL_LOG("%f ", v.f32);
  } else if(dtype == DT_FP16){
    scalar_t v;
    v.f16 =*(float16*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    TPUKERNEL_LOG("%f ", v.f32);
  } else if(dtype == DT_BFP16){
    scalar_t v;
    v.bf16 =*(bfloat16*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    TPUKERNEL_LOG("%f ", v.f32);
  }
}
void tpu_dump_value(FILE* fp, const void* data, data_type_t dtype){
  if(dtype == DT_INT32 || dtype == DT_UINT32) {
    fprintf(fp, "%d", *(int*)data);
  } else if (dtype == DT_FP32){
    fprintf(fp, "%f", *(float*)data);
  } else if(dtype == DT_INT16 || dtype == DT_UINT16){
    fprintf(fp, "%d", *(short*)data);
  } else if(dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_UINT4 || dtype == DT_INT4){
    fprintf(fp, "%d", (*(char*)data)&0xFF);
  } else if(dtype == DT_FP8E5M2){
    scalar_t v;
    v.f8e5m2 = *(float8e5m2*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    fprintf(fp, "%f", v.f32);
  } else if(dtype == DT_FP8E4M3){
    scalar_t v;
    v.f8e3m4 = *(float8e4m3*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    fprintf(fp, "%f", v.f32);
  } else if(dtype == DT_FP16){
    scalar_t v;
    v.f16 =*(float16*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    fprintf(fp, "%f", v.f32);
  } else if(dtype == DT_BFP16){
    scalar_t v;
    v.bf16 =*(bfloat16*)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    fprintf(fp, "%f", v.f32);
  }
}
void tpu_print_local_mem_data(local_addr_t local_offset, int start_idx, const dim4* shape, const dim4* stride, data_type_t dtype){
  if(!stride){
    dim4 true_stride;
    tpu_aligned_stride(&true_stride, start_idx, shape, dtype);
    tpu_print_local_mem_data(local_offset, start_idx, shape, &true_stride, dtype);
    return;
  }
  const int npu_num = NPU_NUM;
  unsigned char* ptrs[npu_num];
  for(int i=0; i<npu_num; i++){
    ptrs[i] = tpu_local_mem_addr(i, local_offset);
  }
  TPUKERNEL_LOG("LOCAL offset=0x%x, idx=%d, shape=(%d %d %d %d), stride=(%d,%d,%d,%d)\n",
   local_offset, start_idx,
   shape->n, shape->c, shape->h, shape->w,
   stride->n, stride->c, stride->h, stride->w);
  dim4 offset;
  int type_len = tpu_data_type_size(dtype);
  unsigned char* ptr = NULL;
  for(int n=0; n<shape->n; n++){
    offset.n = n * stride->n;
    for(int c=0; c<shape->c; c++) {
      ptr = ptrs[(c+start_idx)%npu_num];
      offset.c = ((c+start_idx)/npu_num) * stride->c;
      for(int h=0; h<shape->h; h++) {
        offset.h = h * stride->h;
        TPUKERNEL_LOG("(%d,%d,%d): ", n, c, h);
        for(int w=0; w<shape->w; w++){
          offset.w = w * stride->w;
          tpu_print_value(ptr+(offset.n+offset.c+offset.h+offset.w)*type_len, dtype);
        }
        TPUKERNEL_LOG("\n");
      }
    }
  }
}

void tpu_print_global_mem_data(global_addr_t addr, const dim4* shape, const dim4* stride, data_type_t dtype){
  if(!stride) {
    dim4 true_stride;
    tpu_continuous_stride(&true_stride, shape);
    tpu_print_global_mem_data(addr, shape, &true_stride, dtype);
    return;
  }
  TPUKERNEL_LOG("GLOBAL addr=0x%llx, shape=(%d %d %d %d), stride=(%d,%d,%d,%d)\n",
         addr, shape->n, shape->c, shape->h, shape->w,
         stride->n, stride->c, stride->h, stride->w);
  unsigned char* ptr = tpu_global_mem_addr(addr);
  dim4 offset;
  int type_len = tpu_data_type_size(dtype);
  for(int n=0; n<shape->n; n++){
    offset.n = n * stride->n;
    for(int c=0; c<shape->c; c++) {
      offset.c = c * stride->c;
      for(int h=0; h<shape->h; h++) {
        offset.h = h * stride->h;
        TPUKERNEL_LOG("(%d,%d,%d): ", n, c, h);
        for(int w=0; w<shape->w; w++){
          offset.w = w * stride->w;
          tpu_print_value(ptr+(offset.n+offset.c+offset.h+offset.w)*type_len, dtype);
        }
        TPUKERNEL_LOG("\n");
      }
    }
  }
}

void tpu_dump_global_layer_data(const char* filename, global_addr_t addr, const int* shape, int dims, data_type_t dtype){
  FILE* fp = fopen(filename, "a");
  fprintf(fp, "{\"shape\":[");
  for (int k = 0; k < dims; ++k) {
    fprintf(fp, "%d,", shape[k]);
  }
  fprintf(fp, "],");
  fprintf(fp, "\"is_fp\":%d,", tpu_is_data_type_fp(dtype));
  fprintf(fp, "\"data\":[");
  int elem_num = 1;
  for (int k = 0; k < dims; ++k) {
    elem_num *= shape[k];
  }
  int type_len = tpu_data_type_size(dtype);
  unsigned char* ptr = tpu_global_mem_addr(addr);
  for (int i = 0; i < elem_num; ++i) {
    tpu_dump_value(fp, ptr + i * type_len, dtype);
    fprintf(fp, ",");
  }
  fprintf(fp, "]}");
  fclose(fp);
}

void *tpu_kernel_memcpy(void *dst, const void *src, size_t n)
{
    volatile char *tmp = (char*)dst;
    volatile char *s = (char*)src;
    while (n--) {
        *tmp++ = *s++;
    }
    return dst;
}

void *tpu_kernel_memset(void *dst, int c, size_t n)
{
    volatile char *tmp = (char*)dst;
    char val = (char)c;
    while (n--) {
        *tmp++ = val;
    }
    return dst;
}

#define MAX_FUNC_NAME_LENGTH (64)
#define MAX_NUM_KERNEL_FUNCS (2048)
typedef struct {
    char name[MAX_FUNC_NAME_LENGTH + 1];
    tpu_kernel_func_t func;
} func_pair_t;

static inline func_pair_t *get_func_pairs() {
    static func_pair_t pairs[MAX_NUM_KERNEL_FUNCS];
    return pairs;
}
static inline int *get_func_num() {
    static int num = 0;
    return &num;
}

void tpu_register_kernel_func(const char *name, tpu_kernel_func_t func) {
    TPUKERNEL_ASSERT(strlen(name) <= MAX_FUNC_NAME_LENGTH);
    const int num = *get_func_num();
    func_pair_t *pairs = get_func_pairs();
    TPUKERNEL_ASSERT(num < MAX_NUM_KERNEL_FUNCS);
    strcpy(pairs[num].name, name);
    pairs[num].func = func;
    ++*get_func_num();
#if defined(USING_CMODEL) && defined(DEBUG)
    // TPUKERNEL_LOG("TPUKernel register function %s\n", name);
#endif
}
RTM_EXPORT(tpu_register_kernel_func);

tpu_kernel_func_t tpu_get_kernel_func_by_name(const char *name) {
    func_pair_t *pairs = get_func_pairs();
    const int num = *get_func_num();
    for (int i = 0; i < num; ++i) {
        if (strcmp(name, pairs[i].name) == 0)
            return pairs[i].func;
    }
    TPUKERNEL_LOG("TPUKernel function %s is not found\n", name);
    return NULL;
}

void tpu_dump_registered_kernel_funcs() {
    const int num = *get_func_num();
    func_pair_t *pairs = get_func_pairs();
    for (int i = 0; i < num; ++i)
        TPUKERNEL_LOG("TPUKernel function %d: %s\n", i, pairs[i].name);
    UNUSED(pairs);
}

#define NOT_IMPLEMENTED() \
  TPUKERNEL_ASSERT_INFO(0, "Not implemented %s", __func__)

#define WEAK __attribute__((weak))


WEAK void tpu_debug_cmd(u64 *p_cmd, int length, int command_type){
  NOT_IMPLEMENTED();
}

WEAK int tpu_core_num() {
  return 1;
}
WEAK int tpu_core_index() {
  return 0;
}

WEAK void tpu_vsdma_poll() {
  NOT_IMPLEMENTED();
}

WEAK int tpu_start_physical_core_id() {
  return 0;
}

WEAK int tpu_tpuv7_env() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK void tpu_sync_start() {
  NOT_IMPLEMENTED();
}

WEAK void tpu_sync_end() {
  NOT_IMPLEMENTED();
}

WEAK bool tpu_is_sync_state() {
  NOT_IMPLEMENTED();
  return 0;
}

// query engine status
WEAK int tpu_bdc_busy() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_gdma_busy() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_sdma_busy() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_hau_sort_busy() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_gdma_cmd_overflow() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_sdma_cmd_overflow() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int tpu_hau_sort_cmd_overflow() {
  NOT_IMPLEMENTED();
  return 0;
}

// high level sync func
WEAK void tpu_sync_core() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_all() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_all_bdc() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_all_gdma() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_all_sdma() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_all_hau() {
  NOT_IMPLEMENTED();
}

WEAK void tpu_sync_finish() {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sync_core_innner() {
  NOT_IMPLEMENTED();
}

WEAK void tpu_sync_core_with_external_engine() {
  NOT_IMPLEMENTED();
}

// low level sync interfaces
WEAK int tpu_next_msg_id() {
  NOT_IMPLEMENTED();
  return -1;
}
WEAK int tpu_get_local_msg_id() {
  NOT_IMPLEMENTED();
  return -1;
}
WEAK int tpu_get_global_msg_id() {
  NOT_IMPLEMENTED();
  return -1;
}
WEAK int tpu_get_ccl_msg_id() {
  NOT_IMPLEMENTED();
  return -1;
}
WEAK void tpu_set_base_msg_id(int base_msg_id) {
  NOT_IMPLEMENTED();
}
WEAK void tpu_core_context_setup(int core_idx, int core_num, int core_msg_id) {
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_send_msg(int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_bdc_wait_msg(int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_gdma_send_msg(int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_gdma_wait_msg(int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_hau_send_msg(int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_hau_wait_msg(int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_sdma_send_msg(int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_sdma_wait_msg(int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_vsdma_send_msg(int msg_id, int wait_cnt, int port_id){
  NOT_IMPLEMENTED();
}
WEAK void tpu_vsdma_wait_msg(int msg_id, int send_cnt, int port_id){
  NOT_IMPLEMENTED();
}
WEAK int tpu_cdma_get_port(int self, int peer, int direction) {
  NOT_IMPLEMENTED();
  return -1;
}
WEAK void tpu_cdma_send_msg(int port, int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_nop(int port){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_wait_msg(int port, int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_tx_send_msg(int port, int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_tx_wait_msg(int port, int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_rx_send_msg(int port, int msg_id, int wait_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_rx_wait_msg(int port, int msg_id, int send_cnt){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_perf_port_poll(int *chips, int *ports, int* actions, u64 *info_addr) {
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_perf_poll(int *chips, int *ports, int* actions, u64 *info_addr) {
  NOT_IMPLEMENTED();
}
WEAK void tpu_sr_setup(){
  NOT_IMPLEMENTED();
}
WEAK void tpu_enable_pmu(){
  NOT_IMPLEMENTED();
}
WEAK void tpu_disable_pmu(){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_nop_sync(int port){
  NOT_IMPLEMENTED();
}
WEAK void tpu_cdma_tx_rx_debug(int port){
  NOT_IMPLEMENTED();
}
WEAK int tpu_chip_id() {
    NOT_IMPLEMENTED();
    return 0;
}
WEAK int tpu_chip_num() {
    NOT_IMPLEMENTED();
    return 1;
}

WEAK int tpu_rank() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int* tpu_chip_map() {
  NOT_IMPLEMENTED();
  return NULL;
}

WEAK int tpu_use_ring() {
  NOT_IMPLEMENTED();
  return 1;
}

WEAK void tpu_sccl_init(int chip_num, int rank, int* chip_map, int use_ring) {
  NOT_IMPLEMENTED();
}

WEAK int tpu_cdma_get_used_port_num() {
  NOT_IMPLEMENTED();
  return 0;
}

WEAK int *tpu_cdma_get_used_ports() {
  NOT_IMPLEMENTED();
  return NULL;
}

WEAK void tpu_bdc_end() {
    NOT_IMPLEMENTED();
}

WEAK void tpu_cdma_all_port_nop() {
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_random_gen_init(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    int             need_store_state,
    int             jump_cnt,
    int             c_offset,
    const dim4     *shape,
    data_type_t     dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_random_gen(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    int             need_store_state,
    const dim4     *shape,
    data_type_t     dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_random_gen_load_state(
    local_addr_t    res_addr,
    local_addr_t    store_state_addr,
    local_addr_t    load_state_addr,
    int             need_store_state,
    const dim4     *shape,
    data_type_t     dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_compress_L2S(
    system_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride){
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_compress_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_decompress_S2L(
    local_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_decompress_S2S(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_compress(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_decompress(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_random_mask_init_seed_S2L(
    system_addr_t mask_addr,
    local_addr_t dst_addr,
    const dim4 *shape,
    int size,
    const dim4 *dst_stride,
    data_type_t dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_random_mask_S2L(
    system_addr_t mask_addr,
    local_addr_t dst_addr,
    const dim4 *shape,
    int size,
    const dim4 *dst_stride,
    int use_iter_state,
    data_type_t dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_random_mask_set_seed(const int seed) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bd_set_random_gen_seed(u64 seed) {
    NOT_IMPLEMENTED();
}
WEAK void tpu_bd_rand_seed_gen() {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_compress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_compress_reduce_L12L2(
    system_addr_t dst_addr,
    local_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_compress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_decompress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_lossy_decompress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_cpy_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_cpy_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_reduce_L12L2(
    system_addr_t dst_addr,
    local_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_compress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_compress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_decompress_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_lossy_decompress_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    int reduce_psum_op,
    int reduce_opcode) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_nctrans_reduce_L12L2(
    system_addr_t dst_addr,
    local_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode){
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_nctrans_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode){
    NOT_IMPLEMENTED();
}

WEAK void tpu_gdma_cpy_nctrans_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode){
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_cpy_nctrans_reduce_S2L2(
    system_addr_t dst_addr,
    system_addr_t  src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode){
    NOT_IMPLEMENTED();
}

WEAK void tpu_sdma_cpy_nctrans_reduce_L22L2(
    system_addr_t dst_addr,
    system_addr_t src_addr,
    const dim4 *shape,
    const dim4 *dst_stride,
    const dim4 *src_stride,
    data_type_t dtype,
    int reduce_psum_op,
    int reduce_opcode){
    NOT_IMPLEMENTED();
}

WEAK void tpu_core_set_state(const unsigned char* state_data, int state_size){ }
WEAK void tpu_core_get_state(unsigned char* state_data, int state_size){ }

WEAK void tpu_bdc_fp8_avg_pool2d(
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
    float re_scale) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_hw_reduce_mean(
      local_addr_t output_addr,
      local_addr_t input_addr,
      local_addr_t buffer_addr,
      const dim4   *shape,
      data_type_t  dtype){
      TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype)); //assert float
      TPUKERNEL_ASSERT(!(output_addr == input_addr && input_addr == buffer_addr));//the three addrs are not same
      int eu_num = tpu_eu_num(dtype);
      scalar_t pool_const = {.f32 = 1.0};
      const int type_size = tpu_data_type_size(dtype);

      dim4 shape1 = {shape->n, shape->c, DIV_UP(shape->h * shape->w, eu_num), eu_num};
      dim2 kernel = {shape1.h, 1};
      const dim4 shape2 = {shape1.n, shape1.c, 1, 1};

      int len = shape->h*shape->w;
      int r = len%eu_num;
      const padding_t padding = {0, 0, 0, 0};
      const dim2 stride = {1, 1};
      const dim2 dilation = {1, 1};
      //padding 0s
      if(r != 0){
        const scalar_t const_val = {0};
        const dim4 pad_shape = {shape1.n, shape1.c, 1, eu_num - r};
        dim4 pad_stride;
        tpu_aligned_stride(&pad_stride, 0, &shape1, dtype);
        pad_stride.h = pad_stride.w = 1;
        tpu_bdc_set_C(
            input_addr + len * type_size,
            const_val,
            &pad_shape,
            &pad_stride,
            dtype);
        };
        local_addr_t mean_h_addr = input_addr == buffer_addr?output_addr:buffer_addr;
        local_addr_t mean_hw_addr = input_addr == buffer_addr?buffer_addr:output_addr;
   //set h axis pool
   tpu_bdc_fp_avg_pool2d(
      mean_h_addr,    // aligned
      input_addr,    // aligned
      &shape1,
      &kernel,
      &padding,
      &stride,
      &dilation,
      dtype,
      tpu_cast(pool_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));

    pool_const.f32 = 1.0/len; // normalization ,len=shape.h*shape.w
    shape1.h = 1;
    shape1.w = eu_num;
    kernel.h = shape1.h;
    kernel.w = shape1.w;
    tpu_bdc_fp_avg_pool2d(
      mean_hw_addr,    // aligned
      mean_h_addr,    // aligned
      &shape1,
      &kernel,
      &padding,
      &stride,
      &dilation,
      dtype,
      tpu_cast(pool_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
      if(!(mean_hw_addr == output_addr)){
      tpu_bdc_cpy(
            output_addr,
            mean_hw_addr,
            &shape2,
            NULL,
            NULL,
            dtype);}
}

WEAK void tpu_set_gdma_id(int gdma_id) {
    NOT_IMPLEMENTED();
}
WEAK void tpu_set_bd_id(int bdc_id) {
    NOT_IMPLEMENTED();
}

WEAK int tpu_physical_core_id() {
    return 0;
}
WEAK int tpu_physical_core_num() {
    return 1;
}

WEAK void tpu_bdc_srch_bin_select(
    local_addr_t       dst_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const dim4        *shape,
    int                side,
    int                bin_w,
    data_type_t        src0_src1_dtype,
    data_type_t        dst_dtype);

WEAK void tpu_bdc_fp_conv_bw(
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
    bool              result_add) {
      NOT_IMPLEMENTED();
    }

#define TPU_BDC_FP8_BINARY_C(name, op)                                         \
WEAK void tpu_bdc_fp8_##name##_C(                                                   \
    local_addr_t     dst_addr,                                                 \
    local_addr_t     src_addr,                                                 \
    scalar_t         C,                                                        \
    const dim4      *shape,                                                    \
    const dim4      *dst_stride,                                               \
    const dim4      *src_stride,                                               \
    data_type_t      dst_dtype,                                                \
    data_type_t      src_dtype,                                                \
    data_type_t      C_dtype,                                                  \
    int             satu_mode) {                                               \
    NOT_IMPLEMENTED();                                                     \
}
TPU_BDC_FP8_BINARY_C(add, AR_ADD)
TPU_BDC_FP8_BINARY_C(mul, AR_MUL)
WEAK void tpu_bdc_fp8_C_sub(
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
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp8_mm_all_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data){
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_silu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype){
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    data_type_t   dtype){
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_bf16_sigmoid_fast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_bf16_gelu_fast2(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_hau_sort_specific_index_2d(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_data_addr,
    system_addr_t  input_idx_addr,
    int            row_num,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_hau_sort_natural_index_2d(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_addr,
    int            row_num,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype) {
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_conv2d_with_rescale(
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
    bool              scale_const)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_arithmetic_sequence_bcast_with_dtype(
    local_addr_t  dst_addr,
    int           npu_num,
    int           start,
    int           step,
    int           num,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_arithmetic_sequence_distribute_with_dtype(
    local_addr_t  dst_addr,
    int           start,
    int           step,
    int           num,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_arithmetic_sequence_distribute_aligned_with_dtype(
    local_addr_t  dst_addr,
    int           start,
    int           step,
    int           num,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_arithmetic_sequence_general_with_dtype(
    local_addr_t  dst_addr,
    local_addr_t  buffer_addr, // size = sizeof(dtype)
    int           npu_num,
    int           start,
    int           step,
    int           num,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp8_mm_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp8_mm_R_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_run_commands(const tpu_engine_command_info_t* commands, u32 engine_num) {
    NOT_IMPLEMENTED();
}
WEAK u32 tpu_engine_num() {
  NOT_IMPLEMENTED();
  return 0;
}
WEAK void tpu_bdc_fp_conv2d_rescale_C(
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
    bool              do_relu)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_depthwise2d_with_scale(
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
    bool              scale_is_const)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_sinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_cosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_tanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_elu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    float         alpha,
    const dim4   *shape,
    data_type_t   dtype)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_conv2d_backward(
    local_addr_t     grad_wight_local_addr,
    local_addr_t     forward_input_local_addr,
    local_addr_t     grad_output_local_addr,
    local_addr_t     pad_ins_local_addr,
    const dim4      *forwrad_input_shape,
    const dim4      *forward_output_shape,
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
    data_type_t      grad_dtype)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_conv2d_with_ins(
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
    bool              result_add,
    int               ins_h,
    int               ins_w)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_arctanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    TPUKERNEL_ASSERT(dtype == DT_BFP16 || dtype == DT_FP32 || dtype == DT_FP16);
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        dst_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // WORK = 1 - SRC
    tpu_bdc_fp_C_sub(
        work_addr,
        src_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    // DST = DST / WORK
    tpu_bdc_fp_div(
        dst_addr,
        dst_addr,
        work_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
    //  DST = LN(DST)
    tpu_bdc_fp_log(
        dst_addr,
        dst_addr,
        work_addr,
        coeff_addr,
        shape,
        dtype);
    //  DST = DST * 0.5
    scalar_t C1 = {.f32 = 0.5f};
    tpu_bdc_fp_mul_C(
        dst_addr,
        dst_addr,
        tpu_cast(C1, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
}

WEAK void tpu_bdc_load_fp_arcsin_coeff(local_addr_t coeff_addr, data_type_t dtype) {
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_load_div_lut(local_addr_t table_addr) {
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_arcsin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_arccos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_exp_for_neg_arg(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_fp_log_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  ln_coeff_addr,
    const dim4   *shape,
    data_type_t   dtype)
{
    //  DST = EXP(SRC)
    tpu_bdc_fp_exp(
      dst_addr,
      src_addr,
      work0_addr,
      work1_addr,
      coeff_addr,
      shape,
      dtype);
    //  WORK0 = DST + 1
    scalar_t C = {.f32 = 1.f};
    tpu_bdc_fp_add_C(
        work1_addr,
        dst_addr,
        tpu_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN),
        shape,
        NULL,
        NULL,
        dtype);
    //  WORK0 = LOG(WORK1)
    tpu_bdc_fp_log(
        work0_addr,
        work1_addr,
        dst_addr,
        ln_coeff_addr,
        shape,
        dtype);
    //  DST = SRC - WORK0
    tpu_bdc_fp_sub(
        dst_addr,
        src_addr,
        work0_addr,
        shape,
        NULL,
        NULL,
        NULL,
        dtype);
}

WEAK void tpu_bdc_fp8_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    bool          result_add,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_bdc_f16_group_dequant(
  local_addr_t     dst_addr,
  local_addr_t     src_addr,
  local_addr_t     quant_addr,
  const dim4      *shape,
  data_type_t      src_dtype,
  data_type_t      dst_dtype,
  int              group)
{
  NOT_IMPLEMENTED();
}

WEAK void tpu_reset_base_addr() {}
WEAK void tpu_set_base_addr(const int *base_idx, const u64 *base_addr, int num) {}

WEAK int tpu_workitem_num()
{
    return tpu_core_num();
}

WEAK int tpu_workitem_index()
{
    return tpu_core_index();
}

WEAK int tpu_is_last_workitem() {
    NOT_IMPLEMENTED();
    return 0;
}

WEAK system_addr_t tpu_global_mem_real_addr(system_addr_t addr) {
  return addr;
}