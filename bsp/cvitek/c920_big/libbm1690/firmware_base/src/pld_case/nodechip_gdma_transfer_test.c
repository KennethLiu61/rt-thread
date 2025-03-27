#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

//small case to be stored in static mem
#define N 4
#define C 64
#define H 8
#define W 4
#define loop 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

void nodechip_gdma_transfer_test(
  global_addr_t input_addr,
  global_addr_t output_addr
) {
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < loop; ++t) {
#define END(info, count, type, format) \
    }                   \
    poll_all_engine_done(&id_node);     \
    end_time = firmware_timer_get_time_us();    \
    printf("test %d times %s gdma transfer %s \n", loop, #info, format);            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
            (float)loop * count * get_gdma_format_type_len(type) * 1e6 / (end_time - start_time) / (powf(1024.f, 3)));

    int GDMA_FORMAT[8] = {GDMA_INT8, GDMA_INT16, GDMA_INT32, GDMA_FP8_E4M3, GDMA_FP8_E5M2, GDMA_FP16, GDMA_BF16, GDMA_FP32};
    int PREC[8] = {INT8, INT16, INT32, FP8, FP8, FP16, BFP16, FP32};
    const char * gdma_format[] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32",
                            "GDMA_FP8_E4M3",
                            "GDMA_FP8_E5M2",
                            "GDMA_FP16",
                            "GDMA_BF16",
                            "GDMA_FP32"};
    int count = N * C * H * W;
    u64 start_time = 0ull;
    u64 end_time = 0ull;
    int src_W_stride = 1, src_H_stride, src_C_stride, src_N_stride;
    int dst_W_stride = 1, dst_H_stride, dst_C_stride, dst_N_stride;
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test local to static gdma transfer
    printf("========== GDMA_TRANSFER_LOCAL2STATIC ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    int src_local_idx = NPU_NUM >> 2;
    src_H_stride = W;
    dst_H_stride = W;
    dst_C_stride = dst_H_stride * H;
    dst_N_stride = dst_C_stride * C;
    u64 input_lo_in = 0200;
    u64 input_lo = LOCAL_MEM_START_ADDR + input_lo_in;
    u64 output_st_offset = 0x400;
    u64 output_st = STATIC_MEM_START_ADDR + output_st_offset;
    int src_size;

    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    for(int j=0; j<8; j++){
        src_C_stride = ALIGN(src_H_stride * H, get_gdma_format_type_len(GDMA_FORMAT[j]));
        src_N_stride = ceiling_func(src_local_idx + C, NPU_NUM) * src_C_stride;
        src_size = get_tensor_size_onelocal_mem(N, C, H, W, input_lo, true, PREC[j]);
        TPUKERNEL_ASSERT(input_lo_in + src_size <= LOCAL_MEM_SIZE);
        TPUKERNEL_ASSERT(output_st_offset + N * dst_N_stride  * get_gdma_format_type_len(GDMA_FORMAT[j]) <= STATIC_MEM_SIZE);
        BEGIN()
        atomic_transfer_general_gen_cmd(
        input_lo, //local_addr or smem_addr
        0, //src_core_idx
        N,
        C,
        H,
        W,
        src_N_stride,
        src_C_stride,
        src_H_stride,
        src_W_stride,
        GDMA_FORMAT[j],
        output_st, //local_addr or smem_addr
        1, //dst_core_idx,
        N,
        C,
        H,
        W,
        dst_N_stride,
        dst_C_stride,
        dst_H_stride,
        dst_W_stride,
        0, //thread_id,
        &id_node);
        END(LOCAL2STATIC, count, GDMA_FORMAT[j], gdma_format[j])
    }

    // TODO: move src data to static mem
    // test static to static gdma transfer
    printf("========== GDMA_TRANSFER_STATIC2STATIC ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    u64 input_st_offset = 0x00;
    u64 input_st = STATIC_MEM_START_ADDR;
    src_H_stride = W;
    src_C_stride = src_H_stride * H;
    src_N_stride = src_C_stride * C;
    dst_H_stride = W;
    dst_C_stride = dst_H_stride * H;
    dst_N_stride = dst_C_stride * C;
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    for(int j=0; j<8; j++){
        TPUKERNEL_ASSERT((input_st_offset + N * src_N_stride * get_gdma_format_type_len(GDMA_FORMAT[j])) <= STATIC_MEM_SIZE);
        TPUKERNEL_ASSERT((output_st_offset + N * dst_N_stride * get_gdma_format_type_len(GDMA_FORMAT[j])) <= STATIC_MEM_SIZE);
        BEGIN()
        atomic_transfer_general_gen_cmd(
        input_st, //local_addr or smem_addr
        0, //src_core_idx
        N,
        C,
        H,
        W,
        src_N_stride,
        src_C_stride,
        src_H_stride,
        src_W_stride,
        GDMA_FORMAT[j],
        output_st, //local_addr or smem_addr
        1, //dst_core_idx,
        N,
        C,
        H,
        W,
        dst_N_stride,
        dst_C_stride,
        dst_H_stride,
        dst_W_stride,
        0, //thread_id,
        &id_node);
        END(STATIC2STATIC, count, GDMA_FORMAT[j], gdma_format[j])
    }
}
