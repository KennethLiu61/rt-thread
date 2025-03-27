#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

#define N 4
#define C 64
#define H 32
#define W 32
#define loop 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

void nodechip_gdma_lossy_decompress_test(
  global_addr_t input_addr,
  global_addr_t output_addr
) {
// TODO ：check the bandwith calculation for fp20
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < loop; ++t) {
#define END(info, count, type, reduce_op) \
    }                   \
    poll_all_engine_done(&id_node);     \
    end_time = firmware_timer_get_time_us();    \
    printf("test %d times %s gdma lossy decompress %s\n", loop, #info, reduce_op);            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
            (float)loop * count * get_gdma_format_type_len(type) * 1e6 / (end_time - start_time) / (powf(1024.f, 3)));

    char * all_reduce_opcode[5] = {"ALL_REDUCE_NOP",
                                   "ALL_REDUCE_MUL",
                                   "ALL_REDUCE_MAX",
                                   "ALL_REDUCE_MIN",
                                   "ALL_REDUCE_ADD"};
    int src_H_stride, src_C_stride, src_N_stride;
    int dst_H_stride, dst_C_stride, dst_N_stride;
    int count = N * C * H * W;
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test S2S lossy decompress
    printf("========== GDMA_LOSSY_DECOMPRESS_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    src_H_stride = W;
    src_C_stride = src_H_stride * H;
    src_N_stride = src_C_stride * C;
    dst_H_stride = W;
    dst_C_stride = dst_H_stride * H;
    dst_N_stride = dst_C_stride * C;
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    BEGIN()
    gdma_lossy_decompress_gen_cmd(
      input_addr, // sys
      N,
      C,
      H,
      W,
      src_N_stride,
      src_C_stride,
      src_H_stride,
      output_addr, // local_addr or sys
      0, //dst_local_idx
      dst_N_stride,
      dst_C_stride,
      dst_H_stride,
      GDMA_S2S,
      0, //thread_id
      &id_node);
    END(S2S, count, GDMA_FP32, "")

    // test S2S lossy decompress are
    printf("========== GDMA_LOSSY_DECOMPRESS_REDUCE_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    start_time = firmware_timer_get_time_us();
    for (int i = 0; i < 5; i++) {
        BEGIN()
        gdma_lossy_decompress_reduce_gen_cmd(
            input_addr, // sys
            N,
            C,
            H,
            W,
            src_N_stride,
            src_C_stride,
            src_H_stride,
            tpu_l2_sram_get_start_addr(), // local_addr or sys
            0, //dst_local_idx
            dst_N_stride,
            dst_C_stride,
            dst_H_stride,
            GDMA_S2S, // only l2
            GDMA_ARE_PSUM_RW,
            i,
            0, //thread_id,
            &id_node);
        END(S2S, count, GDMA_FP32, all_reduce_opcode[i])
    }
    

    // test S2L lossy decompress
    printf("========== GDMA_LOSSY_DECOMPRESS_S2L ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    int dst_local_idx = NPU_NUM >>1 ;
    //TODO random output_lo
    local_addr_t output_lo = 0;
    int dst_size = get_tensor_size_onelocal_mem(N, C, H, W, 0, false, FP32);
    dst_H_stride = W;
    dst_C_stride = ALIGN(W * H, get_eu_num(FP32));
    dst_N_stride = dst_C_stride * ceiling_func(C + dst_local_idx, NPU_NUM);
    TPUKERNEL_ASSERT(output_lo + dst_size <= LOCAL_MEM_SIZE);
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    BEGIN()
    gdma_lossy_decompress_gen_cmd(
        input_addr, // sys
        N,
        C,
        H,
        W,
        src_N_stride,
        src_C_stride,
        src_H_stride,
        output_addr, // local_addr or sys
        dst_local_idx,
        dst_N_stride,
        dst_C_stride,
        dst_H_stride,
        GDMA_S2L,
        0, //thread_id
        &id_node);
    END(S2L, count, GDMA_FP32, "")
}

