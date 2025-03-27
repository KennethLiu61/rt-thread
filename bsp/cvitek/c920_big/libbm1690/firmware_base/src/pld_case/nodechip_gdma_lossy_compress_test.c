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

void nodechip_gdma_lossy_compress_test(
  global_addr_t input_addr,
  global_addr_t output_addr
) {
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
    int src_size;
    u64 start_time = 0ull;
    u64 end_time = 0ull;
    int src_H_stride, src_C_stride, src_N_stride;
    int dst_H_stride = W;
    int dst_C_stride = dst_H_stride * H;
    int dst_N_stride = dst_C_stride * C;
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test L2S lossy compress
    printf("========== GDMA_LOSSY_COMPRESS_L2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    int src_local_idx = NPU_NUM << 2 ;
    local_addr_t input_lo = 0;
    src_size = get_tensor_size_onelocal_mem(N, C, H, W, input_lo, false, FP32);
    int count = N * C * H * W;
    src_H_stride = W;
    src_C_stride = ALIGN(W * H, get_eu_num(FP32));
    src_N_stride = src_C_stride * ceiling_func(C + src_local_idx, NPU_NUM);
    TPUKERNEL_ASSERT(input_lo + src_size <= LOCAL_MEM_SIZE);
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    BEGIN()
    gdma_lossy_compress_gen_cmd(
        input_lo, // local_addr or sys
        src_local_idx,
        N,
        C,
        H,
        W,
        src_N_stride,
        src_C_stride,
        src_H_stride,
        output_addr, // sys
        dst_N_stride,
        dst_C_stride,
        dst_H_stride,
        GDMA_L2S, // Support S2S, L2S
        0, //thread_id,
        &id_node);
    END(L2S, count, GDMA_FP32, "")

    // test S2S lossy compress are
    printf("========== GDMA_LOSSY_COMPRESS_REDUCE_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    src_size = N * C * H * W * sizeof(float);
    src_H_stride = W;
    src_C_stride = src_H_stride * H;
    src_N_stride = src_C_stride * C;
    u64 l2_addr = tpu_l2_sram_get_start_addr();
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    start_time = firmware_timer_get_time_us();
    for (int i =0; i<5; i++){
        BEGIN()
        gdma_lossy_compress_reduce_gen_cmd(
            input_addr, // local_addr, global_addr or l2_addr
            0, // use only from local_mem
            N,
            C,
            H,
            W,
            src_N_stride,
            src_C_stride,
            src_H_stride,
            l2_addr, //l2_addr
            dst_N_stride,
            dst_C_stride,
            dst_H_stride,
            GDMA_S2S,
            GDMA_ARE_PSUM_RW, //reduce_psum_op
            i, //reduce_opcode
            0, //thread_id,
            &id_node);
        END(S2S, count, GDMA_FP32, all_reduce_opcode[i])
    }
    
}

// #endif