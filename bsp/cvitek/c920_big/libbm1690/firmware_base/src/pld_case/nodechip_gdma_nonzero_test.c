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

void nodechip_gdma_nonzero_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < loop; ++t) {
#define END(info, src_format, dst_format, count, type) \
    }                   \
    poll_all_engine_done(&id_node);     \
    end_time = firmware_timer_get_time_us();    \
    printf("test %d times %s gdma nonzero src_format %s dst_foramt %s\n", loop, #info, src_format, dst_format);            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
            (float)loop * count * get_gdma_format_type_len(type) * 1e6 / (end_time - start_time) / (powf(1024.f, 3)));

    int GDMA_FORMAT[8] = {GDMA_INT8, GDMA_INT16, GDMA_INT32, GDMA_FP8_E4M3, GDMA_FP8_E5M2, GDMA_FP16, GDMA_BF16, GDMA_FP32};
    int PREC[8] = {INT8, INT16, INT32, FP8, FP8, FP16, BFP16, FP32};
    char * gdma_src_format[8] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32",
                            "GDMA_FP8_E4M3",
                            "GDMA_FP8_E5M2",
                            "GDMA_FP16",
                            "GDMA_BF16",
                            "GDMA_FP32"};
    char * gdma_dst_format[3] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32"};
    u64 start_time = 0ull;
    u64 end_time = 0ull;

    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test S2S nonzero
    printf("========== GDMA_NONZERO_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    int count = N * C * H * W;
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    for (int i = 0; i < 8; i++) {
        BEGIN()
        tensor_move_nonzero_gen_cmd(
            input_addr, // local_addr or global_addr
            0, // use only from local_mem
            output_addr, // global addr only
            GDMA_FORMAT[i], // INT8/INT16/INT32/FP32/FP16/BF16
            GDMA_FORMAT[i%3], //INT8/INT16/INT32
            N,
            C,
            H,
            W,
            0, //base_idx
            GDMA_S2S, // only support L2S, S2S
            0, // thread_id
            &id_node);
        END(S2S, gdma_src_format[i], gdma_dst_format[i%3], count, GDMA_FORMAT[i])
    }

    // test L2S nonzero
    printf("========== GDMA_NONZERO_L2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    local_addr_t input_lo = 0;
    for (int i = 0; i < 8; i++) {
        int src_size = get_tensor_size_onelocal_mem(N, C, H, W, 0, false, PREC[i]);
        TPUKERNEL_ASSERT(input_lo + src_size <= LOCAL_MEM_SIZE);
        BEGIN()
        tensor_move_nonzero_gen_cmd(
            input_lo, // local_addr or global_addr
            0, // use only from local_mem
            output_addr, // global addr only
            GDMA_FORMAT[i], // INT8/INT16/INT32/FP32/FP16/BF16
            GDMA_FORMAT[i%3], //INT8/INT16/INT32
            N,
            C,
            H,
            W,
            0, //base_idx
            GDMA_L2S, // only support L2S, S2S
            0, // thread_id
            &id_node);
        END(L2S, gdma_src_format[i], gdma_dst_format[i%3], count, GDMA_FORMAT[i])
    }
}

