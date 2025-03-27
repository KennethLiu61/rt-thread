#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

#define N 4
#define C 32
#define H 32
#define W 64
#define loop 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

void nodechip_gdma_mask_select_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{

    int GDMA_FORMAT[8] = {GDMA_INT8, GDMA_INT16, GDMA_INT32, GDMA_FP8_E4M3, GDMA_FP8_E5M2, GDMA_FP16, GDMA_BF16, GDMA_FP32};
    char * gdma_format[] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32",
                            "GDMA_FP8_E4M3",
                            "GDMA_FP8_E5M2",
                            "GDMA_FP16",
                            "GDMA_BF16",
                            "GDMA_FP32"};

    int src_size, mask_size;

    u64 start_time = 0ull;
    u64 end_time = 0ull;

    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test S2S mask_select
    printf("========== GDMA_MASK_SELECT_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    src_size = N * C * H * W * sizeof(float);
    int count = N * C * H * W;
    unsigned long long mask_addr = input_addr + src_size;
    printf("src_addr = 0x%08llx, mask_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, mask_addr, output_addr);
    for (int i = 0; i < 8; i++) {
        start_time = firmware_timer_get_time_us();
        for (int j = 0; j < loop; j++) {
            tensor_general_move_with_mask_gen_cmd(
                input_addr,
                0,
                mask_addr,
                0,
                0,
                output_addr,
                GDMA_FORMAT[i],
                GDMA_FORMAT[i],
                N,
                C,
                H,
                W,
                GDMA_S2S,
                MASTER_THREAD,
                &id_node);
        }
        poll_all_engine_done(&id_node);
        end_time = firmware_timer_get_time_us();
        printf("GDMA_FORMAT: %s\n", gdma_format[i]);
        printf("Total time: %lldus, BW: %.3fGB/s\n", (end_time - start_time),
                (float)loop * count * get_gdma_format_type_len(GDMA_FORMAT[i]) * 1e6 / (end_time - start_time) / powf(1024, 3));
    }

    // test L2S mask_select
    printf("========== GDMA_MASK_SELECT_L2S ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    local_addr_t input_lo = 0;
    src_size = get_tensor_size_onelocal_mem(N, C, H, W, 0, false, INT32);
    local_addr_t mask_lo = ALIGN(input_lo + src_size, ALIGN_BYTES);
    mask_size = src_size;
    TPUKERNEL_ASSERT(mask_lo + mask_size <= LOCAL_MEM_SIZE);
    printf("src_lo = 0x%08x, mask_lo = 0x%08x, dst_addr = 0x%08llx\n",
            input_lo, mask_lo, output_addr);
    for (int i = 0; i < 8; i++) {
        start_time = firmware_timer_get_time_us();
        for (int j = 0; j < loop; j++) {
            tensor_general_move_with_mask_gen_cmd(
                input_lo,
                0,
                mask_lo,
                0,
                1,
                output_addr,
                GDMA_FORMAT[i],
                GDMA_FORMAT[i],
                N,
                C,
                H,
                W,
                GDMA_L2S,
                MASTER_THREAD,
                &id_node);
        }
        poll_all_engine_done(&id_node);
        end_time = firmware_timer_get_time_us();
        printf("GDMA_FORMAT: %s\n", gdma_format[i]);
        printf("Total time: %lldus, BW: %.3fGB/s\n", (end_time - start_time),
                (float)loop * count * get_gdma_format_type_len(GDMA_FORMAT[i]) * 1e6 / (end_time - start_time) / powf(1024, 3));
    }
}
