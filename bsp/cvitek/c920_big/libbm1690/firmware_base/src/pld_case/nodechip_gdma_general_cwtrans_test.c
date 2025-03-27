#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

#define loop 10

void nodechip_gdma_general_cwtrans_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{

    int GDMA_FORMAT[3] = {GDMA_INT8, GDMA_INT16, GDMA_INT32};
    char * gdma_format[] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32"};
    u64 start_time = 0ull;
    u64 end_time = 0ull;

    int shape[][4] = {{1, 1, 1, 512}, {1, 1, 1, 1024}, {1, 1, 1, 2048},
                      {1, 32, 1, 256}, {1, 256, 1, 32}, {1, 256, 1, 128},
                      {1, 128, 1, 256}};

    local_addr_t input_lo = 0;
    local_addr_t output_lo = LOCAL_MEM_SIZE / 2;

    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test gdma general cw_trans s2s
    printf("========== GDMA_GENERAL_CW_TRANS_S2S ==========\n");
    printf("loop: %d\n", loop);
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n", input_addr, output_addr);
    for (int i = 0; i < 7; i++) {
        int N = shape[i][0];
        int C = shape[i][1];
        int H = shape[i][2];
        int W = shape[i][3];
        printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
        for (int j = 0; j < 3; j++) {
            u64 tensor_size = N * C * H * W * get_gdma_format_type_len(GDMA_FORMAT[j]);
            start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; iter++) {
                general_cwtrans_gen_cmd(
                    input_addr,
                    0,
                    output_addr,
                    0,
                    N,
                    C,
                    H,
                    W,
                    GDMA_FORMAT[j],
                    0, 0, 0,
                    0, 1, 0,
                    0,
                    GDMA_S2S,
                    MASTER_THREAD,
                    &id_node);
            }
            poll_all_engine_done(&id_node);
            end_time = firmware_timer_get_time_us();
            printf("GDMA_FORMAT: %s\n", gdma_format[j]);
            printf("Total time: %lldus, BW: %fGB/s\n", (end_time - start_time),
                    (float)loop * tensor_size * 1e6 / (end_time - start_time) / powf(1024, 3));
        }
    }

    // test gdma general cw_trans s2l
    printf("========== GDMA_GENERAL_CW_TRANS_S2L ==========\n");
    printf("loop: %d\n", loop);
    printf("src_addr = 0x%08llx, dst_lo = 0x%08x\n", input_addr, output_lo);
    for (int i = 0; i < 7; i++) {
        int N = shape[i][0];
        int C = shape[i][1];
        int H = shape[i][2];
        int W = shape[i][3];
        printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
        for (int j = 0; j < 3; j++) {
            u64 tensor_size = N * C * H * W * get_gdma_format_type_len(GDMA_FORMAT[j]);
            start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; iter++) {
                general_cwtrans_gen_cmd(
                    input_addr,
                    0,
                    output_lo,
                    0,
                    N,
                    C,
                    H,
                    W,
                    GDMA_FORMAT[j],
                    0, 0, 0,
                    0, 1, 0,
                    0,
                    GDMA_S2L,
                    MASTER_THREAD,
                    &id_node);
            }
            poll_all_engine_done(&id_node);
            end_time = firmware_timer_get_time_us();
            printf("GDMA_FORMAT: %s\n", gdma_format[j]);
            printf("Total time: %lldus, BW: %fGB/s\n", (end_time - start_time),
                    (float)loop * tensor_size * 1e6 / (end_time - start_time) / powf(1024, 3));
        }
    }

    // test gdma general cw_trans l2s
    printf("========== GDMA_GENERAL_CW_TRANS_L2S ==========\n");
    printf("loop: %d\n", loop);
    printf("src_lo = 0x%08x, dst_addr = 0x%08llx\n", input_lo, output_addr);
    for (int i = 0; i < 7; i++) {
        int N = shape[i][0];
        int C = shape[i][1];
        int H = shape[i][2];
        int W = shape[i][3];
        printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
        for (int j = 0; j < 3; j++) {
            u64 tensor_size = N * C * H * W * get_gdma_format_type_len(GDMA_FORMAT[j]);
            start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; iter++) {
                general_cwtrans_gen_cmd(
                    input_lo,
                    0,
                    output_addr,
                    0,
                    N,
                    C,
                    H,
                    W,
                    GDMA_FORMAT[j],
                    0, 0, 0,
                    0, 1, 0,
                    0,
                    GDMA_L2S,
                    MASTER_THREAD,
                    &id_node);
            }
            poll_all_engine_done(&id_node);
            end_time = firmware_timer_get_time_us();
            printf("GDMA_FORMAT: %s\n", gdma_format[j]);
            printf("Total time: %lldus, BW: %fGB/s\n", (end_time - start_time),
                    (float)loop * tensor_size * 1e6 / (end_time - start_time) / powf(1024, 3));
        }
    }

    // test gdma general cw_trans l2l
    printf("========== GDMA_GENERAL_CW_TRANS_L2L ==========\n");
    printf("loop: %d\n", loop);
    printf("src_lo = 0x%08x, dst_lo = 0x%08x\n", input_lo, output_lo);
    for (int i = 0; i < 7; i++) {
        int N = shape[i][0];
        int C = shape[i][1];
        int H = shape[i][2];
        int W = shape[i][3];
        printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
        for (int j = 0; j < 3; j++) {
            u64 tensor_size = N * C * H * W * get_gdma_format_type_len(GDMA_FORMAT[j]);
            start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; iter++) {
                general_cwtrans_gen_cmd(
                    input_lo,
                    0,
                    output_lo,
                    0,
                    N,
                    C,
                    H,
                    W,
                    GDMA_FORMAT[j],
                    0, 0, 0,
                    0, 1, 0,
                    0,
                    GDMA_L2L,
                    MASTER_THREAD,
                    &id_node);
            }
            poll_all_engine_done(&id_node);
            end_time = firmware_timer_get_time_us();
            printf("GDMA_FORMAT: %s\n", gdma_format[j]);
            printf("Total time: %lldus, BW: %fGB/s\n", (end_time - start_time),
                    (float)loop * tensor_size * 1e6 / (end_time - start_time) / powf(1024, 3));
        }
    }
}
