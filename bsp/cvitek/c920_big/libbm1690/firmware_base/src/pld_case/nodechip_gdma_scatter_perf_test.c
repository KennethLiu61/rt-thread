#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_scatter_perf_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define LOOP (10)
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < LOOP; ++t) {
#define END(info, prec, tensor_size, inplace_add) \
    }                   \
    poll_all_engine_done(&pid_node);         \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s gdma scatter %s %s \n", #info, prec, inplace_add == 1 ? "with inplace_add" : "");            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
           (float)LOOP * tensor_size * 1e6 / (end_time - start_time) / (powf(1024.f, 3)))

    CMD_ID_NODE pid_node;
    resync_cmd_id(&pid_node);
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int shape[][6] =
    {   {4, 768, 1, 1200, 1, 0},
        {8, 512, 128, 1200, 0, 0},
        {8, 512, 16, 1200, 0, 0},
        {8, 512, 32, 1200, 0, 0},
        {8, 65536, 32, 65600, 0, 0},
    };

    data_type_t data_type[3] = {DT_FP16, DT_INT8, DT_FP32};
    char *type[] = {"FP16", "INT8", "FP32"};
    u64 in_offset = 0;
    u64 idx_offset = 8 * 512 * 128 * 4;
    u64 out_offset = 0;
    for (int i = 0; i < 5; ++i) {
        int C = shape[i][0];
        int src_h = shape[i][1];
        int src_w = shape[i][2];
        int dst_h = shape[i][3];
        int src_c_is1 = shape[i][4];
        int idx_c_is1 = shape[i][5];
        printf("\nC=%d, src_h=%d, src_w=%d, dst_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
               C, src_h, src_w, dst_h, src_c_is1, idx_c_is1);
        for(int k = 0; k < 2; k++){
            for (int j = 0; j < 3; ++j) {
                data_type_t dtype = data_type[j];
                char *perc = type[j];
                u64 tensor_size = C * src_w * src_h * tpu_data_type_size(dtype);
                BEGIN()
                tensor_gdma_scatter_gen_cmd(
                    input_addr + in_offset, 0,
                    input_addr + idx_offset, 0, 0,
                    output_addr + out_offset, 0,
                    C, src_h, src_w, dst_h, j * k * 0x10, //start_pos
                    0, 0,
                    0, 0,
                    0, 0,
                    ((dtype) >> 1),
                    src_c_is1, idx_c_is1,
                    0, GDMA_S2S, k/*inplace_add*/, MASTER_THREAD,
                    &pid_node
                );
                END(S2S, perc, tensor_size, k);
            }
        }
    }
#undef BEGIN
#undef END
}
