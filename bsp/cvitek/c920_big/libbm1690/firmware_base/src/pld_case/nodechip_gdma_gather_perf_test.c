#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_gather_perf_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define LOOP (10)
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < LOOP; ++t) {
#define END(info, prec, tensor_size) \
    }                   \
    poll_all_engine_done(&pid_node);     \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s gdma gather %s \n", #info, prec);            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
            (float)LOOP * tensor_size * 1e6 / (end_time - start_time) / (powf(1024.f, 3)))

    CMD_ID_NODE pid_node;
    resync_cmd_id(&pid_node);
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int shape[][6] =
    {   {4, 30522, 768, 128, 1, 0},
        {6, 65537, 16, 512, 0, 1},
        {8, 15261, 32, 64, 0, 0},
        {8, 15261, 1, 128, 0, 0},
        {6, 65536, 16, 65600, 0, 1},
    };

    data_type_t data_type[3] = {DT_FP16, DT_INT8, DT_FP32};
    char *type[] = {"FP16", "INT8", "FP32"};
    int const_value[3] = {0x1010, 0x10, 0x101010};
    u64 in_offset = 0;
    u64 idx_offset = 30522 * 768 * sizeof(float);
    u64 out_offset = 0;

    for (int i = 0; i < 5; ++i) {
        int C = shape[i][0];
        int src_h = shape[i][1];
        int src_w = shape[i][2];
        int index_h = shape[i][3];
        int src_c_is1 = shape[i][4];
        int idx_c_is1 = shape[i][5];
        printf("\nC=%d, src_h=%d, src_w=%d, idx_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
               C, src_h, src_w, index_h, src_c_is1, idx_c_is1);
        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            int c_value = const_value[j];
            char *perc = type[j];
            u64 tensor_size = C * src_w * index_h * tpu_data_type_size(dtype);
            BEGIN()
            tensor_gdma_gather_gen_cmd(
                input_addr + in_offset, 0,
                input_addr + idx_offset, 0, 0,
                output_addr + out_offset, 0,
                c_value,
                C, src_h, src_w, index_h, j * 0x10,
                0, 0,
                0, 0,
                0, 0,
                ((dtype) >> 1),
                src_c_is1, idx_c_is1,
                0, GDMA_S2S, MASTER_THREAD,
                &pid_node
            );
            END(S2S, perc, tensor_size);
        }
    }
#undef BEGIN
#undef END
#undef LOOP
}
