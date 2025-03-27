#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_tensor_cwtrans_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < 10; ++t) {
#define END(info, prec, tensor_size) \
    }                   \
    poll_all_engine_done(&pid_node);        \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s gdma tensor cwtrans %s \n", #info, prec);            \
    printf("Total %s time: %lldus, BW: %.3fGB/s\n\n", #info, (end_time - start_time), \
            (float)10 * tensor_size * 1e6 / (end_time - start_time) / powf(1024, 3))

    CMD_ID_NODE pid_node;
    resync_cmd_id(&pid_node);
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int shape[][4] =
    {   {1, 1, 1, 256},
        {1, 1, 1, 512},
        {1, 1, 1, 1024},
        {1, 32, 1, 256},
        {1, 256, 1, 32},
        {1, 256, 1, 128},
        {1, 128, 1, 256},
    };

    data_type_t data_type[] = {DT_FP32, DT_FP16, DT_INT8};
    char *type[] = {"FP32", "FP16", "INT8"};
    for (int i = 0; i < 7; ++i) {
        int src_n = shape[i][0];
        int src_c = shape[i][1];
        int src_h = shape[i][2];
        int src_w = shape[i][3];
        printf("\nsrc_n=%d, src_c=%d, src_h=%d, src_w=%d, \n",
               src_n, src_c, src_h, src_w);
        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            char *perc = type[j];
            u64 tensor_size = src_n * src_c * src_h * src_w * tpu_data_type_size(dtype);
            BEGIN()
            general_cwtrans_gen_cmd(
                input_addr, 0,
                0, 0,
                src_n, src_c, src_h, src_w,
                ((dtype) >> 1),
                0, 0, 0,
                0, 1, 0,
                0, GDMA_S2L, MASTER_THREAD,
                &pid_node
            );
            END(Tensor_CWTrans_S2L, perc, tensor_size);

            int src_lsize = tpu_get_local_size((const dim4 *)shape[i], dtype, 0, true);
            int dst_lsize = tpu_get_local_size((const dim4 *)shape[i], dtype, 0, true);
            if (src_lsize <= LOCAL_MEM_SIZE / 2 && dst_lsize <= LOCAL_MEM_SIZE / 2) {
                BEGIN()
                general_cwtrans_gen_cmd(
                    0, 0,
                    LOCAL_MEM_SIZE / 2, 0,
                    src_n, src_c, src_h, src_w,
                    ((dtype) >> 1),
                    0, 0, 0,
                    0, 1, 0,
                    0, GDMA_L2L, MASTER_THREAD,
                    &pid_node
                );
                END(Tensor_CWTrans_L2L, perc, tensor_size);
            }
        }
    }
#undef BEGIN
#undef END
}
