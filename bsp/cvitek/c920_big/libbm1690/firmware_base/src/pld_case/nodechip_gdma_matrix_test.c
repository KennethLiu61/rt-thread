#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_matrix_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < 10; ++t) {
#define END(info, prec) \
    }                   \
    tpu_poll();         \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s gdma matrix %s \n", #info, prec);            \
    printf("Total %s time: %lldus\n", #info, (end_time - start_time));   \
    bw = (float)raw * col * tpu_data_type_size(dtype) / (float)((end_time - start_time) * 1e-6);   \
    printf("Average bandwidth : %.3fGB/s\n\n", bw * 10 / 1024.f / 1024 / 1024)

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int param[][4] =
    {   {128, 256, 32, 256},
        {1024, 1024, 32, 1024},
        {128, 1024, 32, 4096},
    };

    float bw;
    data_type_t data_type[] = {DT_FP32, DT_FP16, DT_INT8};
    char *type[] = {"FP32", "FP16", "INT8"};
    for (int i = 0; i < 3; ++i) {
        int raw = param[i][0];
        int col = param[i][1];
        int row_stride = param[i][3];
        int col_per_channel = param[i][2];

        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            char *perc = type[j];
//            int col_per_channel = tpu_eu_num(dtype);
            printf("\n raw=%d, col=%d, col_per_channel=%d, row_stride=%d, \n",
                   raw, col, col_per_channel, row_stride);
            BEGIN()
            tpu_gdma_matrix_S2L(
                        0,
                        input_addr,
                        raw, col,
                        col_per_channel,
                        row_stride,
                        dtype
                        );
            END(S2L, perc);
            BEGIN()
            tpu_gdma_matrix_L2S(
                        output_addr,
                        0,
                        raw, col,
                        col_per_channel,
                        row_stride,
                        dtype
                        );
            END(L2S, perc);
            printf("\n raw=%d, col=%d, col_per_channel=%d, row_stride=%d, \n",
                   col, raw, col_per_channel, row_stride);
            BEGIN()
            tpu_gdma_matrix_trans_S2L(
                        0,
                        output_addr,
                        col, raw,
                        col_per_channel,
                        row_stride,
                        dtype
                        );
            END(TRANS_S2L, perc);
            BEGIN()
            tpu_gdma_matrix_trans_L2S(
                        output_addr,
                        0,
                        col, raw,
                        col_per_channel,
                        row_stride,
                        dtype
                        );
            END(TRANS_L2S, perc);
        }
    }
#undef BEGIN
#undef END
}
