#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_matrix_perf_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us();
#define END(info, prec) \
    tpu_poll();         \
    end_time = firmware_timer_get_time_us();    \
    printf("test gdma matrix %s \n", prec);            \
    printf("Total time: %lldus\n", (end_time - start_time))

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    data_type_t dtype = DT_INT8;
    char *perc = "INT8";
    int raw = 256;
    int col = 4096;
    int row_stride = 4096 * 4;
    int col_per_channel = tpu_eu_num(dtype);
    int tensor_size = raw * col;
    printf("\nraw=%d, col=%d, col_per_channel=%d, row_stride=%d, \n",
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
    float bw = (float)tensor_size/(float)((end_time-start_time) * 1e-6);
    printf("S2L Average bandwidth : %.3fMB/s\n\n", bw/1024/1024);

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
    printf("\nraw=%d, col=%d, col_per_channel=%d, row_stride=%d, \n",
            col, raw, col_per_channel, row_stride);
    bw = (float)tensor_size/(float)((end_time-start_time) * 1e-6);
    printf("L2S Average bandwidth : %.3fMB/s\n\n", bw/1024/1024);
#undef BEGIN
#undef END
}
