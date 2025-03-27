#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_stride_mv_perf_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us();
#define END(info, prec) \
    tpu_poll();         \
    end_time = firmware_timer_get_time_us();    \
    printf("test tensor stride move %s \n", prec);            \
    printf("Total time: %lldus\n", (end_time - start_time))

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    data_type_t dtype = DT_INT8;
    char *perc = "INT8";

    dim4 shape = {256, 32, 16, 4};
    dim4 l_stride = {64, 64, 4, 1};
    dim4 g_stride = {128 * 64 * 2, 128, 8, 1};
    int tensor_size = shape.n * shape.c * shape.h * shape.w;

    BEGIN()
    tpu_gdma_cpy_S2L(
      0,
      input_addr,
      &shape,
      &l_stride,
      &g_stride,
      dtype);
    END(S2L, perc);
    printf("n:%d, c:%d, h%d, w:%d, ln_stride:%d, lc_stride:%d, lh_stride:%d, lw_stride:%d, gn_stride:%d, gc_stride:%d, gh_stride:%d, gh_stride:%d\n",
            shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    float bw = (float)tensor_size/(float)((end_time-start_time) * 1e-6);
    printf("S2L Average bandwidth : %.3fMB/s\n\n", bw/1024/1024);

    BEGIN()
    tpu_gdma_cpy_L2S(
      output_addr,
      0,
      &shape,
      &g_stride,
      &l_stride,
      dtype);
    END(L2S, perc);
    printf("n:%d, c:%d, h%d, w:%d, ln_stride:%d, lc_stride:%d, lh_stride:%d, lw_stride:%d, gn_stride:%d, gc_stride:%d, gh_stride:%d, gh_stride:%d\n",
            shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    bw = (float)tensor_size/(float)((end_time-start_time) * 1e-6);
    printf("L2S Average bandwidth : %.3fMB/s\n\n", bw/1024/1024);
#undef BEGIN
#undef END
}
