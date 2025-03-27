#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gen_cmd.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_cw_trans_test()
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < 10; ++t) {
#define END(info, prec) \
    }                   \
    tpu_poll();         \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s cw transpose %s \n", #info, prec);   \
    printf("Total %s time: %lldus\n\n", #info, (end_time - start_time))

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int shapes[][4] =
    {   {1, 128, 32, 128},
        {1, 3, 128, 128},
        {1, 1, 1, 1024},
        {1, 1024, 1, 1}
    };

    const int bank_size = LOCAL_BANK_SIZE;
    data_type_t data_type[] = {DT_FP32, DT_FP16, DT_INT8};
    char *type[] = {"FP32", "FP16", "INT8"};
    for (int i = 0; i < 4; ++i) {
        dim4 shape = {.n = shapes[i][0], .c = shapes[i][3],
                      .h = shapes[i][2], .w = shapes[i][1]
                     };
        printf("\n src_n=%d, src_c=%d, src_h=%d, src_w=%d, \n",
               shape.n, shape.w, shape.h, shape.c);
        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            char *perc = type[j];
            BEGIN()
            tpu_bdc_cw_trans(
                8 * bank_size,
                0,
                &shape,
                dtype);
            END(C_W_TRANS, perc);
            BEGIN()
            tpu_bdc_wc_trans(
                0,
                8 * bank_size,
                &shape,
                dtype);
            END(W_C_TRANS, perc);
        }
    }
#undef BEGIN
#undef END
}
