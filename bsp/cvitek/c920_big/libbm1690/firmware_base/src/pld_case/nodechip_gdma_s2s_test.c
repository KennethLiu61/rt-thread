#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_s2s_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us()
#define END(info, prec) \
    tpu_poll();         \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s gdma s2s %d\n", #info, prec);            \
    printf("Total %s time: %lldus\n\n", #info, (end_time - start_time))

#define GDMA_CPY_S2S(in_stride, out_stride, stride_dim) \
    BEGIN();    \
    tpu_gdma_cpy_S2S(               \
        output_addr + out_offset,   \
        input_addr + in_offset,     \
        &shape,                     \
        &out_stride, &in_stride,    \
        dtype);                     \
    END(stride_dim, dtype)

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;
    const dim4 shape1 = {.n = 1, .c = 8, .h = 1024, .w = 1024};
    const dim4 shape2 = {.n = 1024, .c = 1024, .h = 1, .w = 8};

    data_type_t data_type[3] = {DT_UINT8, DT_INT16, DT_INT32};
    dim4 shapes[2] = {shape1, shape2};
    dim4 continue_stride;
    dim4 n_concat_stride;
    dim4 c_concat_stride;
    dim4 h_concat_stride;
    dim4 w_concat_stride;
    dim4 slice_stride;
    u64 in_offset = 0;
    u64 out_offset = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            int type_size = tpu_data_type_size(dtype);
            dim4 shape = shapes[i];
            shape.w = shape.w / type_size;
            tpu_continuous_stride(&continue_stride, &shape);
            n_concat_stride.w = 1;
            n_concat_stride.h = shape.w;
            n_concat_stride.c = shape.h * n_concat_stride.h;
            n_concat_stride.n = shape.c * n_concat_stride.c * 2;
            c_concat_stride.w = 1;
            c_concat_stride.h = shape.w;
            c_concat_stride.c = shape.h * c_concat_stride.h * 2;
            c_concat_stride.n = shape.c * c_concat_stride.c;
            h_concat_stride.w = 1;
            h_concat_stride.h = shape.w * 2;
            h_concat_stride.c = shape.h * h_concat_stride.h;
            h_concat_stride.n = shape.c * h_concat_stride.c;
            w_concat_stride.w = 4;
            w_concat_stride.h = shape.w * w_concat_stride.w;
            w_concat_stride.c = shape.h * w_concat_stride.h ;
            w_concat_stride.n = shape.c * w_concat_stride.c;
            slice_stride.w = 2;
            slice_stride.h = shape.w * slice_stride.w * 2;
            slice_stride.c = shape.h * slice_stride.h * 2;
            slice_stride.n = shape.c * slice_stride.c;

            BEGIN();
            tpu_gdma_cpy_S2S(
                output_addr + out_offset,
                input_addr + in_offset,
                &shape,
                0, 0,
                dtype);
            END(continue_2_continue, dtype);

            GDMA_CPY_S2S(continue_stride, continue_stride, continue_2_continue);
            GDMA_CPY_S2S(continue_stride, n_concat_stride, continue_2_n_concat);
            GDMA_CPY_S2S(n_concat_stride, continue_stride, n_concat_2_continue);
            GDMA_CPY_S2S(continue_stride, c_concat_stride, continue_2_c_concat);
            GDMA_CPY_S2S(c_concat_stride, continue_stride, c_concat_2_continue);
            GDMA_CPY_S2S(continue_stride, h_concat_stride, continue_2_h_concat);
            GDMA_CPY_S2S(h_concat_stride, continue_stride, h_concat_2_continue);
            GDMA_CPY_S2S(continue_stride, w_concat_stride, continue_2_w_concat);
            GDMA_CPY_S2S(w_concat_stride, continue_stride, w_concat_2_continue);
            GDMA_CPY_S2S(continue_stride, slice_stride, continue_2_slice);
            GDMA_CPY_S2S(slice_stride, continue_stride, slice_2_continue);
        }
    }
//    for (int i = 0; i < 3; ++i) {
//        data_type_t dtype = data_type[i];
//        int type_size = tpu_data_type_size(dtype);
//        dim4 shape = {.n = 8, .c = 1024, .h = 256, .w = 4 / type_size};
//        dim4 in_stride = {.n = 0, .c = 1, .h = 0, .w = 0};
//        tpu_continuous_stride(&continue_stride, &shape);
//        GDMA_CPY_S2S(in_stride, continue_stride, broadcast);
//    }

#undef GDMA_CPY_S2S
#undef BEGIN
#undef END
}
