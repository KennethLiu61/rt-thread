#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_ar_stride_test(
    unsigned long long input_addr,
    unsigned long long output_addr) {

    tpu_initialize();

    dim4 shape = {2, 4, 4, 8};

    local_addr_t input_lo = 0;
    local_addr_t output_lo = LOCAL_MEM_SIZE / 2;

    tpu_gdma_cpy_S2L(input_lo,
                     input_addr,
                     &shape,
                     NULL,
                     NULL,
                     DT_FP32);

    dim4 stride;
    tpu_compact_stride(&stride, 0, &shape);
    stride.w = shape.h;
    stride.h = 1;

    tpu_bdc_fp_add(
        output_lo,
        input_lo,
        input_lo,
        &shape,
        &stride,
        NULL,
        NULL,
        DT_FP32);

    tpu_gdma_cpy_L2S(output_addr,
                     output_lo,
                     &shape,
                     NULL,
                     &stride,
                     DT_FP32);

    tpu_poll();
}
