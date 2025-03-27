#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_gdma_stride_test(
    unsigned long long input_addr,
    unsigned long long output_addr) {
  tpu_initialize();

  dim4 shape = {2, 4, 8, 8};
  dim4 in_stride = {
    4 * 8 * 8,
    8 * 8,
    8,
    1};
  dim4 out_stride = {
    8 * 4 * 8,
    8,
    4 * 8,
    1};

  tpu_gdma_cpy_S2S(
    output_addr,
    input_addr,
    &shape,
    &out_stride,
    &in_stride,
    DT_FP32);

  tpu_poll(); 
}
