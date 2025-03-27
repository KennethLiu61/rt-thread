#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_base_bdc_test() {
  tpu_initialize();
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();

  data_type_t dtype = DT_FP32;

  int oc = 64;
  dim4 input_shape = {1, 64, 20, 20};
  dim4 input_stride = {0};
  tpu_aligned_stride(&input_stride, 0, &input_shape, dtype);
  dim2 kernel_size = {3, 3};
  padding_t pad_size = {1, 1, 1, 1};
  dim2 stride = {2, 1};
  dim2 dilation = {1, 1};

  if (tpu_is_data_type_fp(dtype)) {
    tpu_bdc_fp_conv2d(0,
                      4 * bank_size,
                      8 * bank_size,
                      12 * bank_size,
                      &input_shape,
                      &input_stride,
                      oc,
                      &kernel_size,
                      &pad_size,
                      &stride,
                      &dilation,
                      dtype,
                      dtype,
                      1,
                      0);
  } else {
    tpu_bdc_int8_sym_quant_conv2d(
        0,
        4 * bank_size,
        8 * bank_size,
        12 * bank_size,
        &input_shape,
        &input_stride,
        oc,
        &kernel_size,
        &pad_size,
        &stride,
        &dilation,
        dtype,
        dtype,
        dtype,
        dtype,
        2,
        0,
        0);
  }
  tpu_poll();
}