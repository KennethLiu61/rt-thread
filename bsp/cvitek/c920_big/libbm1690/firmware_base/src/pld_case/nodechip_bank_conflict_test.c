#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_bank_conflict_test(
    unsigned long long input_global_addr,
    unsigned long long output_global_addr) {
  tpu_initialize();

  data_type_t input_dtype = DT_INT8;
  data_type_t output_dtype = DT_INT8;
  data_type_t weight_dtype = DT_INT8;

  int oc = 32;
  int rshift_num = 9;
  dim4 input_shape = {1, 32, 20, 20};
  dim2 kernel_size = {3, 3};
  // padding_t pad_size = {1, 1, 1, 1};
  dim2 stride = {2, 1};
  dim2 dilation = {1, 1};
  dim4 output_shape = {1, 64, 10, 20};

  // split two parts for parallel computing
  // weight_shape = (64, 32, 3, 3)
  // output_shape = (1, 64, 10, 20)
  // 1. input_shape = (1, 64, 10, 20) --> output (1, 64, 5, 20)
  //   -- pad_h_t = 1, h = 0~9 => size_per_npu = 10*20=200 bytes
  // 2. input_shape = (1, 64, 11, 20) --> output (1, 64, 5, 20)
  //   -- pad_h_b = 1(unused) h = 9~19 => size_per_npu = 11 * 20 = 200 bytes
  int in_offset[2] = {0, 20*9};
  int out_offset[2] = {0, 20*5};
  dim4 ishape[2] = { {1, 32, 10, 20}, {1, 32, 11, 20} };
  dim4 oshape[2] = { {1, 32, 5, 20}, {1, 32, 5, 20} };
  padding_t padding[2] = { {1, 0, 1, 1}, {0, 1, 1, 1} };

  // change it to test different bank
  local_addr_t input_addr[2] = {0, 1024};
  local_addr_t output_addr[2] = {2048, 3072};
  local_addr_t weight_addr = 4096;

  // load weight and bias
  dim4 wshape = {64, 32, 3, 3};
  dim4 weight_global_stride = {0};
  tpu_continuous_stride(&weight_global_stride, &wshape);
  dim4 weight_local_stride;
  tpu_compact_stride(&weight_local_stride, 0, &wshape);
  tpu_gdma_cpy_S2L(
      weight_addr,
      input_global_addr,
      &wshape,
      &weight_local_stride,
      &weight_global_stride,
      weight_dtype);

  dim4 input_global_stride = {0}, output_global_stride = {0};
  dim4 input_local_stride = {0}, output_local_stride = {0};
  tpu_continuous_stride(&input_global_stride, &input_shape);
  tpu_continuous_stride(&output_global_stride, &output_shape);
  for (int i = 0; i < 3; ++i) {
    // load
    if (i < 2) {
      tpu_aligned_stride(&input_local_stride, 0, &(ishape[i%2]), input_dtype);
      tpu_gdma_cpy_S2L(
          input_addr[i % 2],
          input_global_addr + in_offset[i],
          &(ishape[i % 2]),
          &input_local_stride,
          &input_global_stride,
          input_dtype);
    }

    if (tpu_is_parallel_state()) {
      tpu_parallel_end();
    }
    tpu_parallel_start();

    // compute
    if (i < 2) {
      tpu_bdc_int8_sym_quant_conv2d(
          output_addr[i % 2],
          input_addr[i % 2],
          weight_addr,
          0,
          &(ishape[i % 2]),
          NULL,
          oc,
          &kernel_size,
          &(padding[i % 2]),
          &stride,
          &dilation,
          output_dtype,
          input_dtype,
          weight_dtype,
          DT_INT32,
          rshift_num,
          false,
          false);
    }

    if (i > 0) {
      tpu_aligned_stride(&output_local_stride, 0, &(oshape[(i-1)%2]), output_dtype);
      tpu_gdma_cpy_L2S(
          output_global_addr + out_offset[i-1],
          output_addr[(i-1) % 2],
          &(oshape[(i-1) % 2]),
          &output_global_stride,
          &output_local_stride,
          output_dtype);
    }
  }

  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }

  tpu_poll();
}
