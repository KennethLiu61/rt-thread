
#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_conv_tf32_test(
    unsigned long long input_global_addr,
    unsigned long long output_global_addr) {
  tpu_initialize();

  data_type_t input_tiu_dtype = DT_TF32;
  data_type_t input_dtype = DT_FP32;
  data_type_t output_dtype = DT_FP32;

  dim4 ishape = {1, 32, 8, 8};
  dim2 kernel = {7, 7};
  padding_t pad = {3, 3, 3, 3};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim4 oshape = {1, 64, 8, 8};

  local_addr_t input_addr = 0;
  local_addr_t output_addr = ishape.n * DIV_UP(ishape.c, NPU_NUM) * ALIGN(ishape.h * ishape.w, 64) * sizeof(float);
  local_addr_t weight_addr = output_addr + oshape.n * DIV_UP(oshape.c, NPU_NUM) * ALIGN(oshape.h * oshape.w, 64) * sizeof(float);
  printf("output_addr = %x  weight_addr = %x\n", output_addr, weight_addr);

  int isz = ishape.n * ishape.c * ishape.h * ishape.w * sizeof(float);
  unsigned long long weight_global_addr = input_global_addr + isz;

  //////////////////////////////////////////////////////
  //    float conv
  /////////////////////////////////////////////////////
  // load weight
  const int ic_parallel = 16;
  dim4 wshape = {1, oshape.c, DIV_UP(ishape.c, ic_parallel), kernel.h * kernel.w * ic_parallel};
  tpu_gdma_compact_S2L(
      weight_addr,
      weight_global_addr,
      &wshape,
      input_dtype);
  // load input
  tpu_gdma_cpy_S2L(
      input_addr,
      input_global_addr,
      &ishape,
      NULL,
      NULL,
      input_dtype);
  // conv float
  tpu_bdc_fp_conv2d(
      output_addr,
      input_addr,
      weight_addr,
      0,
      &ishape,
      NULL,
      oshape.c,
      &kernel,
      &pad,
      &stride,
      &dilation,
      output_dtype,
      input_tiu_dtype,
      false,
      false);
  // store output
  tpu_gdma_cpy_L2S(
      output_global_addr,
      output_addr,
      &oshape,
      NULL,
      NULL,
      output_dtype);
  tpu_poll();
}
