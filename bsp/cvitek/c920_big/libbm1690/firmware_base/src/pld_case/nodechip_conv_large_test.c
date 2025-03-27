#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_conv_large_test(
    unsigned long long input_global_addr,
    unsigned long long output_global_addr) {
  tpu_initialize();

  data_type_t input_dtype = DT_FP32;
  data_type_t output_dtype = DT_FP32;

  dim4 ishape = {1, 32, 32, 300};
  dim2 kernel = {7, 7};
  padding_t pad = {3, 3, 3, 3};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim4 oshape = {1, 64, 32, 300};

  local_addr_t input_addr = 0;
  local_addr_t output_addr = ishape.n * DIV_UP(ishape.c, NPU_NUM) * ALIGN(ishape.h * ishape.w, 64) * sizeof(float);
  local_addr_t weight_addr = output_addr + oshape.n * DIV_UP(oshape.c, NPU_NUM) * ALIGN(oshape.h * oshape.w, 64) * sizeof(float);
  printf("output_addr = %x  weight_addr = %x\n", output_addr, weight_addr);

  //////////////////////////////////////////////////////
  //    float conv
  /////////////////////////////////////////////////////
  // load weight
  dim4 wshape = {oshape.c, ishape.c, kernel.h, kernel.w};
  tpu_gdma_compact_S2L(
      weight_addr,
      input_global_addr,
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
      input_dtype,
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
  //////////////////////////////////////////////////////
  //    quant conv
  /////////////////////////////////////////////////////
  input_dtype = DT_INT8;
  output_dtype = DT_INT8;
  ishape.n *= 4;
  oshape.n *= 4;
  tpu_gdma_compact_S2L(
      weight_addr,
      input_global_addr + ishape.n * ishape.c * ishape.h * ishape.w,
      &wshape,
      input_dtype);
  // load input
  tpu_gdma_cpy_S2L(
      input_addr,
      input_global_addr + ishape.n * ishape.c * ishape.h * ishape.w,
      &ishape,
      NULL,
      NULL,
      input_dtype);
  // conv quant
  tpu_bdc_int8_sym_quant_conv2d(
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
      input_dtype,
      input_dtype,
      input_dtype,
      10,
      false,
      false);
  // store output
  tpu_gdma_cpy_L2S(
      output_global_addr + oshape.n * oshape.c * oshape.h * oshape.w,
      output_addr,
      &oshape,
      NULL,
      NULL,
      output_dtype);
  tpu_poll();
}
