#include "nodechip_pld_test.h"
#include "tpu_kernel.h"
#include "conv_util.h"

static void nodechip_conv_bw_common_test(
  global_addr_t input_global_addr,
  global_addr_t out_grad_global_addr,
  global_addr_t pad_ins_addr,
  global_addr_t res_global_addr,
  data_type_t   opd0_dtype,
  data_type_t   opd1_dtype,
  dim4          opd0_large_shape,
  int           with_str)
{
  tpu_initialize();
  int oc = 16;
  int n = opd0_large_shape.n;
  int ic = opd0_large_shape.c;
  dim4 opd0_shape = {n, ic, 8, 8};
  dim2 kernel = {3, 3};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim2 insert = {1, 1};
  padding_t pad = {1, 1, 2, 2};
  data_type_t res_dtype = DT_FP32;

  int oh = cal_conv2d_out_size(opd0_shape.h, kernel.h, dilation.h, stride.h, pad.top, pad.bottom, insert.h);
  int ow = cal_conv2d_out_size(opd0_shape.w, kernel.w, dilation.w, stride.w, pad.left, pad.right, insert.w);
  dim4 opd1_shape = {n, oc, oh, ow};
  dim4 res_shape = {1, opd1_shape.c, 1, 1};
  if (opd0_dtype == DT_FP32) {
    res_shape.h = kernel.h * kernel.w * opd0_shape.c;
  } else if (opd0_dtype == DT_FP8E5M2 || opd0_dtype == DT_FP8E4M3 ||
             opd0_dtype == DT_TF32 || opd0_dtype == DT_FP16 || opd0_dtype == DT_BFP16) {
    res_shape.h = kernel.h * kernel.w * DIV_UP(opd0_shape.c, tpu_eu_num(opd0_dtype));
    res_shape.w = tpu_eu_num(opd0_dtype);
  } else {
    ASSERT(0);
  }

  dim4 opd0_stride, opd1_stride, res_stride;
  TPUKERNEL_ASSERT(with_str == 1);
  tpu_aligned_stride(&opd0_stride, 0, &opd0_large_shape, opd0_dtype);
  tpu_aligned_stride(&opd1_stride, 0, &opd1_shape, opd1_dtype);
  tpu_aligned_stride(&res_stride, 0, &res_shape, res_dtype);

  local_addr_t opd0_addr = 0;
  local_addr_t opd1_addr = opd0_addr +  opd0_stride.n * opd0_large_shape.n * get_bytesize(opd0_dtype);
  local_addr_t res_addr = opd1_addr + opd1_stride.n * opd1_shape.n * get_bytesize(opd1_dtype);
  ASSERT((int)res_addr + res_shape.n * res_stride.n * get_bytesize(res_dtype) < LOCAL_MEM_SIZE);

  tpu_gdma_cpy_S2L(opd0_addr, input_global_addr, &opd0_large_shape, NULL, NULL, opd0_dtype);
  tpu_gdma_cpy_S2L(opd1_addr, out_grad_global_addr, &opd1_shape, NULL, NULL, opd1_dtype);

  CORE_PRINT("opd0_addr:%u  opd1_addr:%u  res_addr:%u  opd0_shape(%d %d %d %d) opd1_shape(%d %d %d %d)\n",
              opd0_addr, opd1_addr, res_addr, opd0_shape.n, opd0_shape.c, opd0_shape.h, opd0_shape.w, opd1_shape.n, opd1_shape.c, opd1_shape.h, opd1_shape.w);
  CORE_PRINT("res_shape(%d %d %d %d) opd0_stride(%d %d %d %d)\n", res_shape.n, res_shape.c, res_shape.h, res_shape.w, opd0_stride.n, opd0_stride.c, opd0_stride.h, opd0_stride.w);
  CORE_PRINT("kh:%d   kw:%d  stride_h:%d  stride_w:%d  dilation_h:%d  dilation_w:%d  insert_h:%d  insert_w:%d  pad(top:%d  bottom:%d  left:%d  right:%d)\n\n",
            kernel.h, kernel.w, stride.h, stride.w, dilation.h, dilation.w, insert.h, insert.w, pad.top, pad.bottom, pad.left, pad.right);

  tpu_bdc_fp_conv2d_backward(res_addr, opd0_addr, opd1_addr, NO_USE, &opd0_shape, &opd1_shape, &kernel,
                            &insert, &pad, &stride, &dilation, &opd0_stride, true, 0, PAD_CONSTANT, opd0_dtype, opd1_dtype);


  tpu_gdma_cpy_L2S(res_global_addr, res_addr, &res_shape, NULL, NULL, res_dtype);
  tpu_poll();
}

void nodechip_conv_bw_fp8_test(
  unsigned long long input_global_addr,
  unsigned long long output_global_addr) {
  int ic = 16;
  int n = 1;
  dim4 opd0_large_shape = {n, ic, 16, 16};
  dim4 opd0_large_stride;
  tpu_continuous_stride(&opd0_large_stride, &opd0_large_shape);
  unsigned long long out_grad_shape = input_global_addr + opd0_large_shape.n * opd0_large_stride.n * tpu_data_type_size(DT_FP8E4M3);
  CORE_PRINT("------ conv bw fp8 test -----\n");
  nodechip_conv_bw_common_test(input_global_addr, out_grad_shape, NO_USE, output_global_addr, DT_FP8E4M3, DT_FP8E5M2, opd0_large_shape, true);
}

void nodechip_conv_bw_fp32_test(
  unsigned long long input_global_addr,
  unsigned long long output_global_addr) {
  int ic = 16;
  int n = 1;
  dim4 opd0_large_shape = {n, ic, 16, 16};
  dim4 opd0_large_stride;
  tpu_continuous_stride(&opd0_large_stride, &opd0_large_shape);
  unsigned long long out_grad_shape = input_global_addr + opd0_large_shape.n * opd0_large_stride.n * tpu_data_type_size(DT_FP32);
  CORE_PRINT("------ conv bw fp32 test -----\n");
  nodechip_conv_bw_common_test(input_global_addr, out_grad_shape, NO_USE, output_global_addr, DT_FP32, DT_FP32, opd0_large_shape, true);
}

void nodechip_conv_bw_tf32_test(
  unsigned long long input_global_addr,
  unsigned long long output_global_addr) {
  int ic = 16;
  int n = 1;
  dim4 opd0_large_shape = {n, ic, 16, 16};
  dim4 opd0_large_stride;
  tpu_continuous_stride(&opd0_large_stride, &opd0_large_shape);
  unsigned long long out_grad_shape = input_global_addr + opd0_large_shape.n * opd0_large_stride.n * tpu_data_type_size(DT_TF32);
  CORE_PRINT("------ conv bw tf32 test -----\n");
  nodechip_conv_bw_common_test(input_global_addr, out_grad_shape, NO_USE, output_global_addr, DT_TF32, DT_TF32, opd0_large_shape, true);
}