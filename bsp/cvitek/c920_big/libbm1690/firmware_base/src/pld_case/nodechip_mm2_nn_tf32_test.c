    #include "nodechip_pld_test.h"
#include "tpu_kernel.h"


void nodechip_mm2_nn_tf32_test(
    unsigned long long input_addr,
    unsigned long long output_addr) {
  tpu_initialize();

  const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

  data_type_t output_dtype = DT_FP32;
  data_type_t left_right_dtype = DT_TF32;
  data_type_t bias_dtype = DT_FP32;

  int left_rows = 8;
  int left_cols = 32;
  int right_cols = 16;
  bool result_add = false;
  bool bias_is_const = false;
  bool do_relu = false;

  dim4 Y_shape = {1, left_rows, 1, right_cols};
  dim4 L_shape = {1, left_rows, 1, left_cols};
  dim4 R_shape = {1, left_cols, 1, right_cols};

  dim4 Y_stride, L_stride, R_stride;
  local_addr_t Y_l_addr = 0;
  tpu_aligned_stride(&Y_stride, 0, &Y_shape, output_dtype);
  system_addr_t Y_g_addr = output_addr;

  tpu_aligned_stride(&L_stride, 0, &L_shape, output_dtype);
  local_addr_t L_l_addr = ALIGN(Y_l_addr + Y_stride.n * tpu_data_type_bits(output_dtype) / 8, bank_size);
  system_addr_t L_g_addr = input_addr;

  tpu_aligned_stride(&R_stride, 0, &R_shape, output_dtype);
  local_addr_t R_l_addr = ALIGN(L_l_addr + L_stride.n * tpu_data_type_bits(output_dtype) / 8, bank_size);
  system_addr_t R_g_addr = L_g_addr + left_rows * left_cols * sizeof(float);

  local_addr_t bias_l_addr = ALIGN(R_l_addr + R_stride.n * tpu_data_type_bits(output_dtype) / 8, bank_size);
  var_context_t bias_data = {.addr = bias_l_addr};
  system_addr_t bias_g_addr = R_g_addr + left_cols * right_cols * sizeof(float);

  {
    dim4 shape = {1, left_rows, 1, left_cols};
    tpu_gdma_cpy_S2L(L_l_addr,
                     L_g_addr,
                     &shape,
                     NULL,
                     NULL,
                     DT_FP32);
  }
  {
    dim4 shape = {1, left_cols, 1, right_cols};
    tpu_gdma_cpy_S2L(R_l_addr,
                     R_g_addr,
                     &shape,
                     NULL,
                     NULL,
                     DT_FP32);
  }

  {
    dim4 shape = {1, tpu_npu_num(), 1, right_cols};
    tpu_gdma_channel_bcast_S2L(bias_l_addr,
                               bias_g_addr,
                               &shape,
                               NULL,
                               NULL,
                               DT_FP32);
  }

  tpu_bdc_fp_mm_with_bias(
      Y_l_addr,
      L_l_addr,
      R_l_addr,
      left_rows,
      left_cols,
      right_cols,
      output_dtype,
      left_right_dtype,
      bias_dtype,
      result_add,
      bias_is_const,
      bias_data,
      do_relu);

  {
    dim4 shape = {1, left_rows, 1, right_cols};
    tpu_gdma_cpy_L2S(Y_g_addr,
                     Y_l_addr,
                     &shape,
                     NULL,
                     NULL,
                     DT_FP32);
  }

  tpu_poll();
}
