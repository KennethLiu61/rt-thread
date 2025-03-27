#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

static const int LR_dtypes[] = {
    DT_INT8,
    DT_FP16,
    DT_BFP16,
    DT_FP16,
    DT_BFP16,
    DT_TF32,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP8E5M2,
    DT_FP8E4M3,
};
static const int odtypes[] = {
    DT_INT8,
    DT_INT16,
    DT_INT32,
};
static const char *const odtypes_str[] = {
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
};
static const int idtypes[] = {
    DT_INT8,
};
static const char *const idtypes_str[] = {
    "DT_INT8",
};
static const int Y_dtypes[] = {
    DT_INT32,
    DT_FP16,
    DT_BFP16,
    DT_FP32,
    DT_FP32,
    DT_FP32,
    DT_FP32,
    DT_FP32,
    DT_FP16,
    DT_FP16,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP8E4M3,
    DT_FP8E5M2,
};
static const char *const dtype_str[] = {
    "LR-DT_INT8, Y-DT_INT32",
    "LR-DT_FP16, Y-DT_FP16",
    "LR-DT_BFP16, Y-DT_BFP16",
    "LR-DT_FP16, Y-DT_FP32",
    "LR-DT_BFP16, Y-DT_FP32",
    "LR-DT_TF32, Y-DT_FP32",
    "LR-DT_FP8E5M2,Y-DT_FP32",
    "LR-DT_FP8E4M3,Y-DT_FP32",
    "LR-DT_FP8E5M2,Y-DT_FP16",
    "LR-DT_FP8E4M3,Y-DT_FP16",
    "LR-DT_FP8E5M2,Y-DT_FP8E5M2",
    "LR-DT_FP8E4M3,Y-DT_FP8E4M3",
    "LR-DT_FP8E5M2,Y-DT_FP8E4M3",
    "LR-DT_FP8E4M3,Y-DT_FP8E5M2",
};
static const scalar_t LR_const[] = {
    {.s8 = 1},
    {.f16.bits = 0x7800},
    {.bf16.bits = 0x3f80},
    {.f16.bits = 0x7800},
    {.bf16.bits = 0x3f80},
    {.f32 = 1.234f},
    {.f8e3m4.bits = 0x70},
    {.f8e5m2.bits = 0xab},
    {.f8e3m4.bits = 0x70},
    {.f8e5m2.bits = 0xab},
    {.f8e3m4.bits = 0x70},
    {.f8e5m2.bits = 0xab},
    {.f8e5m2.bits = 0xab},
    {.f8e3m4.bits = 0x70},
};
static const scalar_t ZP_const = {.s16 = 3};
static const unsigned int count = sizeof(LR_dtypes)/sizeof(int);

u64 mm2_ops(
    int M,
    int N,
    int K,
    bool has_bias,
    bool do_rq,
    bool do_relu,
    bool result_add)
{
  u64 total = (u64)M * N *
              (2 * K +
               (has_bias ? 1 : 0) +
               (do_relu ? 1 : 0) +
               (do_rq ? 1 : 0) +
               (result_add ? 1 : 0));
  return total;
}

void nodechip_mm2_perf_test()
{
  (void) ZP_const;
  tpu_initialize();
  dim4 Y_stride, L_stride, R_stride, ZP_stride;
  const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  const int loop = 5;
  u64 st = 0, et = 0, ops;
  optional_info_t left = {0};
  optional_info_t right = {0};
  optional_info_t rzp = {0};
  requant_int_info_t requant = {0};
  optional_info_t bias = {0};

  int left_rows = 512;
  int left_cols = 512 * 2;
  int right_cols = 512; // right_rows if trans
  bool result_add = false;
  printf("Test MM2:\n");
  printf("loop time = %d per case\n", loop);
  printf("left_rows = %d, left_cols = %d, right_cols = %d\n",
         left_rows, left_cols, right_cols);

  for (int odtype = 0; odtype < 3; ++odtype)
  {
    dim4 Y_shape = {1, left_rows, 1, right_cols};
    dim4 L_shape = {1, left_rows, 1, left_cols};
    dim4 R_shape = {1, left_cols, 1, right_cols};
    dim4 ZP_shape = {1, NPU_NUM, 1, right_cols};
    tpu_aligned_stride(&ZP_stride, 0, &ZP_shape, DT_INT16);
    local_addr_t Y_addr = 0;
    tpu_aligned_stride(&Y_stride, 0, &Y_shape, odtypes[odtype]);
    for (int ldtype = 0; ldtype < 1; ++ldtype)
    {
      tpu_aligned_stride(&L_stride, 0, &L_shape, idtypes[ldtype]);
      local_addr_t L_addr = ALIGN(Y_addr + Y_stride.n * tpu_data_type_bits(odtypes[odtype]) / 8, bank_size);
      for (int rdtype = 0; rdtype < 1; ++rdtype)
      {
        tpu_aligned_stride(&R_stride, 0, &R_shape, idtypes[ldtype]);
        local_addr_t R_addr = ALIGN(L_addr + L_stride.n * tpu_data_type_bits(idtypes[ldtype]) / 8, bank_size);
        local_addr_t ZP_addr = ALIGN(R_addr + R_stride.n * tpu_data_type_bits(idtypes[rdtype]) / 8, bank_size);
        TPUKERNEL_ASSERT(ZP_addr + ZP_stride.n * sizeof(short) <= (unsigned int)LOCAL_MEM_SIZE);
        for (int left_trans = 0; left_trans < 2; ++left_trans)
        {
          for (int right_trans = 0; right_trans < 2; ++right_trans)
          {
            if (right_trans == 0 && left_trans == 1)
              continue;
            for (int do_relu = 0; do_relu < 2; ++do_relu)
            {
              for (int result_add = 0; result_add < 2; ++result_add)
              {
                if (right_trans == 1 && left_trans == 0 && result_add == 1)
                  continue;
                bias.dtype = DT_INT32;
                rzp.dtype = DT_INT16;
                left.dtype = idtypes[ldtype];
                right.dtype = idtypes[rdtype];
                requant.do_sym_saturate = true;
                requant.is_perchannel = true;
                requant.round_mode = RM_HALF_TO_EVEN;
                left.addr = L_addr;
                right.addr = R_addr;
                left.is_const = false;
                right.is_const = false;
                bias.is_const = true;
                scalar_t bias_C = {.s32 = 2};
                bias.value = bias_C;
                rzp.addr = ZP_addr;

                st = firmware_timer_get_time_us();
                for (int i = 0; i < loop; ++i)
                {
                  tpu_bdc_quant_mm(
                      Y_addr,
                      odtypes[odtype],
                      &left,
                      &right,
                      left_rows,  // output_rows
                      left_cols,  // inner_cols
                      right_cols, // output_cols
                      left_trans,
                      right_trans,
                      result_add,
                      do_relu,
                      &rzp,
                      &bias,
                      &requant);
                }
                tpu_poll();
                et = firmware_timer_get_time_us();
                ops = mm2_ops(left_rows, right_cols, left_cols, true, true, do_relu, result_add);
                printf("Test %s with requant+bias_C+rzp L_dtype=%s R_dtype=%s Y_dtype=%s L_trans=%d R_trans=%d do_relu=%d result_add=%d time %lld us Tops/s=%.2f\n",
                       "tpu_bdc_quant_mm", idtypes_str[ldtype], idtypes_str[rdtype], odtypes_str[odtype],
                       left_trans, right_trans, do_relu, result_add, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
              }
            }
          }
        }
      }
    }
  }

  left_rows = 512;
  left_cols = 256;
  right_cols = 512; // right_rows if trans
  result_add = false;
  printf("Test MM2:\n");
  printf("loop time = %d per case\n", loop);
  printf("left_rows = %d, left_cols = %d, right_cols = %d\n",
         left_rows, left_cols, right_cols);

  dim4 Y_shape = {1, left_rows, 1, right_cols};
  dim4 L_shape = {1, left_rows, 1, left_cols};
  dim4 R_shape = {1, left_cols, 1, right_cols};
  dim4 ZP_shape = {1, NPU_NUM, 1, right_cols};
  tpu_aligned_stride(&Y_stride, 0, &Y_shape, DT_FP32);
  tpu_aligned_stride(&L_stride, 0, &L_shape, DT_FP32);
  tpu_aligned_stride(&R_stride, 0, &R_shape, DT_FP32);
  tpu_aligned_stride(&ZP_stride, 0, &ZP_shape, DT_INT8);

  local_addr_t Y_addr = 0;
  local_addr_t L_addr = ALIGN(Y_addr + Y_stride.n * sizeof(float), bank_size);
  local_addr_t R_addr = ALIGN(L_addr + L_stride.n * sizeof(float), bank_size);
  local_addr_t ZP_addr = ALIGN(R_addr + R_stride.n * sizeof(float), bank_size);
  TPUKERNEL_ASSERT(ZP_addr + ZP_stride.n * sizeof(char) <= (unsigned int)LOCAL_MEM_SIZE);

  // ------------------------------------------------------
  // test MM_NN
  // ------------------------------------------------------
  printf("Test MM_NN:\n");
  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm(
          Y_addr,
          L_addr,
          R_addr,
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          result_add);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, result_add);
    printf("Test %s %s result_add = %d time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm",
            dtype_str[d], result_add, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_L_const(
          Y_addr,
          R_addr,
          LR_const[d],
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          result_add);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, result_add);
    printf("Test %s %s result_add = %d time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_L_const",
            dtype_str[d], result_add, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_R_const(
          Y_addr,
          L_addr,
          LR_const[d],
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          result_add);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, result_add);
    printf("Test %s %s result_add = %d time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_R_const",
            dtype_str[d], result_add, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  // ------------------------------------------------------
  // test MM_NT
  // ------------------------------------------------------
  printf("Test MM_NT:\n");
  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_R_trans(
          Y_addr,
          L_addr,
          R_addr,
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d]);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, false);
    printf("Test %s %s time %lld us, Tops/s = %2.f\n", "tpu_bdc_fp_mm_R_trans",
           dtype_str[d], et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_R_trans(
          Y_addr,
          L_addr,
          R_addr,
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d]);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, false);
    printf("Test %s %s time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_R_trans",
           dtype_str[d], et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }


  // ------------------------------------------------------
  // test MM_TT
  // ------------------------------------------------------
  printf("Test MM_TT:\n");
  result_add = false;
  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_all_trans(
          Y_addr,
          L_addr,
          R_addr,
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          false);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, false);
    printf("Test %s %s time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_all_trans",
           dtype_str[d], et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_L_const_all_trans(
          Y_addr,
          R_addr,
          LR_const[d],
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          false);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, false);
    printf("Test %s %s time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_L_const_all_trans",
           dtype_str[d], et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 1; d < count; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_mm_R_const_all_trans(
          Y_addr,
          L_addr,
          LR_const[d],
          left_rows,
          left_cols,
          right_cols,
          Y_dtypes[d],
          LR_dtypes[d],
          false);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = mm2_ops(left_rows, right_cols, left_cols, 0, 0, 0, false);
    printf("Test %s %s time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm_R_const_all_trans",
           dtype_str[d], et - st, ops * (float)loop / ((float)(et - st) * 1e6));
  }
}
