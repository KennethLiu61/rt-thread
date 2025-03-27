#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

static const scalar_t L_const_fp32 = {.f32 = 1.1};
static const scalar_t L_const_int = {.s8 = 11};
static const scalar_t shift_const = {.s8 = 3};
static const rounding_mode_t round_mode = RM_UP;
static const data_type_t dtypes[] = {DT_INT8, DT_INT16, DT_INT32};
static const char * const dtypes_str[] = {"DT_INT8", "DT_INT16", "DT_INT32"};
static const int cols_per_channel[] = {1, 16};

u64 mm_ops(
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

void nodechip_mm_perf_test() {
  tpu_initialize();

  const int loop = 10;
  const int left_rows = 32;
  const int left_cols = 64;
  const int right_cols = 64 * 16 * 2;
  printf("Test MM:\n");
  printf("loop time = %d per case\n", loop);
  printf("left_rows = %d, left_cols = %d, right_cols = %d\n",
         left_rows, left_cols, right_cols);
  printf("shift_bits = %d, round_mode = RM_UP\n", shift_const.s8);

  u64 st = 0, et = 0, ops = 0;
  int left_cols_per_channel = 1;
  int right_cols_per_channel = 1;
  for (int lc = 0; lc < 2; ++lc) {
    for (int rc = 1; rc < 2; ++rc) {
      left_cols_per_channel = cols_per_channel[lc];
      right_cols_per_channel = cols_per_channel[rc];
      printf("\nleft_cols_per_channel = %d, rightt_cols_per_channel = %d\n",
             left_cols_per_channel, right_cols_per_channel);

      int left_c = DIV_UP(left_cols, left_cols_per_channel);
      int right_c = DIV_UP(right_cols, right_cols_per_channel);

      dim4 Y_shape = {1, left_rows, 1, right_cols};
      dim4 L_shape = {left_rows, left_c, 1, left_cols_per_channel};
      dim4 R_shape = {left_cols, right_c, 1, right_cols_per_channel};
      dim4 B_shape = {1, right_c, 1, right_cols};
      dim4 Y_stride, L_stride, R_stride, B_stride;
      tpu_aligned_stride(&Y_stride, 0, &Y_shape, DT_FP32);
      tpu_aligned_stride(&L_stride, 0, &L_shape, DT_FP32);
      tpu_aligned_stride(&R_stride, 0, &R_shape, DT_FP32);
      tpu_aligned_stride(&B_stride, 0, &B_shape, DT_FP32);

      const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
      local_addr_t Y_addr = 0;
      local_addr_t L_addr = ALIGN(Y_addr + Y_stride.n * sizeof(float), bank_size);
      local_addr_t R_addr = ALIGN(L_addr + L_stride.n * sizeof(float), bank_size);
      local_addr_t B_addr = ALIGN(R_addr + R_stride.n * sizeof(float), bank_size);
      local_addr_t S_addr = B_addr; // shift
      TPUKERNEL_ASSERT(B_addr + B_stride.n * sizeof(float) <= (unsigned int)LOCAL_MEM_SIZE);

      // ------------------------------------------------------
      // test MM_NORMAL
      // ------------------------------------------------------
      printf("----Test MM_NORMAL:\n");
      for (int has_bias = 0; has_bias < 2; ++has_bias) {
        for (int result_add = 0; result_add< 2; ++result_add) {
          st = firmware_timer_get_time_us();
          for (int i = 0; i < loop; ++i) {
            tpu_bdc_fp32_mm(
                Y_addr,
                L_addr,
                R_addr,
                B_addr,
                left_rows,
                left_cols,
                right_cols,
                left_cols_per_channel,
                right_cols_per_channel,
                has_bias,
                result_add);
          }
          tpu_poll();
          et = firmware_timer_get_time_us();
          ops = mm_ops(left_rows, right_cols, left_cols, has_bias, 0, 0, result_add);
          printf("Test %s has_bias = %d result_add = %d time %lld us, Tops/s = %.2f\n", "tpu_bdc_fp_mm",
                  has_bias, result_add, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
        }
      }

      for (int has_bias = 0; has_bias < 2; ++has_bias) {
        for (int result_add = 0; result_add< 2; ++result_add) {
          st = firmware_timer_get_time_us();
          for (int i = 0; i < loop; ++i) {
            tpu_bdc_fp32_mm_L_trans(
                Y_addr,
                L_addr,
                R_addr,
                B_addr,
                left_rows,
                left_cols,
                right_cols,
                left_cols_per_channel,
                right_cols_per_channel,
                has_bias,
                result_add);
          }
          tpu_poll();
          et = firmware_timer_get_time_us();
          printf("Test %s(has_bias = %d result_add = %d), cost time %lld us\n",
                 "tpu_bdc_fp32_mm_L_trans", has_bias, result_add, et-st);
        }
      }

      for (int has_bias = 0; has_bias < 2; ++has_bias) {
        for (int result_add = 0; result_add< 2; ++result_add) {
          st = firmware_timer_get_time_us();
          for (int i = 0; i < loop; ++i) {
            tpu_bdc_fp32_mm_L_const(
                Y_addr,
                R_addr,
                B_addr,
                L_const_fp32.f32,
                left_rows,
                left_cols,
                right_cols,
                right_cols_per_channel,
                has_bias,
                result_add);
          }
          tpu_poll();
          et = firmware_timer_get_time_us();
          printf("Test %s(has_bias = %d result_add = %d), cost time %lld us\n",
                 "tpu_bdc_fp32_mm_L_const", has_bias, result_add, et-st);
        }
      }

      for (int has_bias = 0; has_bias < 2; ++has_bias) {
        for (int result_add = 0; result_add< 2; ++result_add) {
          st = firmware_timer_get_time_us();
          for (int i = 0; i < loop; ++i) {
            tpu_bdc_fp32_mm_L_const(
                Y_addr,
                R_addr,
                B_addr,
                L_const_fp32.f32,
                left_rows,
                left_cols,
                right_cols,
                right_cols_per_channel,
                has_bias,
                result_add);
          }
          tpu_poll();
          et = firmware_timer_get_time_us();
          printf("Test %s(has_bias = %d result_add = %d), cost time %lld us\n",
                 "tpu_bdc_fp32_mm_L_const", has_bias, result_add, et-st);
        }
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_mm(
              Y_addr,
              L_addr,
              R_addr,
              left_rows,
              left_cols,
              right_cols,
              left_cols_per_channel,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              shift_const.s8,
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_mm", dtypes_str[d], et-st);
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_mm_L_trans(
              Y_addr,
              L_addr,
              R_addr,
              left_rows,
              left_cols,
              right_cols,
              left_cols_per_channel,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              shift_const.s8,
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_mm_L_trans", dtypes_str[d], et-st);
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_mm_L_const(
              Y_addr,
              R_addr,
              L_const_int,
              left_rows,
              left_cols,
              right_cols,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              shift_const.s8,
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_mm_L_const", dtypes_str[d], et-st);
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_pcs_mm(
              Y_addr,
              L_addr,
              R_addr,
              S_addr,
              left_rows,
              left_cols,
              right_cols,
              left_cols_per_channel,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_pcs_mm", dtypes_str[d], et-st);
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_pcs_mm_L_trans(
              Y_addr,
              L_addr,
              R_addr,
              S_addr,
              left_rows,
              left_cols,
              right_cols,
              left_cols_per_channel,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_pcs_mm_L_trans", dtypes_str[d], et-st);
      }

      for (int d = 0; d < 3; ++d) {
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; ++i) {
          tpu_bdc_int_pcs_mm_L_const(
              Y_addr,
              R_addr,
              S_addr,
              L_const_int,
              left_rows,
              left_cols,
              right_cols,
              right_cols_per_channel,
              dtypes[d],
              dtypes[d],
              round_mode);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("Test %s(LR_dtype = %s), cost time %lld us\n",
               "tpu_bdc_int_pcs_mm_L_const", dtypes_str[d], et-st);
      }
    } // right_cols_per_channel
  } // left_cols_per_channel
}