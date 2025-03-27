#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

typedef void (*cmp_minmax_select_func_t)(
    local_addr_t,
    local_addr_t,
    const variable_t *,
    const variable_t *,
    const variable_t *,
    const variable_t *,
    const dim4 *,
    data_type_t,
    data_type_t);
typedef void (*cmp_select_func_t)(
    local_addr_t,
    const variable_t *,
    const variable_t *,
    const variable_t *,
    const variable_t *,
    const dim4 *,
    data_type_t,
    data_type_t);
static const cmp_minmax_select_func_t cmp_minmax_select_funcs[] = {
    tpu_bdc_maximum_greater_select,
    tpu_bdc_minimum_less_select,
};
static const char *const cmp_minmax_select_func_str[] = {
    "tpu_bdc_maximum_greater_select",
    "tpu_bdc_minimum_less_select",
};
static const cmp_select_func_t cmp_select_funcs[] = {
    tpu_bdc_greater_select,
    tpu_bdc_less_select,
    tpu_bdc_equal_select,
};
static const char *const cmp_select_func_str[] = {
    "tpu_bdc_greater_select",
    "tpu_bdc_less_select",
    "tpu_bdc_equal_select",
};
static const int AB_dtypes[] = {
    DT_INT8,
    DT_INT16,
    DT_INT32,
    DT_FP16,
    DT_BFP16,
    DT_FP32,
    DT_FP8E5M2,
    DT_FP8E4M3};
static const char *const AB_dtypes_str[] = {
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
    "DT_FP16",
    "DT_BFP16",
    "DT_FP32",
    "DT_FP8E5M2",
    "DT_FP8E4M3"};
static const int CD_dtypes[] = {
    // only care dtype width
    DT_INT8,
    DT_INT16,
    DT_INT32,
};

static const char *const CD_dtypes_str[] = {
    "DT_INT8",
    "DT_INT16",
    "DT_INT32"};
static const int CD_dtype_size[] = {1, 2, 4, 0, 0, 0, 0, 0};
static const char *const type_str[] = {
    "TENSOR",
    "SCALAR",
    "VECTOR",
};
void nodechip_fused_cmp_perf_test()
{
  tpu_initialize();

  const int loop = 10;
  const int N = 1;
  const int C = NPU_NUM;
  const int H = 32;
  const int W = 64;
  printf("Test fused cmp:\n");
  printf("loop time = %d per case\n", loop);
  printf("N = %d, C = %d, H = %d, W = %d\n", N, C, H, W);

  dim4 shape = {N, C, H, W};
  float len = (float)N * C * H * W;
  dim4 stride;
  tpu_aligned_stride(&stride, 0, &shape, DT_FP32);

  const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  local_addr_t A_addr = 0;
  local_addr_t B_addr = ALIGN(A_addr + stride.n * sizeof(float), bank_size);
  local_addr_t C_addr = ALIGN(B_addr + stride.n * sizeof(float), bank_size);
  local_addr_t D_addr = ALIGN(C_addr + stride.n * sizeof(float), bank_size);
  local_addr_t R0_addr = ALIGN(D_addr + stride.n * sizeof(float), bank_size);
  local_addr_t R1_addr = ALIGN(R0_addr + stride.n * sizeof(float), bank_size);
  TPUKERNEL_ASSERT(R1_addr + stride.n * sizeof(float) <= (unsigned int)LOCAL_MEM_SIZE);
  u64 st = 0, et = 0;

  variable_t A_var[] = {
      {.type = TENSOR, .context.addr = A_addr},
      {.type = SCALAR, .context.scalar.f32 = 1.0},
      {.type = VECTOR, .context.addr = A_addr},
  };
  variable_t B_var[] = {
      {.type = TENSOR, .context.addr = B_addr},
      {.type = SCALAR, .context.scalar.f32 = 2.0},
      {.type = VECTOR, .context.addr = B_addr},
  };
  variable_t C_var[] = {
      {.type = TENSOR, .context.addr = C_addr},
      {.type = SCALAR, .context.scalar.f32 = 3.0},
  };
  variable_t D_var[] = {
      {.type = TENSOR, .context.addr = D_addr},
      {.type = SCALAR, .context.scalar.f32 = 4.0},
  };
  for (unsigned int f = 0; f < sizeof(cmp_minmax_select_funcs) / sizeof(cmp_minmax_select_funcs[0]); ++f)
  {
    for (int atype_idx = 0; atype_idx < 3; ++atype_idx)
    { // TENSOR, SCALAR, VECTOR
      for (int btype_idx = 0; btype_idx < 3; ++btype_idx)
      { // TENSOR, SCALAR, VECTOR
        for (int ctype_idx = 0; ctype_idx < 2; ++ctype_idx)
        { // TENSOR, SCALAR
          for (int dtype_idx = 0; dtype_idx < 2; ++dtype_idx)
          { // TENSOR, SCALAR
            for (unsigned int d = 1; d < sizeof(AB_dtypes) / sizeof(AB_dtypes[0]); ++d)
            {
              st = firmware_timer_get_time_us();
              for (int i = 0; i < loop; ++i)
              {
                cmp_minmax_select_funcs[f](
                    R0_addr,
                    R1_addr,
                    A_var + atype_idx,
                    B_var + btype_idx,
                    C_var + ctype_idx,
                    D_var + dtype_idx,
                    &shape,
                    AB_dtypes[d],
                    AB_dtypes[d]);
              }
              tpu_poll();
              et = firmware_timer_get_time_us();
              printf("Test %s(AB_dtype = %s, atype: %s, btype: %s, ctype: %s, dtype: %s) time %lld us, Tops:%.6f\n",
                     cmp_minmax_select_func_str[f], AB_dtypes_str[d],
                     type_str[atype_idx], type_str[btype_idx],
                     type_str[ctype_idx], type_str[dtype_idx], et - st,
                     len / ((et - st) / (float)loop * 1e6));
            }
          } // dtype_idx
        }   // ctype_idx
      }     // btype_idx
    }       // atype_idx
  }         // func

  for (unsigned int f = 0; f < sizeof(cmp_select_funcs) / sizeof(cmp_select_funcs[0]); ++f)
  {
    for (int atype_idx = 0; atype_idx < 3; ++atype_idx)
    { // TENSOR, SCALAR, VECTOR
      for (int btype_idx = 0; btype_idx < 3; ++btype_idx)
      { // TENSOR, SCALAR, VECTOR
        for (int ctype_idx = 0; ctype_idx < 2; ++ctype_idx)
        { // TENSOR, SCALAR
          for (int dtype_idx = 0; dtype_idx < 2; ++dtype_idx)
          { // TENSOR, SCALAR
            for (unsigned int d0 = 1; d0 < sizeof(AB_dtypes) / sizeof(AB_dtypes[0]); ++d0)
            {
              int min_d1 = 0;
              int max_d1 = sizeof(CD_dtypes) / sizeof(CD_dtypes[0]) - 1;
              if (AB_dtypes[d0] == DT_INT16 || AB_dtypes[d0] == DT_UINT16)
              {
                min_d1 = 1;
              }
              else if (AB_dtypes[d0] == DT_FP16 || AB_dtypes[d0] == DT_BFP16)
              {
                min_d1 = max_d1 = 1;
              }
              else if (AB_dtypes[d0] == DT_INT32 || AB_dtypes[d0] == DT_UINT32 ||
                       AB_dtypes[d0] == DT_FP32)
              {
                min_d1 = max_d1 = 2;
              }
              for (int d1 = min_d1; d1 <= max_d1; ++d1)
              {
                st = firmware_timer_get_time_us();
                for (int i = 0; i < loop; ++i)
                {
                  cmp_select_funcs[f](
                      R0_addr,
                      A_var + atype_idx,
                      B_var + btype_idx,
                      C_var + ctype_idx,
                      D_var + dtype_idx,
                      &shape,
                      AB_dtypes[d0],
                      CD_dtypes[d1]);
                }
                tpu_poll();
                et = firmware_timer_get_time_us();
                printf("Test %s(AB_dtype = %s, CD_dtype_width = %d, atype: %s, btype: %s, ctype: %s, dtype: %s) time %lld us, Tops:%.6f\n",
                       cmp_select_func_str[f], AB_dtypes_str[d0], CD_dtype_size[d1],
                       type_str[atype_idx], type_str[btype_idx],
                       type_str[ctype_idx], type_str[dtype_idx], et - st,
                       len / ((et - st) / (float)loop * 1e6));
              }
            }
          } // dtype_idx
        }   // ctype_idx
      }     // btype_idx
    }       // atype_idx
  }         // func

  // using large shape, otherwise c906 gen_cmd may block
  printf("sfu test using larger shape\n");
  for (unsigned int d = 0; d < sizeof(AB_dtypes) / sizeof(AB_dtypes[0]); ++d)
  {
    const int dtype_len = tpu_data_type_size(AB_dtypes[d]);
    shape.n = dtype_len == 1 ? 8 : 4;
    len = shape.n * shape.c * shape.h * shape.w;
    A_addr = 0;
    B_addr = ALIGN(A_addr + stride.n * dtype_len, bank_size);
    R0_addr = ALIGN(B_addr + stride.n * dtype_len, bank_size);
    R1_addr = ALIGN(R0_addr + stride.n * dtype_len, bank_size);
    TPUKERNEL_ASSERT(R1_addr + stride.n * dtype_len <= (unsigned int)LOCAL_MEM_SIZE);
    printf("N = %d, C = %d, H = %d, W = %d\n", shape.n, shape.c, shape.h, shape.w);
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      cmp_minmax_select_funcs[0](
          R0_addr,
          R1_addr,
          A_var,
          B_var,
          C_var + 1,
          D_var + 1,
          &shape,
          AB_dtypes[d],
          AB_dtypes[d]);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    printf("Test %s(AB_dtype = %s, atype: %s, btype: %s, ctype: %s, dtype: %s) time %lld us, Tops:%.6f\n",
           cmp_minmax_select_func_str[0], AB_dtypes_str[d],
           type_str[0], type_str[0],
           type_str[1], type_str[1], et - st,
           len / ((et - st) / (float)loop * 1e6));
  }

  for (unsigned int d = 0; d < sizeof(AB_dtypes) / sizeof(AB_dtypes[0]); ++d)
  {
    const int dtype_len = tpu_data_type_size(AB_dtypes[d]);
    shape.n = dtype_len == 1 ? 8 : 4;
    len = shape.n * shape.c * shape.h * shape.w;
    A_addr = 0;
    B_addr = ALIGN(A_addr + stride.n * dtype_len, bank_size);
    R0_addr = ALIGN(B_addr + stride.n * dtype_len, bank_size);
    R1_addr = ALIGN(R0_addr + stride.n * dtype_len, bank_size);
    TPUKERNEL_ASSERT(R1_addr + stride.n * dtype_len <= (unsigned int)LOCAL_MEM_SIZE);
    printf("N = %d, C = %d, H = %d, W = %d\n", shape.n, shape.c, shape.h, shape.w);
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      cmp_select_funcs[0](
          R0_addr,
          A_var,
          B_var,
          C_var + 1,
          D_var + 1,
          &shape,
          AB_dtypes[d],
          AB_dtypes[d]);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    printf("Test %s(AB_dtype = %s, CD_dtype_width = %d, atype: %s, btype: %s, ctype: %s, dtype: %s) time %lld us, Tops:%.6f\n",
           cmp_select_func_str[0], AB_dtypes_str[d], CD_dtype_size[d],
           type_str[0], type_str[0],
           type_str[1], type_str[1], et - st,
           len / ((et - st) / (float)loop * 1e6));
  }

  for (unsigned int d1 = 0; d1 < sizeof(AB_dtypes) / sizeof(AB_dtypes[0]); ++d1)
  {
    const int dtype_len = tpu_data_type_size(AB_dtypes[d1]);
    shape.n = dtype_len == 1 ? 8 : 4;
    len = shape.n * shape.c * shape.h * shape.w;
    A_addr = 0;
    B_addr = ALIGN(A_addr + stride.n * dtype_len, bank_size);
    R0_addr = ALIGN(B_addr + stride.n * dtype_len, bank_size);
    int side = rand() % 2;
    int bin_w = shape.w;
    TPUKERNEL_ASSERT(R1_addr + stride.n * dtype_len <= (unsigned int)LOCAL_MEM_SIZE);
    printf("N = %d, C = %d, H = %d, W = %d\n", shape.n, shape.c, shape.h, shape.w);
    for (int d2 = 0; d2 < 3; ++d2)
    {
      const int dst_dtype_len = tpu_data_type_size((data_type_t)CD_dtypes[d2]);
      if(dtype_len != dst_dtype_len)
          continue;
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; ++i)
      {
        tpu_bdc_srch_bin_select(
            R0_addr,
            A_var,
            B_var,
            &shape,
            side,
            bin_w,
            AB_dtypes[d1],
            CD_dtypes[d2]);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      printf("Test %s(AB_dtype = %s, Res_dtype = %s, atype: %s, btype: %s) time %lld us, Tops:%.6f\n",
             "tpu_bdc_srch_bin_select", AB_dtypes_str[d1], CD_dtypes_str[d2],
             type_str[0], type_str[0], et - st,
             len / ((et - st) / (float)loop * 1e6));
    }
  }
}