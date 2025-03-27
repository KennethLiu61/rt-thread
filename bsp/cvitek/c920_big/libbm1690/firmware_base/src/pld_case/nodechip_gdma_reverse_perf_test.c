#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#include "split_util.h"

static const int loop = 5;

#define BEGIN(loop)                                \
  {                                                \
    u64 start_time = firmware_timer_get_time_us(); \
    int __loop = (loop);                           \
    for (int __i = 0; __i < __loop; __i++)         \
    {

#define END(info, slice, dtype)                                                    \
  }                                                                                \
  tpu_poll();                                                                      \
  u64 end_time = firmware_timer_get_time_us();                                     \
  dim4 *pshape = (slice);                                                          \
  u64 bytes = pshape->n * pshape->c * pshape->w * pshape->h * tpu_data_type_size(dtype); \
  float avg_time = 1.0 * (end_time - start_time) / __loop;                         \
  float bw = ((float)__loop * bytes * 1e6 / (end_time - start_time)) / (1024 * 1024 * 1024.f); \
  printf("Total %s time: %lldus, loop:%d, bytes=%lld, avg_time:%gus, bw=%gGB/s\n", \
         info, (end_time - start_time), __loop, bytes, avg_time, bw);              \
  }

#define PRINT_GADDR(addr) printf(#addr "=0x%llx, ", addr)
#define PRINT_LADDR(addr) printf(#addr "=0x%x, ", addr)
#define PRINT_DIM4(d) if(d) printf("(%d %d %d %d), ", d->n, d->c, d->h, d->w); else printf(#d "=NULL, ")
#define PRINT_INT(d) printf(#d "=%d, ", (int)d)


static void __nodechip_local_reverse_test(
  global_addr_t input_gaddr,
  global_addr_t output_gaddr,
  global_addr_t reversed_gaddr,
  int loffset,
  dim4* shape,
  int axis,
  data_type_t dtype
) {
  local_addr_t reverse0_laddr = loffset;
  local_addr_t reverse1_laddr = LOCAL_MEM_SIZE/2+loffset;
  dim4* reverse0_stride = NULL;
  dim4* reverse1_stride = NULL;
  dim4* S2S_stride = NULL;

  PRINT_GADDR(input_gaddr);
  PRINT_GADDR(output_gaddr);
  PRINT_GADDR(reversed_gaddr);
  PRINT_LADDR(reverse0_laddr);
  PRINT_LADDR(reverse1_laddr);
  PRINT_DIM4(shape);
  PRINT_DIM4(reverse0_stride);
  PRINT_DIM4(reverse1_stride);
  PRINT_DIM4(S2S_stride);
  PRINT_INT(axis);
  PRINT_INT(dtype);

  // 1. reverse n, c, h from system memory to local memory
  PRINT_GADDR(input_gaddr);
  PRINT_LADDR(reverse0_laddr);
  PRINT_DIM4(shape);
  PRINT_INT(axis);
  PRINT_INT(dtype);
  BEGIN(loop)
  tpu_gdma_reverse_S2L(
    reverse0_laddr,
    input_gaddr,
    shape,
    reverse0_stride,
    NULL,
    axis, dtype);
  END("Reverse S2L", shape, dtype)

  // 2. reverse  c L2L
  PRINT_LADDR(reverse0_laddr);
  PRINT_LADDR(reverse1_laddr);
  PRINT_DIM4(shape);
  PRINT_INT(dtype);
  BEGIN(loop)
  tpu_gdma_reverse_L2L(reverse1_laddr, reverse0_laddr, shape, reverse1_stride, reverse0_stride, 1, dtype);
  END("Reverse C L2L", shape, dtype)

  // 3. reverse  c L2S 
  PRINT_LADDR(reverse1_laddr);
  PRINT_GADDR(reversed_gaddr);
  PRINT_DIM4(shape);
  PRINT_INT(dtype);
  BEGIN(loop)
  tpu_gdma_reverse_L2S(reversed_gaddr, reverse1_laddr, shape, S2S_stride, reverse1_stride, 1, dtype);
  END("Reverse C L2S", shape, dtype)

  // 4. reverse S2S again to output
  PRINT_GADDR(reversed_gaddr);
  PRINT_GADDR(output_gaddr);
  PRINT_DIM4(shape);
  PRINT_INT(axis);
  PRINT_INT(dtype);
  BEGIN(loop)
  tpu_gdma_reverse_S2S(output_gaddr, reversed_gaddr, shape, NULL, S2S_stride, axis, dtype);
  END("Reverse S2S", shape, dtype)
}

void nodechip_gdma_reverse_perf_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{

  tpu_initialize();
  data_type_t dtypes[] = {DT_INT8, DT_INT16, DT_INT32};
  dim4 base_shape = {4, NPU_NUM, 16, 16};
  int hw_scales[] = {4};
  int n_scales[] = {4};
  dim4 shape = base_shape;
  for (size_t dtype_idx = 0; dtype_idx < sizeof(dtypes) / sizeof(dtypes[0]); dtype_idx++)
  {
    data_type_t dtype = dtypes[dtype_idx];
    for (size_t ni = 0; ni < sizeof(n_scales) / sizeof(n_scales[0]); ni++)
    {
      shape.n = base_shape.n * n_scales[ni];
      for (int cscale = 1; cscale < 2; cscale++)
      {
        shape.c = base_shape.c * cscale;
        for (size_t hi = 0; hi < sizeof(hw_scales) / sizeof(hw_scales[0]); hi++)
        {
          shape.h = base_shape.h * hw_scales[hi];
          // shape.w = base_shape.w * hw_scales[hi];
          for (int offset = 0; offset < 1; offset += tpu_data_type_size(dtype))
          {
            if (offset + shape.n * cscale * shape.h * shape.w > LOCAL_MEM_SIZE / 2)
              continue;
            for (int axis = 0; axis < 4; axis++)
            {
              __nodechip_local_reverse_test(
                  input_addr,
                  output_addr,
                  input_addr + offset,
                  offset,
                  &shape,
                  axis,
                  dtype);
            }
          }
        }
      }
    }
  }
}