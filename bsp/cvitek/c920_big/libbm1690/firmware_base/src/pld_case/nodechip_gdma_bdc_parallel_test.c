#include "nodechip_pld_test.h"
#include <stdlib.h>
#include "tpu_kernel.h"

void nodechip_gdma_bdc_parallel_test(
    unsigned long long input_global_addr,
    unsigned long long output_global_addr
) {
  tpu_initialize();
  const int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();

  data_type_t dtype = DT_FP32;
//  dim4 shape = {1, 64, 20, 20};
//  dim4 stride, global_stride;
//  tpu_aligned_stride(&stride, 0, &shape, dtype);
//  tpu_continuous_stride(&global_stride, &shape);
//
//  local_addr_t A_addr = 0;
//  local_addr_t B_addr = bank_size;
//  local_addr_t C_addr = 2 * bank_size;
//  scalar_t val = {.f32 = 12.21};
// 
//  // simple but wrong result
//  tpu_parallel_start();
//  tpu_gdma_cpy_S2L(
//      A_addr,
//      input_global_addr,
//      &shape,
//      &stride,
//      &global_stride,
//      dtype);
//  tpu_bdc_fp_add_C(
//      C_addr,
//      A_addr,
//      val,
//      &shape,
//      NULL,
//      NULL,
//      dtype);
//  
//  tpu_gdma_cpy_L2S(
//      output_global_addr,
//      C_addr,
//      &shape,
//      &global_stride,
//      &stride,
//      dtype);
//
//  tpu_parallel_end();

  // split two parts to compute  
  dim4 Ashape = {1, 64, 20, 20};
  dim4 shape = {1, 64, 10, 20};
  dim4 stride, global_stride;
  tpu_continuous_stride(&global_stride, &Ashape);
  tpu_aligned_stride(&stride, 0, &shape, dtype);

  local_addr_t A_addr[2] = {0, bank_size};
  local_addr_t C_addr[2] = {2 * bank_size, 3 * bank_size};
  scalar_t val = {.f32 = 12.21};

  int offset = 10*20*sizeof(float);
  for (int i = 0; i < 3; ++i) {
    // load
    if (i < 2) {
      tpu_gdma_cpy_S2L(
          A_addr[i % 2],
          input_global_addr + offset*i,
          &shape,
          &stride,
          &global_stride,
          dtype);
    }

    if (tpu_is_parallel_state()) {
      tpu_parallel_end();
    }
    tpu_parallel_start();

    // compute
    if (i < 2) {
      tpu_bdc_fp_add_C(
          C_addr[i % 2],
          A_addr[i % 2],
          val,
          &shape,
          NULL,
          NULL,
          dtype);
    }

    if (i > 0) {
      tpu_gdma_cpy_L2S(
          output_global_addr + offset*(i-1),
          C_addr[(i-1) % 2],
          &shape,
          &global_stride,
          &stride,
          dtype);
    }
  }

  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }

  tpu_poll();
}
