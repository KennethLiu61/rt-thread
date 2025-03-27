#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_dma_filter_nonzero(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
  tpu_initialize();

  // layout: input, mask
  // gdma fiter number, gdma nonzero number, filterd_out, nonzero out
  // shape = {2, 10, 8, 8}, DT_FP32
  data_type_t dtype = DT_FP32;
  data_type_t mask_dtype = DT_INT32;
  dim4 shape = {2, 10, 8, 8};
  uint32_t count = shape.n * shape.c * shape.h * shape.w;
  system_addr_t filterd_addr = output_addr + 2 * sizeof(float);
  system_addr_t mask_addr = input_addr + count * tpu_data_type_size(dtype);
  tpu_sdma_system_cpy(filterd_addr, input_addr, count * 2, dtype);

  tpu_sync_core();

  // gdma
  uint32_t gdma_filter_num = tpu_gdma_mask_select_S2S_with_ret(filterd_addr, input_addr, mask_addr, false, &shape, dtype, mask_dtype);
  system_addr_t nonzero_addr = filterd_addr + gdma_filter_num * tpu_data_type_size(dtype);
  tpu_gdma_nonzero_S2S(nonzero_addr, mask_addr, &shape, mask_dtype, 5);
  uint32_t gdma_nonzero_num = tpu_gdma_get_filter_num();

  uint32_t *output_ptr = tpu_global_mem_addr(output_addr);
  output_ptr[0] = gdma_filter_num;
  output_ptr[1] = gdma_nonzero_num;
  tpu_flush_cache(output_addr, tpu_cache_line_size());

  tpu_poll();
}
