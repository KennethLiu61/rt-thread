#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "memmap.h"
#include "tpu_kernel.h"

void nodechip_seg_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
  tpu_initialize();

  const int len = 32;

  tpu_bdc_npu_bcast_from_static(
      0, SERIAL_NUMBER_OFFSET,
      32, len, DT_INT32);

  tpu_bdc_npu_distribute_from_static(
      tpu_local_mem_size_per_npu() / 2,
      SERIAL_NUMBER_OFFSET,
      32, DT_INT32);

  dim4 shape = {1, 32, 1, len};
  tpu_gdma_compact_L2S(output_addr, 0, &shape, DT_INT32);

  shape.w = 1;
  tpu_gdma_cpy_L2S(
      output_addr + 32 * len * sizeof(int),
      tpu_local_mem_size_per_npu() / 2,
      &shape, NULL, NULL, DT_INT32);
  tpu_poll();
}
