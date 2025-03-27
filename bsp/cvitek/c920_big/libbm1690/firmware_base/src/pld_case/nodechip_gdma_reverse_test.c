#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#include "split_util.h"

void nodechip_gdma_reverse_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{

  tpu_initialize();
  local_addr_t input_local_addr = 0;
  local_addr_t output_local_addr = LOCAL_MEM_SIZE / 2;
  data_type_t dtype = DT_FP32;
  dim4 input_shape = {.n = 3, .c = 64, .h = 8, .w = 10};
  dim4 output_shape = {.n = 3, .c = 64, .h = 8, .w = 10};
  u64 output_offset = output_shape.n * output_shape.c * output_shape.h * output_shape.w * tpu_data_type_size(dtype);
  u64 output_local_offset = output_shape.n * DIV_UP(output_shape.c, NPU_NUM) * ALIGN(output_shape.h * output_shape.w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);

  // test src mem is GMEM, support reverse n, c, h, w
  // test S2S
  // 1. reverse n
  tpu_gdma_reverse_S2S(
      output_addr,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      0,
      dtype);
  // 2. reverse c
  tpu_gdma_reverse_S2S(
      output_addr + output_offset,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      1,
      dtype);
  // 3. reverse h
  tpu_gdma_reverse_S2S(
      output_addr + output_offset * 2,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      2,
      dtype);
  // 4. reverse w
  tpu_gdma_reverse_S2S(
      output_addr + output_offset * 3,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      3,
      dtype);

  // test S2L
  // 1. reverse n
  tpu_gdma_reverse_S2L(
      output_local_addr,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      0,
      dtype);
  // 2. reverse c
  tpu_gdma_reverse_S2L(
      output_local_addr + output_local_offset,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      1,
      dtype);
  // 3. reverse h
  tpu_gdma_reverse_S2L(
      output_local_addr + output_local_offset * 2,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      2,
      dtype);
  // 4. reverse w
  tpu_gdma_reverse_S2L(
      output_local_addr + output_local_offset * 3,
      input_addr,
      &input_shape,
      NULL,
      NULL,
      3,
      dtype);
  dim4 output_local_shape_4 = {.n = 4*3, .c = 64, .h = 8, .w = 10};
  tpu_gdma_cpy_L2S(output_addr + output_offset * 4,
                   output_local_addr,
                   &output_local_shape_4,
                   NULL,
                   NULL,
                   dtype);

  // test src mem is LMEM, only support reverse c
  tpu_gdma_cpy_S2L(input_local_addr,
                   input_addr,
                   &input_shape,
                   NULL,
                   NULL,
                   dtype);
  // test L2S
  tpu_gdma_reverse_L2S(
      output_addr + output_offset * 8,
      input_local_addr,
      &output_shape,
      NULL,
      NULL,
      1,
      dtype);
  // test L2L
  tpu_gdma_reverse_L2L(
      output_local_addr + output_local_offset * 4,
      input_local_addr,
      &output_shape,
      NULL,
      NULL,
      1,
      dtype);
  tpu_gdma_cpy_L2S(output_addr + output_offset * 9,
                   output_local_addr + output_local_offset * 4,
                   &output_shape,
                   NULL,
                   NULL,
                   dtype);
  tpu_poll();
}
