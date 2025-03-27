#include "nodechip_pld_test.h"
#include "atomic_sys_gen_cmd.h"
#include "tpu_kernel.h"

void nodechip_tpu_sys_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
  tpu_initialize();
  int npu_num = tpu_npu_num();
  data_type_t dtype = DT_FP32;

  int lmem_size = tpu_local_mem_size_per_npu();
  local_addr_t l_src_addr = 0;
  local_addr_t l_dst_addr0 = lmem_size / 2;
  local_addr_t l_dst_addr1 = lmem_size * 3 / 4;
  local_addr_t l_mask_addr = lmem_size / 2 - 128;

  // init all local memory as 0.f
  scalar_t const_val = {.f32 = 0.f};
  dim4 shape = {1, npu_num, lmem_size / 16 / 4, 16};
  dim4 stride = {1, 1, 16, 1};
  tpu_bdc_set_C(0, const_val, &shape, &stride, DT_FP32);

  shape.n = 1; shape.c = npu_num; shape.h = 16; shape.w = 16;
  dim4 l_stride = {0};
  tpu_aligned_stride(&l_stride, 0, &shape, dtype);
  dim4 g_stride = {0};
  tpu_continuous_stride(&g_stride, &shape);
  tpu_gdma_cpy_S2L(l_src_addr, input_addr, &shape, &l_stride, &g_stride, dtype);

  // set lane mask
  BD_SYS_TYPE sys_type = BD_SYS_SWR;
  u64 imm = 0;
  for (int i = 0; i < npu_num; i++) imm |= ((i % 2 ? 1ull : 0ull) << i);
  int short_valid = 1;
  int long_valid = 0;
  int src_is_const = (sys_type == BD_SYS_SWR ? 1 : 0);

  CMD_ID_NODE pid_node;
  tpu_get_id_node(&pid_node);
  // short valid test
  atomic_bd_swr_gen_cmd(l_mask_addr, src_is_const, imm,
                         short_valid, long_valid, sys_type, MASTER_THREAD,
                         &pid_node);
  tpu_set_id_node(&pid_node);

  scalar_t scalar = {.f32 = 1.f};
  tpu_bdc_fp_add_C(l_dst_addr0, l_src_addr, scalar, &shape, &l_stride, &l_stride, dtype);

  scalar.f32 = -1.f;
  tpu_bdc_fp_add_C(l_dst_addr1, l_dst_addr0, scalar, &shape, &l_stride, &l_stride, dtype);

  tpu_get_id_node(&pid_node);
  // long term valid test
  long_valid = 1; short_valid = 0;
  atomic_bd_swr_gen_cmd(l_mask_addr, src_is_const, imm,
                         short_valid, long_valid, sys_type, MASTER_THREAD,
                         &pid_node);
  tpu_set_id_node(&pid_node);

  scalar.f32 = 1.f;
  tpu_bdc_fp_add_C(l_dst_addr0, l_dst_addr1, scalar, &shape, &l_stride, &l_stride, dtype);

  scalar.f32 = -1.f;
  tpu_bdc_fp_add_C(l_dst_addr1, l_dst_addr0, scalar, &shape, &l_stride, &l_stride, dtype);

  sys_type = BD_SYS_SWR_FROM_LMEM;
  dim4 mask_shape = {1, 1, 1, npu_num};
  tpu_gdma_cpy_S2L(l_mask_addr,
                   input_addr + shape.n * g_stride.n * tpu_data_type_size(dtype),
                   &mask_shape, NULL, NULL,
                   INT8);

  tpu_get_id_node(&pid_node);

  long_valid = 1; short_valid = 0;
  atomic_bd_swr_gen_cmd(l_mask_addr, src_is_const, imm,
                        short_valid, long_valid, sys_type, MASTER_THREAD,
                        &pid_node);
  tpu_set_id_node(&pid_node);

  tpu_bdc_fp_add(l_dst_addr0, l_src_addr, l_dst_addr1, &shape, &l_stride, &l_stride, &l_stride, dtype);

  scalar.f32 = -0.8;
  tpu_bdc_fp_add_C(l_dst_addr1, l_dst_addr0, scalar, &shape, &l_stride, &l_stride, dtype);

  sys_type = BD_SYS_SWR_COL_FROM_LMEM;
  dim4 mask_shape_2 = {1, npu_num, 1, 1};
  tpu_gdma_cpy_S2L(l_mask_addr,
                   input_addr + shape.n * g_stride.n * tpu_data_type_size(dtype),
                   &mask_shape_2, NULL, NULL,
                   INT8);

  tpu_get_id_node(&pid_node);
  long_valid = 0; short_valid = 1;
  atomic_bd_swr_gen_cmd(l_mask_addr, src_is_const, imm,
                        short_valid, long_valid, sys_type, MASTER_THREAD,
                        &pid_node);
  tpu_set_id_node(&pid_node);

  tpu_bdc_fp_add(l_dst_addr0, l_src_addr, l_dst_addr1, &shape, &l_stride, &l_stride, &l_stride, dtype);

  scalar.f32 = -3.4;
  tpu_bdc_fp_add_C(l_dst_addr1, l_dst_addr0, scalar, &shape, &l_stride, &l_stride, dtype);

  tpu_gdma_cpy_L2S(output_addr, l_dst_addr1, &shape, &g_stride, &l_stride, dtype);

  tpu_get_id_node(&pid_node);
  // clear valid bit
  atomic_bd_swr_gen_cmd(l_mask_addr, src_is_const, imm,
                         0, 0, BD_SYS_SWR, MASTER_THREAD,
                         &pid_node);
  tpu_set_id_node(&pid_node);

  tpu_poll();
}
