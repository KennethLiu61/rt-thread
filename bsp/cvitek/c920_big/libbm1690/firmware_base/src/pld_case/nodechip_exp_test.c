#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

void nodechip_exp_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
  tpu_initialize();

  dim4 shape = {1, 64, 8, 8};
  u32 offset = shape.h * shape.w * sizeof(float);
  local_addr_t src_laddr = 0;
  local_addr_t dst_laddr = offset * 2;
  local_addr_t coeff_laddr = dst_laddr + offset * 2;
  local_addr_t table_laddr = coeff_laddr + offset * 2;
  local_addr_t imm0_laddr = table_laddr + offset * 2;
  local_addr_t imm1_laddr = imm0_laddr + offset * 2;
  TPUKERNEL_ASSERT(imm0_laddr + offset < (u32)tpu_local_mem_size_per_npu());
  int psum_op = ALL_REDUCE_PSUM_WR;
  int op_code = ALL_REDUCE_ADD;

  tpu_gdma_cpy_S2L(src_laddr, input_addr, &shape, NULL, NULL, DT_FP32);
  // tpu_bdc_load_fp32_exp_coeff(coeff_laddr);
  tpu_gdma_system_bcast(coeff_laddr | LOCAL_MEM_START_ADDR, STATIC_MEM_START_ADDR, 10, NPU_NUM, DT_FP32);
  //Useless. Exp does not need table now
  tpu_bdc_load_fp32_exp_table(table_laddr);
  tpu_bdc_fp32_exp(
      dst_laddr,
      src_laddr,
      imm0_laddr,
      imm1_laddr,
      coeff_laddr,
      table_laddr,
      &shape);
  // tpu_gdma_cpy_L2S(output_addr, dst_laddr, &shape, NULL, NULL, DT_FP32);
  // uese ARE
  tpu_gdma_cpy_L2S(tpu_l2_sram_get_start_addr(), dst_laddr, &shape, NULL, NULL, DT_FP32);
  tpu_gdma_cpy_reduce_L12L2(tpu_l2_sram_get_start_addr(), dst_laddr, &shape, NULL, NULL, DT_FP32, psum_op, op_code);
  tpu_sdma_cpy_S2S(output_addr, tpu_l2_sram_get_start_addr(), &shape, NULL, NULL, DT_FP32);

  tpu_poll();
}