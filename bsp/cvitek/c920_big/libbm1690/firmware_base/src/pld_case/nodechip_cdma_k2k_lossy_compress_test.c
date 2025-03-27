#include "nodechip_cdma_pld_test_utils.h"

void nodechip_cdma_k2k_lossy_compress_test(unsigned char *api_buf) {
  CORE_PRINT("=================================\n");
  sg_api_pld_cdma_k2k_t *api = (sg_api_pld_cdma_k2k_t *)api_buf;
  CORE_PRINT("%s: input_addr=0x%llx, output_addr=0x%llx, reduce=%s\n", __func__,
             api->input_addr, api->output_addr,
             _reduce_type_name[api->reduce_op]);
  u64 l2mem_index = 0;
  CDMA_OPCODE reduce_type = (CDMA_OPCODE)(api->reduce_op);
  global_addr_t input = (global_addr_t)api->input_addr;
  global_addr_t output = (global_addr_t)api->output_addr;
  global_addr_t reduce = api->input_addr + CNT * sizeof(float);
  global_addr_t l2_reduce =
      tpu_l2_sram_get_start_addr() + l2mem_index * 0x800000;
  CORE_PRINT("l2_reduce: 0x%llx, l2mem_index: %lld\n", l2_reduce, l2mem_index);
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  float *input_p = (float *)tpu_global_mem_addr(input);
  float *reduce_input_p = (float *)tpu_global_mem_addr(reduce);
  for (int i = 0; i < W; i++) {
    CORE_PRINT("input[%d]=%.2f, reduce_input[%d]=%.2f\n", i, input_p[i], i,
               reduce_input_p[i]);
  }
#endif
  tpu_initialize();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("set l2 ori data by gdma\n");
  dim4 d_shape = {.n = N, .c = C, .h = H, .w = W};
  // tpu_gdma_cpy_S2S(l2_reduce, reduce, &d_shape, NULL, NULL, DT_FP32);
  tpu_gdma_lossy_compress_S2S(l2_reduce, reduce, &d_shape, NULL, NULL);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  // fp20 *l2_output_p = (fp20 *)tpu_l2_sram_addr(l2_reduce);
  // print_fp20(l2_output_p, N*C*H, W);
  uint32_t *l2_output_p = (uint32_t *)tpu_l2_sram_addr(l2_reduce);
  for (int i = 0; i < 20; i++) {
    CORE_PRINT("%u, ", l2_output_p[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
#endif
#if defined(USING_CMODEL) && !defined(CDMA_DES_PLD_TEST)
  CORE_PRINT("=================================\n");
  CORE_PRINT("cmodel reduce ori data to l2 by gdma\n");
  tpu_gdma_lossy_compress_reduce_S2L2(l2_reduce, input, &d_shape, NULL, NULL,
                                      ALL_REDUCE_PSUM_WR, (int)reduce_type);
  tpu_poll();
#else
  int dst_chipid = 0;
  int src_chipid = 0;
  CORE_PRINT("=================================\n");
  CORE_PRINT("reduce ori data to l2 by cdma\n");
  // A. switch config for PIO K2K
  // write32(0x6c00791000,0) set pio mode
  u32 write_reg = READ_REG(CDMA_ENGINE_MAIN_CTRL(PLD_K2K_CDMA_TEST_PORT));
  write_reg = write_reg & 0xfffffffe;
  WRITE_REG((CDMA_ENGINE_MAIN_CTRL(PLD_K2K_CDMA_TEST_PORT)), write_reg,
            NODECHIP_REG);
  atomic_cdma_route_configuration(dst_chipid, CDMA_ROUTE_AXI_RN, -1);

  // B. des mode config(jump this step since it's pio mode)
  // C. config cdma instruction
  atomic_cdma_config_tcredit_for_pld_test(dst_chipid, l2_reduce, N, C, H, W,
                                          N_STRIDE, C_STRIDE, H_STRIDE,
                                          reduce_type, -1);
  tpu_cdma_lossy_compress(dst_chipid, src_chipid, input, N, C, H, W, N_STRIDE,
                          C_STRIDE, H_STRIDE, reduce_type);

  // D. sys end config for des mode(jump this step since it's pio mode)
  // E. pio run cmd config
  atomic_cdma_pio_interrupt_poll(dst_chipid);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  // print_fp20(l2_output_p, N*C*H, W);
  for (int i = 0; i < 20; i++) {
    CORE_PRINT("%u, ", l2_output_p[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("move l2 reduced data to ddr outout by gdma\n");
  tpu_gdma_lossy_decompress_S2S(output, l2_reduce, &d_shape, NULL, NULL);
  // tpu_gdma_cpy_S2S(output, tpu_l2_sram_get_start_addr(), &d_shape, NULL,
  //                  NULL, dtype);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  tpu_invalidate_cache(GLOBAL_MEM_START_ADDR + output, 0x1000);
  float *output_p = (float *)tpu_global_mem_addr(output);
  for (int i = 0; i < W; i++) {
    CORE_PRINT("output[%d]=%f\n", i, output_p[i]);
  }
  uint32_t *print_output = (uint32_t *)output_p;
  for (int i = 0; i < 32; i++) {
    CORE_PRINT("%u, ", print_output[i]);
    if (i % 8 == 7)
      CORE_PRINT("\n");
  }
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: all test finised\n", __func__);
  CORE_PRINT("=================================\n");
}