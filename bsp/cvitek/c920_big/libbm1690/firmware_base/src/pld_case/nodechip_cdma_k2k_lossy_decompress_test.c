#include "nodechip_cdma_pld_test_utils.h"
void nodechip_cdma_k2k_lossy_decompress(global_addr_t input,
                                        global_addr_t reduce,
                                        global_addr_t l2_reduce,
                                        global_addr_t l2_reduce_buffer,
                                        global_addr_t output,
                                        CDMA_OPCODE reduce_type) {
  tpu_initialize();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("set l2 ori data by gdma\n");
  dim4 d_shape = {.n = N, .c = C, .h = H, .w = W};
  tpu_gdma_lossy_compress_S2S(l2_reduce, input, &d_shape, NULL, NULL);
  tpu_gdma_cpy_S2S(l2_reduce_buffer, reduce, &d_shape, NULL, NULL, DT_FP32);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  float *l2_reduce_buffer_p = (float *)tpu_l2_sram_addr(l2_reduce_buffer);
  for (int i = 0; i < W; i++) {
    CORE_PRINT("l2 reduce value[%d]=%f\n", i, l2_reduce_buffer_p[i]);
  }
#endif
#if defined(USING_CMODEL) && !defined(CDMA_DES_PLD_TEST)
  CORE_PRINT("=================================\n");
  CORE_PRINT("cmodel reduce l2 data decompress to ddr by gdma\n");
  tpu_gdma_lossy_decompress_reduce_S2L2(l2_reduce_buffer, l2_reduce, &d_shape,
                                        NULL, NULL, ALL_REDUCE_PSUM_WR,
                                        (int)reduce_type);
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
  atomic_cdma_config_tcredit_for_pld_test(dst_chipid, l2_reduce_buffer, N, C, H,
                                          W, N_STRIDE, C_STRIDE, H_STRIDE,
                                          reduce_type, -1);
  tpu_cdma_lossy_decompress(dst_chipid, src_chipid, l2_reduce, N, C, H, W,
                            N_STRIDE, C_STRIDE, H_STRIDE, reduce_type);

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
  for (int i = 0; i < W; i++) {
    CORE_PRINT("reduce output[%d]=%f\n", i, l2_reduce_buffer_p[i]);
  }
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("move l2 reduced data to ddr outout by gdma\n");
  tpu_gdma_cpy_S2S(output, l2_reduce_buffer, &d_shape, NULL, NULL, DT_FP32);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  tpu_invalidate_cache(GLOBAL_MEM_START_ADDR + output, 0x1000);
  float *output_p = (float *)tpu_global_mem_addr(output);
  for (int i = 0; i < W; i++) {
    CORE_PRINT("ddr output[%d]=%f\n", i, output_p[i]);
  }
  uint32_t *print_output_u32 = (uint32_t *)output_p;
  for (int i = 0; i < W; i++) {
    CORE_PRINT("%u, ", print_output_u32[i]);
    if (i % 8 == 7)
      CORE_PRINT("\n");
  }
#endif
}
void nodechip_cdma_k2k_lossy_decompress_test(unsigned char *api_buf) {
  CORE_PRINT("=================================\n");
  sg_api_pld_cdma_k2k_t *api = (sg_api_pld_cdma_k2k_t *)api_buf;
  CORE_PRINT("%s: input_addr=0x%llx, output_addr=0x%llx, reduce=%s\n", __func__,
             api->input_addr, api->output_addr,
             _reduce_type_name[api->reduce_op]);
  CDMA_OPCODE reduce_type = (CDMA_OPCODE)(api->reduce_op);
  global_addr_t input = (global_addr_t)api->input_addr;
  global_addr_t output = (global_addr_t)api->output_addr;
  global_addr_t reduce = api->input_addr + CNT * sizeof(float);
  global_addr_t l2_reduce = tpu_l2_sram_get_start_addr();
  global_addr_t l2_reduce_buffer = ALIGN(l2_reduce + CNT * sizeof(float), 128);
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  float *input_p = (float *)tpu_global_mem_addr(input);
  float *reduce_input_p = (float *)tpu_global_mem_addr(reduce);
  for (int i = 0; i < W; i++) {
    CORE_PRINT("input[%d]=%f, reduce_input[%d]=%f\n", i, input_p[i], i,
               reduce_input_p[i]);
  }
#endif
  if (api->reduce_op != 0) {
    nodechip_cdma_k2k_lossy_decompress(input, reduce, l2_reduce,
                                       l2_reduce_buffer, output, reduce_type);
  } else {
    CORE_PRINT("================================= CDMA_OPCODE_ADD\n");
    nodechip_cdma_k2k_lossy_decompress(
        input, reduce, l2_reduce, l2_reduce_buffer, output, CDMA_OPCODE_ADD);
    CORE_PRINT("================================= CDMA_OPCODE_MUL\n");
    nodechip_cdma_k2k_lossy_decompress(
        input, reduce, l2_reduce, l2_reduce_buffer, output, CDMA_OPCODE_MUL);
    CORE_PRINT("================================= CDMA_OPCODE_MAX\n");
    nodechip_cdma_k2k_lossy_decompress(
        input, reduce, l2_reduce, l2_reduce_buffer, output, CDMA_OPCODE_MAX);
    CORE_PRINT("================================= CDMA_OPCODE_MIN\n");
    nodechip_cdma_k2k_lossy_decompress(
        input, reduce, l2_reduce, l2_reduce_buffer, output, CDMA_OPCODE_MIN);
  }
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: all test finised\n", __func__);
  CORE_PRINT("=================================\n");
}