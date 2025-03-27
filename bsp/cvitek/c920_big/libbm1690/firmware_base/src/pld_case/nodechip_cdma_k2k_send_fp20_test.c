#include "nodechip_cdma_pld_test_utils.h"
void msg_sync_test_txrx();
void msg_sync_test_tx();
void msg_sync_test_rx();
void msg_sync_test_txrx_diff_msgid();
void msg_sync_test_allinone();
void nodechip_cdma_k2k_send_fp20(global_addr_t input_addr,
                                 global_addr_t reduce_addr,
                                 global_addr_t l2_reduce_addr,
                                 global_addr_t output_addr, data_type_t dtype,
                                 CDMA_OPCODE reduce_type) {
  global_addr_t l2_reduce_res =
      ALIGN(l2_reduce_addr + CNT * sizeof(float), 128);
  tpu_initialize();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("set l2 ori data by gdma\n");
  dim4 d_shape = {.n = N, .c = C, .h = H, .w = W};
  tpu_gdma_lossy_compress_S2S(l2_reduce_addr, input_addr, &d_shape, NULL, NULL);
  tpu_gdma_lossy_compress_S2S(l2_reduce_res, reduce_addr, &d_shape, NULL, NULL);
  tpu_poll();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  void *l2_input = (void *)tpu_l2_sram_addr(l2_reduce_addr);
  for (int i = 0; i < 20; i++) {
    uint32_t *print_l2_input = (uint32_t *)l2_input;
    CORE_PRINT("%u, ", print_l2_input[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
  void *l2_output = (void *)tpu_l2_sram_addr(l2_reduce_res);
  for (int i = 0; i < 20; i++) {
    uint32_t *print_l2_output = (uint32_t *)l2_output;
    CORE_PRINT("%u, ", print_l2_output[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
#endif
#if defined(USING_CMODEL) && !defined(CDMA_DES_PLD_TEST)
  CORE_PRINT("=================================\n");
  CORE_PRINT("cmodel reduce ori data to l2 by sdma\n");
  tpu_gdma_cpy_reduce_L22L2(l2_reduce_res, l2_reduce_addr, &d_shape, NULL, NULL,
                            dtype, ALL_REDUCE_PSUM_WR, (int)reduce_type);
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
  atomic_cdma_config_tcredit_for_pld_test(dst_chipid, l2_reduce_res, N, C, H, W,
                                          N_STRIDE, C_STRIDE, H_STRIDE,
                                          reduce_type, -1);
  tpu_cdma_send(dst_chipid, src_chipid, l2_reduce_addr, N, C, H, W, N_STRIDE,
                C_STRIDE, H_STRIDE, reduce_type, dtype);

  // D. sys end config for des mode(jump this step since it's pio mode)
  // E. pio run cmd config
  atomic_cdma_pio_interrupt_poll(dst_chipid);
  tpu_poll();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  for (int i = 0; i < 20; i++) {
    uint32_t *print_l2_output = (uint32_t *)l2_output;
    CORE_PRINT("%u, ", print_l2_output[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("move l2 reduced data to ddr outout by gdma\n");
  tpu_gdma_lossy_decompress_S2S(output_addr, l2_reduce_res, &d_shape, NULL,
                                NULL);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  tpu_invalidate_cache(GLOBAL_MEM_START_ADDR + output_addr, 0x1000);
  void *output = (void *)tpu_global_mem_addr(output_addr);
  for (int i = 0; i < W; i++) {
    float *print_output = (float *)output;
    CORE_PRINT("output[%d]=%f\n", i, print_output[i]);
  }
  uint32_t *output_u32 = (uint32_t *)output;
  for (int i = 0; i < W; i++) {
    CORE_PRINT("%u, ", output_u32[i]);
    if (i % 8 == 7)
      CORE_PRINT("\n");
  }
#endif
}

void nodechip_cdma_k2k_send_fp20_test(unsigned char *api_buf) {
  if (PLD_K2K_CDMA_TEST_PORT == 8) {
    CORE_PRINT("=================================\n");
    CORE_PRINT("%s: p2p port, jump send reduce\n", __func__);
  } else {
    CORE_PRINT("=================================\n");
    sg_api_pld_cdma_k2k_t *api = (sg_api_pld_cdma_k2k_t *)api_buf;
    CDMA_OPCODE reduce_type = (CDMA_OPCODE)(api->reduce_op);
    data_type_t dtype = DT_FP20;
    u64 l2mem_index = 0;
    global_addr_t reduce_addr = api->input_addr + CNT * sizeof(float);
    CORE_PRINT("%s: input_addr=0x%llx, output_addr=0x%llx, reduce_addr=0x%llx, "
               "reduce=%s\n",
               __func__, api->input_addr, api->output_addr, reduce_addr,
               _reduce_type_name[api->reduce_op]);
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
    void *input = (void *)tpu_global_mem_addr(api->input_addr);
    float *print_output = (float *)input;
    for (int i = 0; i < W; i++) {
      CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i, print_output[i],
                 i, print_output[i + 32]);
    }
#endif
    CORE_PRINT("tpu_l2_sram_get_start_addr()+ %lld * 0x800000=%llx\n",
               l2mem_index,
               tpu_l2_sram_get_start_addr() + l2mem_index * 0x800000);
    if (api->reduce_op != 0) {
      nodechip_cdma_k2k_send_fp20(api->input_addr, reduce_addr,
                                  tpu_l2_sram_get_start_addr() +
                                      l2mem_index * 0x800000,
                                  api->output_addr, dtype, reduce_type);
    } else {
      CORE_PRINT("================================= CDMA_OPCODE_ADD\n");
      nodechip_cdma_k2k_send_fp20(api->input_addr, reduce_addr,
                                  tpu_l2_sram_get_start_addr() +
                                      l2mem_index * 0x800000,
                                  api->output_addr, dtype, CDMA_OPCODE_ADD);
      CORE_PRINT("================================= CDMA_OPCODE_MUL\n");
      nodechip_cdma_k2k_send_fp20(api->input_addr, reduce_addr,
                                  tpu_l2_sram_get_start_addr() +
                                      l2mem_index * 0x800000,
                                  api->output_addr, dtype, CDMA_OPCODE_MUL);
      CORE_PRINT("================================= CDMA_OPCODE_MAX\n");
      nodechip_cdma_k2k_send_fp20(api->input_addr, reduce_addr,
                                  tpu_l2_sram_get_start_addr() +
                                      l2mem_index * 0x800000,
                                  api->output_addr, dtype, CDMA_OPCODE_MAX);
      CORE_PRINT("================================= CDMA_OPCODE_MIN\n");
      nodechip_cdma_k2k_send_fp20(api->input_addr, reduce_addr,
                                  tpu_l2_sram_get_start_addr() +
                                      l2mem_index * 0x800000,
                                  api->output_addr, dtype, CDMA_OPCODE_MIN);
    }
  }

  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: all test finised\n", __func__);
  CORE_PRINT("=================================\n");
}