#include "nodechip_cdma_pld_test_utils.h"
#ifdef USING_CMODEL
#include "sg_fp16.h"
#endif
void msg_sync_test_txrx();
void msg_sync_test_tx();
void msg_sync_test_rx();
void msg_sync_test_txrx_diff_msgid();
void msg_sync_test_allinone();
void nodechip_cdma_k2k_send(global_addr_t input_addr, global_addr_t reduce_addr,
                            global_addr_t l2_reduce_addr,
                            global_addr_t output_addr, data_type_t dtype,
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
  tpu_gdma_cpy_S2S(l2_reduce_addr, reduce_addr, &d_shape, NULL, NULL, dtype);
  tpu_poll();
#ifndef CDMA_DES_PLD_TEST
  tpu_msg_sync_delete_later();
#else
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  void *l2_output = (void *)tpu_l2_sram_addr(l2_reduce_addr);
  // for (int i = 0; i < W; i++) {
  //   if (DT_INT32 == dtype) {
  //     uint32_t *print_l2_output = (uint32_t *)l2_output;
  //     CORE_PRINT("l2_output[%d]=%u\n", i, print_l2_output[i]);
  //   } else if (DT_FP32 == dtype) {
  //     float *print_l2_output = (float *)l2_output;
  //     CORE_PRINT("l2_output[%d]=%f\n", i, print_l2_output[i]);
  //   } else if (DT_BFP16 == dtype || DT_FP16 == dtype) {
  //     uint16_t *print_l2_output = (uint16_t *)l2_output;
  //     CORE_PRINT("l2_output[%d]=%u\n", i, print_l2_output[i]);
  //   }
  // }
#endif
#if defined(USING_CMODEL) && !defined(CDMA_DES_PLD_TEST)
  CORE_PRINT("=================================\n");
  CORE_PRINT("cmodel reduce ori data to l2 by sdma\n");
  tpu_sdma_cpy_reduce_S2L2(l2_reduce_addr, input_addr, &d_shape, NULL, NULL,
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
  atomic_cdma_config_tcredit_for_pld_test(dst_chipid, l2_reduce_addr, N, C, H,
                                          W, N_STRIDE, C_STRIDE, H_STRIDE,
                                          reduce_type, -1);
  tpu_cdma_send(dst_chipid, src_chipid, input_addr, N, C, H, W, N_STRIDE,
                C_STRIDE, H_STRIDE, reduce_type, dtype);

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
  CORE_PRINT("l2 output:\n");
  for (int i = 0; i < W; i++) {
    if (DT_INT32 == dtype || DT_FP32 == dtype) {
      uint32_t *print_l2_output = (uint32_t *)l2_output;
      CORE_PRINT("%u, ", print_l2_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    } else if (DT_BFP16 == dtype || DT_FP16 == dtype) {
      uint16_t *print_l2_output = (uint16_t *)l2_output;
      CORE_PRINT("%u, ", print_l2_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    }
  }
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("move l2 reduced data to ddr outout by gdma\n");
  tpu_gdma_cpy_S2S(output_addr, l2_reduce_addr, &d_shape, NULL, NULL, dtype);
  tpu_poll();
#ifdef CDMA_DES_PLD_TEST
  msg_sync_for_des_test();
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  tpu_invalidate_cache(GLOBAL_MEM_START_ADDR + output_addr, 0x1000);
  void *output = (void *)tpu_global_mem_addr(output_addr);
  for (int i = 0; i < W; i++) {
    if (DT_INT32 == dtype) {
      uint32_t *print_output = (uint32_t *)output;
      CORE_PRINT("output[%d]=%u\n", i, print_output[i]);
    } else if (DT_FP32 == dtype) {
      float *print_output = (float *)output;
      CORE_PRINT("output[%d]=%f\n", i, print_output[i]);
    }
  }
  if (DT_FP32 == dtype) {
    CORE_PRINT("output:\n");
    uint32_t *print_output = (uint32_t *)output;
    for (int i = 0; i < W; i++) {
      CORE_PRINT("%u, ", print_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    }
  }
  if (DT_BFP16 == dtype ||
      DT_FP16 == dtype) {
    CORE_PRINT("output:\n");
    uint16_t *print_output = (uint16_t *)output;
    for (int i = 0; i < W; i++) {
      CORE_PRINT("%u, ", print_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    }
  }
#ifdef USING_CMODEL
  for (int i = 0; i < W; i++) {
    if (DT_FP16 == dtype) {
      fp16 *print_output = (fp16 *)output;
      CORE_PRINT("output[%d]=%f\n", i, fp16_to_fp32(print_output[i]).fval);
    } else if (DT_BFP16 == dtype) {
      bf16 *print_output = (bf16 *)output;
      CORE_PRINT("output[%d]=%f\n", i, bf16_to_fp32(print_output[i]).fval);
    }
  }
#endif
#endif
}

void nodechip_cdma_k2k_send_test(unsigned char *api_buf) {
  if (PLD_K2K_CDMA_TEST_PORT == 8) {
    CORE_PRINT("=================================\n");
    CORE_PRINT("%s: p2p port, jump send reduce\n", __func__);
  } else {
    CORE_PRINT("=================================\n");
    sg_api_pld_cdma_k2k_t *api = (sg_api_pld_cdma_k2k_t *)api_buf;
    data_type_t dtype = get_prec(api->dtype);
    CDMA_OPCODE reduce_type = (CDMA_OPCODE)(api->reduce_op);
    u64 l2mem_index = 0;
    global_addr_t reduce_addr = api->input_addr + CNT * get_byte_size(dtype);
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
    const char *_prec_name[] = {"INT8",  "FP16", "FP32", "INT16", "INT32",
                                "BFP16", "INT4", "FP8",  "FP20",  "INT64"};
#endif
    CORE_PRINT("%s: input_addr=0x%llx, output_addr=0x%llx, reduce_addr=0x%llx, "
               "reduce=%s, prec=%s\n",
               __func__, api->input_addr, api->output_addr, reduce_addr,
               _reduce_type_name[api->reduce_op], _prec_name[api->dtype]);
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
    void *input = (void *)tpu_global_mem_addr(api->input_addr);
    for (int i = 0; i < W; i++) {
      if (DT_INT32 == dtype) {
        uint32_t *print_input = (uint32_t *)input;
        CORE_PRINT("ori_input[%d]=%u, ori_l2_input[%d]=%u\n", i, print_input[i],
                   i, print_input[i + 32]);
      } else if (DT_FP32 == dtype) {
        float *print_input = (float *)input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i, print_input[i],
                   i, print_input[i + 32]);
      }
    }
    if (DT_BFP16 == dtype || DT_FP16 == dtype) {
      CORE_PRINT("ori input:\n");
      for (int i = 0; i < W; i++) {
        uint16_t *print_input = (uint16_t *)input;
        CORE_PRINT("%u, ", print_input[i]);
        if (i % 8 == 7)
          CORE_PRINT("\n");
      }
      CORE_PRINT("ori l2 input:\n");
      for (int i = 0; i < W; i++) {
        uint16_t *print_input = (uint16_t *)input;
        CORE_PRINT("%u, ", print_input[i + 32]);
        if (i % 8 == 7)
          CORE_PRINT("\n");
      }
    }
#ifdef USING_CMODEL
    for (int i = 0; i < W; i++) {
      if (DT_FP16 == dtype) {
        fp16 *print_input = (fp16 *)input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i,
                   fp16_to_fp32(print_input[i]).fval, i,
                   fp16_to_fp32(print_input[i + 32]).fval);
      } else if (DT_BFP16 == dtype) {
        bf16 *print_input = (bf16 *)input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i,
                   bf16_to_fp32(print_input[i]).fval, i,
                   bf16_to_fp32(print_input[i + 32]).fval);
      }
    }
#endif
#endif
    CORE_PRINT("tpu_l2_sram_get_start_addr()+ %lld * 0x800000=%llx\n",
               l2mem_index,
               tpu_l2_sram_get_start_addr() + l2mem_index * 0x800000);
    if (api->reduce_op != 0) {
      nodechip_cdma_k2k_send(api->input_addr, reduce_addr,
                             tpu_l2_sram_get_start_addr() +
                                 l2mem_index * 0x800000,
                             api->output_addr, dtype, reduce_type);
    } else {
      CORE_PRINT("================================= CDMA_OPCODE_ADD\n");
      nodechip_cdma_k2k_send(api->input_addr, reduce_addr,
                             tpu_l2_sram_get_start_addr() +
                                 l2mem_index * 0x800000,
                             api->output_addr, dtype, CDMA_OPCODE_ADD);
      if (dtype != DT_INT32) {
        CORE_PRINT("================================= CDMA_OPCODE_MUL\n");
        nodechip_cdma_k2k_send(api->input_addr, reduce_addr,
                               tpu_l2_sram_get_start_addr() +
                                   l2mem_index * 0x800000,
                               api->output_addr, dtype, CDMA_OPCODE_MUL);
      }
      CORE_PRINT("================================= CDMA_OPCODE_MAX\n");
      nodechip_cdma_k2k_send(api->input_addr, reduce_addr,
                             tpu_l2_sram_get_start_addr() +
                                 l2mem_index * 0x800000,
                             api->output_addr, dtype, CDMA_OPCODE_MAX);
      CORE_PRINT("================================= CDMA_OPCODE_MIN\n");
      nodechip_cdma_k2k_send(api->input_addr, reduce_addr,
                             tpu_l2_sram_get_start_addr() +
                                 l2mem_index * 0x800000,
                             api->output_addr, dtype, CDMA_OPCODE_MIN);
    }
  }
  msg_sync_test_tx();
  tpu_poll();
  msg_sync_test_rx();
  tpu_poll();
  msg_sync_test_txrx();
  tpu_poll();
  msg_sync_test_txrx_diff_msgid();
  tpu_poll();
#ifndef CDMA_DES_PLD_TEST
  msg_sync_test_allinone();
  tpu_poll();
#endif
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: all test finised\n", __func__);
  CORE_PRINT("=================================\n");
}
void msg_sync_test_txrx() {
  int msg_id = tpu_get_ccl_msg_id();
  int msg_cnt = 2;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id=%d, msg_cnt=%d\n", __func__, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_tx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_tx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}
void msg_sync_test_tx() {
  int msg_id = tpu_get_ccl_msg_id();
  int msg_cnt = 1;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id=%d, msg_cnt=%d\n", __func__, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_tx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_tx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}
void msg_sync_test_rx() {
  int msg_id = tpu_get_ccl_msg_id();
  int msg_cnt = 1;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id=%d, msg_cnt=%d\n", __func__, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_rx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}
void msg_sync_test_txrx_diff_msgid() {
  int msg_id_tx = tpu_get_ccl_msg_id();
  int msg_id_rx = tpu_get_ccl_msg_id();
  int msg_cnt = 1;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id_tx=%d, msg_id_rx=%d, msg_cnt=%d\n", __func__,
             msg_id_tx, msg_id_rx, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_tx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id_tx, msg_cnt);
  tpu_cdma_rx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id_rx, msg_cnt);
  tpu_cdma_tx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id_tx, msg_cnt);
  tpu_cdma_rx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id_rx, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}
void msg_sync_test_allinone() {
  int msg_id = tpu_get_local_msg_id();
  int msg_cnt = 7;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id=%d, msg_cnt=%d\n", __func__, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_tx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_vsdma_send_msg(msg_id, msg_cnt, tpu_core_index());
  tpu_hau_send_msg(msg_id, msg_cnt);
  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_vsdma_wait_msg(msg_id, msg_cnt, tpu_core_index());
  tpu_hau_wait_msg(msg_id, msg_cnt);
  tpu_cdma_tx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}