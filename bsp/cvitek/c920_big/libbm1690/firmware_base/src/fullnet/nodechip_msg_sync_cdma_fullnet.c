#ifdef USING_CMODEL
#include "cmodel_multi_thread.h"
#endif

#include "atomic_cdma_gen_cmd.h"
#include "atomic_gdma_gen_cmd.h"
#include "common_def.h"
#include "firmware_common.h"
#include "firmware_common_inline.h"
#include "firmware_runtime.h"
#include "firmware_timer.h"
#include "gdma_reg_value.h"
#include "sg_api_struct.h"
#include "tpu_kernel.h"

#ifdef USING_CMODEL
#include "cmodel_common.h"
#include "sg_fp16.h"
#else
#include "firmware_top.h"
#endif
#ifdef USING_FAKE_DDR_MODE
#include "atomic_sys_gen_cmd.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

void nodechip_multi_msg_sync_cdma(const sg_api_msg_sync_cdma_t *api,
                                  unsigned int case_id,
                                  u64 total_cmd_addr_shift) {
  unsigned int cdma_port = case_id % MAX_CDMA_NUM;
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  const char *CDMA_OPCODE_NAME[CDMA_OPCODE_NUM] = {
      "CDMA_OPCODE_NONE", "CDMA_OPCODE_MUL", "CDMA_OPCODE_MAX",
      "CDMA_OPCODE_MIN", "CDMA_OPCODE_ADD"};
  const char *CDMA_DTYPE_NAME[CDMA_DTYPE_FORMAT_NUM] = {
      "CDMA_DTYPE_INT8",  "CDMA_DTYPE_FP16",  "CDMA_DTYPE_FP32",
      "CDMA_DTYPE_INT16", "CDMA_DTYPE_INT32", "CDMA_DTYPE_BF16",
      "CDMA_DTYPE_FP20"};
#endif
  CORE_PRINT(
      "======== current test case: %u, cdma_port: %u/%u, dtype:%s, reduce:%s\n",
      case_id, cdma_port, MAX_CDMA_NUM,
      CDMA_DTYPE_NAME[api->engine_param[case_id].dtype],
      CDMA_OPCODE_NAME[api->engine_param[case_id].reduce_opcde]);
#ifdef USING_FAKE_DDR_MODE
  const int core_offset = PLD_BASE_ADDR;
  const u64 CMODEL_GMEM_START_ADDR = 0x0;
  int base_idx = 0;
  //实际中基地址可能为负数
  // 0xc000000 is 192M for jumping multicore firmware packages
  u64 base_addr = total_cmd_addr_shift;
  base_addr += (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR + core_offset);
  CORE_PRINT(" bdc gdma hau sdma cdma cmd num( %d, %d, %d, %d, %d) ,offset( "
             "%llx, %llx, %llx, %llx, %llx), base_addr=0x%llx, core_id=%d\n",
             api->engine_param[case_id].tpu_cmd_nums,
             api->engine_param[case_id].gdma_cmd_nums,
             api->engine_param[case_id].hau_cmd_nums,
             api->engine_param[case_id].sdma_cmd_nums,
             api->engine_param[case_id].cdma_cmd_nums,
             api->engine_param[case_id].tpu_cmd_addr,
             api->engine_param[case_id].gdma_cmd_addr,
             api->engine_param[case_id].hau_cmd_addr,
             api->engine_param[case_id].sdma_cmd_addr,
             api->engine_param[case_id].cdma_cmd_addr, base_addr, CORE_ID);
#endif

  CMD_ID_NODE id_node;
  resync_cmd_id(&id_node);
  u32 write_gdma_reg, write_sdma_reg;

  if (api->engine_param[case_id].tpu_cmd_nums != 0) {
    ASSERT((api->engine_param[case_id].tpu_cmd_addr & 0x7f) == 0x0);
    u64 bdc_cmd_offset_shift = api->engine_param[case_id].tpu_cmd_addr >> 5;
    u32 write_bdc_reg = (bdc_cmd_offset_shift & 0xffffffff);
    write_bdc_reg = write_bdc_reg | 0x1;
    WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
    CORE_PRINT(" bdc des config done, cmd_num=%d, core_id=%d\n",
               api->engine_param[case_id].tpu_cmd_nums, CORE_ID);
  }
  if (api->engine_param[case_id].gdma_cmd_nums != 0) {
    ASSERT((api->engine_param[case_id].gdma_cmd_addr & 0x7f) == 0x0);
    u64 gdma_cmd_offset_shift = api->engine_param[case_id].gdma_cmd_addr >> 7;
    write_gdma_reg = (gdma_cmd_offset_shift & 0x0fffffff);
    WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
#ifdef USING_FAKE_DDR_MODE
    atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_GDMA);
#endif

    write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
    write_gdma_reg = write_gdma_reg | 0x1;
    WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
    CORE_PRINT(" gdma des config done, cmd_num=%d, core_id=%d\n",
               api->engine_param[case_id].gdma_cmd_nums, CORE_ID);
  }

  if (api->engine_param[case_id].hau_cmd_nums != 0) {
    ASSERT((api->engine_param[case_id].hau_cmd_addr & 0x3) == 0);
#ifdef USING_FAKE_DDR_MODE
    atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_HAU);
#endif
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
    FW_REG_ID_WRITE(
        HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2,
        ((api->engine_param[case_id].hau_cmd_addr & 0xffffffff) >> 2));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32,
                    (api->engine_param[case_id].hau_cmd_addr >> 32));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
    CORE_PRINT(" hau des config done, cmd_num=%d, core_id=%d\n",
               api->engine_param[case_id].hau_cmd_nums, CORE_ID);
  }
  if (api->engine_param[case_id].sdma_cmd_nums != 0) {
    ASSERT((api->engine_param[case_id].sdma_cmd_addr & 0x7f) == 0x0);
    u64 sdma_cmd_offset_shift = api->engine_param[case_id].sdma_cmd_addr >> 7;
    write_sdma_reg = (sdma_cmd_offset_shift & 0x0fffffff);
    WRITE_REG((SDMA_ENGINE_MAIN_CTRL + 0x4), write_sdma_reg, NODECHIP_REG);
#ifdef USING_FAKE_DDR_MODE
    atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_SDMA);
#endif

    write_sdma_reg = READ_REG(SDMA_ENGINE_MAIN_CTRL);
    write_sdma_reg = write_sdma_reg | 0x1;
    WRITE_REG(SDMA_ENGINE_MAIN_CTRL, write_sdma_reg, NODECHIP_REG);
    CORE_PRINT(" sdma des config done, cmd_num=%d, core_id=%d\n",
               api->engine_param[case_id].sdma_cmd_nums, CORE_ID);
  }

  if (api->engine_param[case_id].cdma_cmd_nums != 0) {
#ifdef USING_FAKE_DDR_MODE
    // atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_CDMA);
    atomic_cdma_port_set_base_ddr(&base_idx, &base_addr, 1, cdma_port);
#endif
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
    // float *fake_input = (float *)tpu_global_mem_addr(0xc000000);
    void *ori_input = (void *)tpu_global_mem_addr(api->input_cmd_addr);
    CORE_PRINT("input_addr=0x%llx, output_addr=0x%llx, ori_input=%p\n",
               api->input_addr, api->output_addr, ori_input);
    for (int i = 0; i < 32; i++) {
      if (CDMA_DTYPE_INT32 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
        uint32_t *print_input = (uint32_t *)ori_input;
        CORE_PRINT("ori_input[%d]=%u, ori_l2_input[%d]=%u\n", i,
                   print_input[i], i, print_input[i + 32]);
      } else if (CDMA_DTYPE_FP32 ==
                 (CDMA_DTYPE)api->engine_param[case_id].dtype) {
        float *print_input = (float *)ori_input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i,
                   print_input[i], i, print_input[i + 32]);
      }
    }
    if (CDMA_DTYPE_FP32 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      CORE_PRINT("ori_input:\n");
      uint32_t *print_input = (uint32_t *)ori_input;
      for (int i = 0; i < 32; i++) {
        CORE_PRINT("%u, ", print_input[i]);
        if (i % 8 == 7)
          CORE_PRINT("\n");
      }
    }
    if (CDMA_DTYPE_BF16 == (CDMA_DTYPE)api->engine_param[case_id].dtype ||
        CDMA_DTYPE_FP16 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      CORE_PRINT("ori_input\n");
      for (int i = 0; i < 32; i++) {
        uint16_t *print_input = (uint16_t *)ori_input;
        CORE_PRINT("%u, ", print_input[i]);
        if (i % 8 == 7)
          CORE_PRINT("\n");
      }
      CORE_PRINT("ori_l2_input\n");
      for (int i = 0; i < 32; i++) {
        uint16_t *print_input = (uint16_t *)ori_input;
        CORE_PRINT("%u, ", print_input[i + 32]);
        if (i % 8 == 7)
          CORE_PRINT("\n");
      }
    }
#ifdef USING_CMODEL
    for (int i = 0; i < 32; i++) {
      if (CDMA_DTYPE_FP16 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
        fp16 *print_input = (fp16 *)ori_input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i,
                   fp16_to_fp32(print_input[i]).fval, i,
                   fp16_to_fp32(print_input[i + 32]).fval);
      } else if (CDMA_DTYPE_BF16 ==
                 (CDMA_DTYPE)api->engine_param[case_id].dtype) {
        bf16 *print_input = (bf16 *)ori_input;
        CORE_PRINT("ori_input[%d]=%f, ori_l2_input[%d]=%f\n", i,
                   bf16_to_fp32(print_input[i]).fval, i,
                   bf16_to_fp32(print_input[i + 32]).fval);
      }
    }
#endif
#endif

    // write cdma des addr
    ASSERT((api->engine_param[case_id].cdma_cmd_addr & 0x7f) == 0x0);
    u64 cdma_des_addr_l32 = api->engine_param[case_id].cdma_cmd_addr >> 7;
    u64 cdma_des_addr_h1 = api->engine_param[case_id].cdma_cmd_addr >> 39;
    WRITE_REG(CDMA_CSR_REG_DES_ADDR_L32(cdma_port), cdma_des_addr_l32,
              NODECHIP_REG);
    WRITE_REG(CDMA_CSR_REG_DES_ADDR_H1(cdma_port), cdma_des_addr_h1,
              NODECHIP_REG);
    // write cdma des rw route
    u32 des_rw_addr = READ_REG(CDMA_CSR_REG_DES_RW_ADDR(cdma_port));
    des_rw_addr &= (~(0xff));
    des_rw_addr |= 0x80;
    WRITE_REG(CDMA_CSR_REG_DES_RW_ADDR(cdma_port), des_rw_addr, NODECHIP_REG);
    // write cdma route
    atomic_cdma_route_configuration(0, CDMA_ROUTE_AXI_RN, cdma_port);
    // write cdma tcredit
    atomic_cdma_config_tcredit_for_pld_test(
        0, api->engine_param[case_id].l2mem_reduce_addr, 1, 1, 1, 32, 32, 32,
        32, (CDMA_OPCODE)(api->engine_param[case_id].reduce_opcde), cdma_port);
    // enable cdma desr mode
    u32 cdma_csr0 = READ_REG(CDMA_ENGINE_MAIN_CTRL(cdma_port));
    cdma_csr0 = cdma_csr0 | 0x1;
    WRITE_REG(CDMA_ENGINE_MAIN_CTRL(cdma_port), cdma_csr0, NODECHIP_REG);
    CORE_PRINT(" cdma des config done, cmd_num=%d, core_id=%d, PORT:%d\n",
               api->engine_param[case_id].cdma_cmd_nums, CORE_ID,
               (int)cdma_port);
    CORE_PRINT("DES_ADDR_L32 addr:%llx, value:%x;\n",
               (u64)CDMA_CSR_REG_DES_ADDR_L32(cdma_port),
               (u32)(READ_REG(CDMA_CSR_REG_DES_ADDR_L32(cdma_port))));
    CORE_PRINT("DES_ADDR_H1 addr:%llx, value:%x;\n",
               (u64)CDMA_CSR_REG_DES_ADDR_H1(cdma_port),
               (u32)(READ_REG(CDMA_CSR_REG_DES_ADDR_H1(cdma_port))));
    CORE_PRINT("DES_RW_ADDR addr:%llx, value:%x;\n",
               (u64)CDMA_CSR_REG_DES_RW_ADDR(cdma_port),
               (u32)(READ_REG(CDMA_CSR_REG_DES_RW_ADDR(cdma_port))));
    CORE_PRINT("CDMA_CSR0 addr:%llx, value:%x;\n",
               (u64)CDMA_ENGINE_MAIN_CTRL(cdma_port),
               (u32)(READ_REG(CDMA_ENGINE_MAIN_CTRL(cdma_port))));
    CORE_PRINT("CDMA_CSR_INTER_DIE_RW addr: %llx, value:%x;\n",
               (u64)CDMA_CSR_INTER_DIE_RW(cdma_port),
               (u32)(READ_REG(CDMA_CSR_INTER_DIE_RW(cdma_port))));
    CORE_PRINT("CDMA_CSR_INTRA_DIE_RW addr: %llx, value:%x.\n",
               (u64)CDMA_CSR_INTRA_DIE_RW(cdma_port),
               (u32)(READ_REG(CDMA_CSR_INTRA_DIE_RW(cdma_port))));
  }

#ifdef USING_CMODEL
  bool using_cmd_arr[5];
  using_cmd_arr[ENGINE_BD] = (api->engine_param[case_id].tpu_cmd_nums != 0);
  using_cmd_arr[ENGINE_GDMA] = (api->engine_param[case_id].gdma_cmd_nums != 0);
  using_cmd_arr[ENGINE_HAU] = (api->engine_param[case_id].hau_cmd_nums != 0);
  using_cmd_arr[ENGINE_SDMA] = (api->engine_param[case_id].sdma_cmd_nums != 0);
  using_cmd_arr[ENGINE_CDMA] = (api->engine_param[case_id].cdma_cmd_nums != 0);

  u64 engine_address_arr[5];
  engine_address_arr[ENGINE_BD] = api->engine_param[case_id].tpu_cmd_addr;
  engine_address_arr[ENGINE_GDMA] = api->engine_param[case_id].gdma_cmd_addr;
  engine_address_arr[ENGINE_HAU] = api->engine_param[case_id].hau_cmd_addr;
  engine_address_arr[ENGINE_SDMA] = api->engine_param[case_id].sdma_cmd_addr;
  engine_address_arr[ENGINE_CDMA] = api->engine_param[case_id].cdma_cmd_addr;

  cmodel_multi_engine_cdma(engine_address_arr, using_cmd_arr,
                           get_cur_nodechip_idx(), (int)cdma_port);
#endif

  id_node.bd_cmd_id = api->engine_param[case_id].tpu_cmd_nums;
  id_node.gdma_cmd_id = api->engine_param[case_id].gdma_cmd_nums;
  id_node.hau_cmd_id = api->engine_param[case_id].hau_cmd_nums;
  id_node.sdma_cmd_id = api->engine_param[case_id].sdma_cmd_nums;
  id_node.cdma_cmd_id[cdma_port] = api->engine_param[case_id].cdma_cmd_nums;
  poll_all_engine_done(&id_node);
  tpu_cdma_port_poll((int)cdma_port);
  CORE_PRINT("poll done.  bdc gdma hau cdma sdma cmd num(%d, %d, %d, %d, %d)\n",
             api->engine_param[case_id].tpu_cmd_nums,
             api->engine_param[case_id].gdma_cmd_nums,
             api->engine_param[case_id].hau_cmd_nums,
             api->engine_param[case_id].cdma_cmd_nums,
             api->engine_param[case_id].sdma_cmd_nums);
#if defined(USING_FAKE_DDR_MODE) &&                                            \
    ((defined(USING_CMODEL) && defined(DEBUG)) ||                              \
     (defined(USING_FW_DEBUG) && !defined(USING_CMODEL)))
  // CORE_PRINT("l2 output:\n");
  // uint32_t *l2_output_32 = (uint32_t *)tpu_l2_sram_addr(
  //     api->engine_param[case_id].l2mem_reduce_addr);
  // uint16_t *l2_output_16 = (uint16_t *)tpu_l2_sram_addr(
  //     api->engine_param[case_id].l2mem_reduce_addr);
  // if (CDMA_DTYPE_INT32 == (CDMA_DTYPE)api->engine_param[case_id].dtype ||
  //     CDMA_DTYPE_FP32 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
  //   for (int i = 0; i < 32; i++) {
  //     CORE_PRINT("%u, ", l2_output_32[i]);
  //     if (i % 8 == 7)
  //       CORE_PRINT("\n");
  //   }
  // } else if (CDMA_DTYPE_BF16 ==
  //                (CDMA_DTYPE)api->engine_param[case_id].dtype ||
  //            CDMA_DTYPE_FP16 ==
  //                (CDMA_DTYPE)api->engine_param[case_id].dtype) {
  //   for (int i = 0; i < 32; i++) {
  //     CORE_PRINT("%u, ", l2_output_16[i]);
  //     if (i % 8 == 7)
  //       CORE_PRINT("\n");
  //   }
  // }
  // for (int i = 0; i < 20; i++) {
  //   CORE_PRINT("%u, ", l2_output_32[i]);
  //   if (i % 5 == 4)
  //     CORE_PRINT("\n");
  // }
  tpu_invalidate_cache(
      base_addr + 0xc000000 + api->output_addr - api->input_addr, 0x1000);
  void *ori_output = (void *)tpu_global_mem_addr(0xc000000 + api->output_addr -
                                                 api->input_addr);
  for (int i = 0; i < 32; i++) {
    // CORE_PRINT("output[%d]=%u\n", i, output[i]);
    if (CDMA_DTYPE_INT32 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      uint32_t *print_output = (uint32_t *)ori_output;
      CORE_PRINT("output[%d]=%u\n", i, print_output[i]);
    } else if (CDMA_DTYPE_FP32 ==
               (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      float *print_output = (float *)ori_output;
      CORE_PRINT("output[%d]=%f\n", i, print_output[i]);
    }
  }
  if (CDMA_DTYPE_FP32 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
    CORE_PRINT("output:\n");
    uint32_t *print_output = (uint32_t *)ori_output;
    for (int i = 0; i < 32; i++) {
      CORE_PRINT("%u, ", print_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    }
  }
  if (CDMA_DTYPE_BF16 == (CDMA_DTYPE)api->engine_param[case_id].dtype ||
      CDMA_DTYPE_FP16 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
    CORE_PRINT("output:\n");
    for (int i = 0; i < 32; i++) {
      uint16_t *print_output = (uint16_t *)ori_output;
      CORE_PRINT("%u, ", print_output[i]);
      if (i % 8 == 7)
        CORE_PRINT("\n");
    }
  }
#ifdef USING_CMODEL
  for (int i = 0; i < 32; i++) {
    if (CDMA_DTYPE_FP16 == (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      fp16 *print_output = (fp16 *)ori_output;
      CORE_PRINT("output[%d]=%f\n", i, fp16_to_fp32(print_output[i]).fval);
    } else if (CDMA_DTYPE_BF16 ==
               (CDMA_DTYPE)api->engine_param[case_id].dtype) {
      bf16 *print_output = (bf16 *)ori_output;
      CORE_PRINT("output[%d]=%f\n", i, bf16_to_fp32(print_output[i]).fval);
    }
  }
#endif
#endif
}

sg_fw_status_t sg_api_msg_sync_cdma(unsigned char *api_buf, int size) {
  // #ifdef USING_CMODEL
  //     return SG_FW_SUCCESS;
  // #endif
  const sg_api_msg_sync_cdma_t *msg_sync_ptr =
      (sg_api_msg_sync_cdma_t *)api_buf;
  ASSERT(api_buf && size == sizeof(sg_api_msg_sync_cdma_t));
  sg_api_msg_sync_cdma_t api = {0};
  memcpy(&api, msg_sync_ptr, sizeof(sg_api_msg_sync_cdma_t));
  // PLD_CMD_ADDR = PLD_GLOBAL_MEM_START_ADDR - CMODEL_GLOBAL_MEM_START_ADDR
  u64 CMODEL_GMEM_START_ADDR = 0x0;
  u64 fixed_addr = GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR;
  api.input_addr = msg_sync_ptr->input_addr + fixed_addr;
  api.output_addr = msg_sync_ptr->output_addr + fixed_addr;
  api.pio_addr = msg_sync_ptr->pio_addr + fixed_addr;
  u64 total_cmd_addr_shift = 0;
  for (unsigned int case_id = 0; case_id < api.cdma_test_loop; case_id++) {
    api.engine_param[case_id].tpu_cmd_addr =
        msg_sync_ptr->engine_param[case_id].tpu_cmd_addr + fixed_addr;
    api.engine_param[case_id].gdma_cmd_addr =
        msg_sync_ptr->engine_param[case_id].gdma_cmd_addr + fixed_addr;
    api.engine_param[case_id].hau_cmd_addr =
        msg_sync_ptr->engine_param[case_id].hau_cmd_addr + fixed_addr;
    api.engine_param[case_id].sdma_cmd_addr =
        msg_sync_ptr->engine_param[case_id].sdma_cmd_addr + fixed_addr;
    api.engine_param[case_id].cdma_cmd_addr =
        msg_sync_ptr->engine_param[case_id].cdma_cmd_addr + fixed_addr;
    api.engine_param[case_id].imm_buf_addr =
        msg_sync_ptr->engine_param[case_id].imm_buf_addr + fixed_addr;
    total_cmd_addr_shift +=
        (ALIGN((api.engine_param[case_id].tpu_cmd_byte_size), 4096) +
         ALIGN((api.engine_param[case_id].gdma_cmd_byte_size), 4096) +
         ALIGN((api.engine_param[case_id].hau_cmd_byte_size), 4096) +
         ALIGN((api.engine_param[case_id].sdma_cmd_byte_size), 4096) +
         ALIGN((api.engine_param[case_id].cdma_cmd_byte_size), 4096) +
         ALIGN((api.engine_param[case_id].imm_buf_byte_size), 4096));
  }
  for (unsigned int case_id = 0; case_id < api.cdma_test_loop; case_id++) {
    nodechip_multi_msg_sync_cdma(&api, case_id, total_cmd_addr_shift);
  }
  return SG_FW_SUCCESS;
}
#ifdef __cplusplus
}
#endif
