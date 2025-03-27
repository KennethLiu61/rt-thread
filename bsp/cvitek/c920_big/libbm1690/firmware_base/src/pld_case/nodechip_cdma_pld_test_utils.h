#include "atomic_cdma_gen_cmd.h"
#include "firmware_common_inline.h"
#include "firmware_common_macro.h"
#include "nodechip_pld_test.h"
#include "tpu_defs.h"
#include "tpu_kernel.h"
// #include "sg_fp20.h"

#define N 1
#define C 1
#define H 1
#define W 32
#define CNT N *C *H *W
#define H_STRIDE (W)
#define C_STRIDE (H_STRIDE * H)
#define N_STRIDE (C_STRIDE * C)
#define FP20_NUM_PER_BLOCK 51
#define FP20_NUM_PER_BLOCK_SIZE 128

#define MASK_12BIT ((u32)0xfff)
#define MASK_20BIT ((u32)0xfffff)
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
static const char *_reduce_type_name[] = {"CDMA_OPCODE_ALL", "CDMA_OPCODE_MUL",
                                          "CDMA_OPCODE_MAX", "CDMA_OPCODE_MIN",
                                          "CDMA_OPCODE_ADD"};
#endif
static inline data_type_t get_prec(PREC precision) {
  data_type_t dtype = -1;
  if (precision == FP16) {
    return DT_FP16;
  } else if (precision == FP32) {
    return DT_FP32;
  } else if (precision == INT32) {
    return DT_INT32;
  } else if (precision == BFP16) {
    return DT_BFP16;
  } else if (precision == FP20) {
    return DT_FP20;
  }
  TPUKERNEL_ASSERT(0);
  return dtype;
}
static int get_byte_size(data_type_t dtype) {
  int byte_size = 0;
  if (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8E4M3 ||
      dtype == DT_FP8E5M2) {
    return 1;
  } else if (dtype == DT_FP16 || dtype == DT_INT16 || dtype == DT_UINT16 ||
             dtype == DT_BFP16) {
    return 2;
  } else if (dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32) {
    return 4;
  }
  TPUKERNEL_ASSERT(0);
  return byte_size;
}
static void tpu_msg_sync_delete_later() {
  int msg_id = tpu_get_local_msg_id();
  int msg_cnt = 5;
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
}

static void msg_sync_for_des_test() {
  int msg_id = tpu_get_local_msg_id();
  int msg_cnt = 3;
  CORE_PRINT("=================================\n");
  CORE_PRINT("%s: msg_id=%d, msg_cnt=%d\n", __func__, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
  tpu_cdma_tx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_send_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_cdma_tx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_rx_wait_msg(PLD_K2K_CDMA_TEST_PORT, msg_id, msg_cnt);
  tpu_cdma_nop_sync(PLD_K2K_CDMA_TEST_PORT);
}
static void print_short_fp(void *data, int len) {
#if (defined(USING_CMODEL) && defined(DEBUG)) ||                               \
    (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  uint32_t *l2_output_p = (uint32_t *)data;
  for (int i = 0; i < len; i++) {
    CORE_PRINT("%u, ", l2_output_p[i]);
    if (i % 5 == 4)
      CORE_PRINT("\n");
  }
#endif
}
