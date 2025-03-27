#include "atomic_sys_gen_cmd.h"
#include "firmware_common_macro.h"
#include "nodechip_pld_test.h"

void nodechip_read_msg_reg() {
  CMD_ID_NODE id_node;
  resync_cmd_id(&id_node);
  const uint32_t msg_count = 128;
  for (uint32_t i = 0; i < msg_count; i++) {
    atomic_send_msg_gen_cmd(ENGINE_BD, i, (i % 7) + 1, MASTER_THREAD, &id_node);
    poll_all_engine_done(&id_node);
    #ifndef USING_CMODEL
    uint32_t val = READ_REG(TPU_SYS_MSG_REG_ADDR + i * sizeof(int));
    ASSERT_FS_INFO((val & 0x7) == ((i % 7) + 1) && ((val >> 16) & 0x7) == 1,
                   "val:%d\n", val);
    #endif
  }
}