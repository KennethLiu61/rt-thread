#include "atomic_sys_gen_cmd.h"
#include "bd_reg_value.h"
#include "gdma_reg_value.h"
#include "sdma_reg_def.h"
#include "sdma_reg_value.h"
#include "cdma_reg_value.h"
#include "gen_cmd.h"
#include "atomic_gen_cmd.h"

void atomic_bd_end_gen_cmd(CMD_ID_NODE *pid_node) {
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_END");
  pid_node->cur_op_cycle = 0;
#endif
  FW_DBG("%s\n", __func__);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_END << 45) |
          ((u64)bd_power_step() << 59) |
          ((u64)1ull << 63);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_END, 1, pid_node, &high, &low, 1);
}

void atomic_bd_rand_seed_gen_cmd(int thread_id, CMD_ID_NODE *pid_node) {
  #ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_RAND_SEED");
  pid_node->cur_op_cycle = 0;
#endif
  FW_DBG("%s\n", __func__);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR + thread_id * BD_THREAD_OFFSET;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_RANDOM_SEED << 45) |
          ((u64)bd_power_step() << 59);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_RANDOM_SEED, 0, pid_node, &high, &low, 1);
}

void atomic_bd_nop_gen_cmd(CMD_ID_NODE *pid_node) {
    FW_DBG("%s\n", __func__);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
    u64 high = 0, low = 0;
    BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
      low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SYS << 41) |
            ((u64)BD_SYS_NOP << 45);
      WRITE_CMD_EX(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_BD(pid_node)
    profile_time_set_node(ENGINE_BD, SYS,
        BD_SYS_NOP, 0, pid_node, &high, &low, 1);
}

// software power boot, this is useless for software
void atomic_bd_spb_gen_cmd(CMD_ID_NODE *pid_node) {
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_SPB");
  pid_node->cur_op_cycle = 0;
#endif
  FW_DBG("%s\n", __func__);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_SPB << 45) |
          ((u64)bd_power_step() << 59);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_SPB, 0, pid_node, &high, &low, 1);
}

void atomic_bd_fork_gen_cmd(CMD_ID_NODE *pid_node){
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_FORK");
#endif
  FW_DBG("%s\n", __func__);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull<<37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_FORK << 45);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_FORK, 0, pid_node, &high, &low, 1);
}

void atomic_bd_join_gen_cmd(CMD_ID_NODE *pid_node){
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_JOIN");
#endif
  FW_DBG("%s\n", __func__);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull<<37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_JOIN << 45);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_JOIN, 0, pid_node, &high, &low, 1);
}

void atomic_bd_exit_gen_cmd(CMD_ID_NODE *pid_node){
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_EXIT");
#endif
  FW_DBG("%s\n", __func__);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR + SLAVE_THREAD * BD_THREAD_OFFSET;
  u64 high = 0, low = 0;
  BEGIN_FAST_GEN_CMD_BD(SLAVE_THREAD)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull<<37) |
          ((u64)SYS << 41) |
          ((u64)BD_SYS_EXIT << 45);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_EXIT, 0, pid_node, &high, &low, 1);
}

// this atomic can't sync gdma id
void atomic_bd_swr_gen_cmd(
    u32 src_addr, // local addr
    int src_is_const,
    u64 src_const_value,
    int short_term_valid, // 0: clear valid bit, others: set valid bit
    int long_term_valid,  // 0: clear valid bit, others: set valid bit
    BD_SYS_TYPE sys_type, int thread_id, CMD_ID_NODE *pid_node) {
  FW_DBG("%s, src_addr:%u, src_is_const:%d, src_value:%llu, "
         "short_valid:%d, long_valid:%d, sys_type:%d\n",
         __func__, src_addr, src_is_const, src_const_value, short_term_valid,
         long_term_valid, sys_type);

#ifdef USING_CMODEL
  ASSERT(sys_type == BD_SYS_SWR || sys_type == BD_SYS_SWR_FROM_LMEM ||
         sys_type == BD_SYS_SWR_COL_FROM_LMEM);
  if (sys_type == BD_SYS_SWR) {
    ASSERT(src_is_const);
  } else if (sys_type == BD_SYS_SWR_FROM_LMEM) {
    ASSERT(src_addr % ALIGN_BYTES == 0);
  } else if (sys_type == BD_SYS_SWR_COL_FROM_LMEM) {
    ASSERT(src_addr % sizeof(char) == 0);
    ASSERT(src_addr >> LOCAL_MEM_ADDRWIDTH == 0);
  }
#endif

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR + thread_id * BD_THREAD_OFFSET;
  u64 high = 0;
  u64 low = 0;
  if (sys_type == BD_SYS_SWR)
    high = src_const_value;
  else
    memcpy(&high, &src_addr, sizeof(u32));
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)SYS << 41) |
          ((u64)sys_type << 45) |
          ((short_term_valid ? 1ull : 0ull) << 50) |
          ((long_term_valid ? 1ull : 0ull) << 51);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node);
  profile_time_set_node(ENGINE_BD, SYS,
      sys_type, long_term_valid << 1 | short_term_valid, pid_node, &high, &low, 1);
}

// gen command for TIU general control register
void atomic_bd_trwr_gen_cmd(const u32 *value, const int *indice, int count, int thread_id,
                            CMD_ID_NODE *pid_node) {
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_SYS_TRWR");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR + thread_id * BD_THREAD_OFFSET;
  int done = 0;
  while (done < count) {
    u32 register_value[3] = {0};
    u8 register_idx[3] = {0x7f, 0x7f, 0x7f};
    for (int idx = done; idx < count; idx++) {
      if (idx - done >= 3)
        break;
      ;
      register_idx[idx - done] = indice[idx];
      register_value[idx - done] = value[idx];
#ifdef USING_CMODEL
      ASSERT(indice[idx] < (1 << 8));
      // only these register idx is valid for bm1686
      ASSERT(indice[idx] == 5 || indice[idx] == 6 || indice[idx] == 32 ||
             indice[idx] == 33 || indice[idx] == 127 || indice[idx] == 128 ||
             indice[idx] == 129);
#endif
      if (indice[idx] == 129) {
        ASSERT_FS_INFO(value[idx] < (1u << 24),
                       "indice 126 is tpu_id, only 24bit valid, %d",
                       value[idx]);
      }
    }
    done += 3;

    BEGIN_FAST_GEN_CMD_BD(thread_id)
      u64 low = 0, high = 0;
      low = (u64)register_value[0] |
            ((u64)register_idx[0] << 32) |
            ((u64)SYS_TRWR << 41) |
            ((u64)register_idx[1] << 48) |
            ((u64)register_idx[2] << 56);
      high = (u64)register_value[1] | ((u64)register_value[2] << 32);
    WRITE_CMD_EX(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_BD(pid_node)
    profile_time_set_node(ENGINE_BD, SYS_TRWR,
      0, 0, pid_node, &high, &low, 1);
  }
}

void atomic_set_bdid_gen_cmd(u32 bdc_id, CMD_ID_NODE *pid_node) {
  ASSERT(bdc_id < (1 << 24));
  int indice = 129;
  atomic_bd_trwr_gen_cmd(&bdc_id, &indice, 1, MASTER_THREAD, pid_node);
  pid_node->bd_cmd_id = bdc_id;
}
/// @brief TPU work mode
/// @param mode 0: standalone mode, 1: combination mode
/// @param pid_node
void atomic_set_tpu_mode_gen_cmd(int32_t mode, CMD_ID_NODE *pid_node) {
  ASSERT(mode == 0 || mode == 1);
  int32_t indice = 128;
  atomic_bd_trwr_gen_cmd((uint32_t *)&mode, &indice, 1, MASTER_THREAD, pid_node);
}

/* gdma sys instruction */
void atomic_gdma_end_gen_cmd(CMD_ID_NODE *pid_node) {
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "GDMA_SYS_END");
    pid_node->cur_op_cycle = 0;
#endif
    FW_DBG("%s\n", __func__);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
        u64 low = 1ull | (1ull << 3) | ((u64)GDMA_SYS << 32) |
                  ((u64)GDMA_SYS_END << 37);
        u64 high = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    WRITE_CMD_EX(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_SYS,
      GDMA_SYS_END, 0, pid_node, &high, &low, 1);
}
void atomic_gdma_nop_gen_cmd(CMD_ID_NODE *pid_node, int thread_id) {
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "GDMA_SYS_NOP");
    pid_node->cur_op_cycle = 0;
#endif
    FW_DBG("%s\n", __func__);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR + thread_id * GDMA_THREAD_OFFSET;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low = (1ull << 3) | ((u64)GDMA_SYS << 32) |
            ((u64)GDMA_SYS_NOP << 37);
        u64 high = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    WRITE_CMD_EX(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_SYS,
      GDMA_SYS_NOP, 0, pid_node, &high, &low, 1);
}

void atomic_set_gdma_id_gen_cmd(u32 gdma_id, CMD_ID_NODE *pid_node) {
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "GDMA_SYS_TRWR");
    pid_node->cur_op_cycle = 0;
#endif
    FW_DBG("%s, gdma_id: %d\n", __func__, gdma_id);
    ASSERT(gdma_id < (1 << 24));
    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low = 0, high = 0;
    BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
        low = (1ull << 3) |
              ((u64)GDMA_SYS << 32) |
              ((u64)GDMA_SYS_TRWR << 37);
        high = ((u64)pid_node->bd_cmd_id & 0xfffff) |
               ((u64)0 << 22) | // register idx
               ((u64)gdma_id << 32);
        WRITE_CMD_EX(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_GDMA(pid_node)
    pid_node->gdma_cmd_id = gdma_id;
    profile_time_set_node(ENGINE_GDMA, GDMA_SYS,
      GDMA_SYS_TRWR, 0, pid_node, &high, &low, 1);
}

void atomic_set_sdma_id_gen_cmd(u32 sdma_id, CMD_ID_NODE *pid_node, int port_id) {
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "SDMA_SYS_TRWR");
    pid_node->cur_op_cycle = 0;
#endif
    FW_DBG("%s, sdma_id: %d\n", __func__, sdma_id);
    ASSERT(sdma_id < (1 << 24));
    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    u64 low = 0, high = 0;
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        low = (1ull << 3) |
              ((u64)SDMA_SYS << 32) |
              ((u64)SDMA_SYS_TRWR << 37);
        high = ((u64)0 << 22) | // register idx
               ((u64)sdma_id << 32);
        WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
    pid_node->sdma_cmd_id = sdma_id;
}

void atomic_sdma_end_gen_cmd(CMD_ID_NODE *pid_node, int port_id) {
    FW_DBG("%s\n", __func__);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low = 1ull | (1ull << 3) | ((u64)SDMA_SYS << 32) |
                  ((u64)SDMA_SYS_END << 37);
        u64 high = 0;
    WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_SYS,
        SDMA_SYS_END, 0, pid_node, &high, &low, 1);
}

void atomic_sdma_nop_gen_cmd(CMD_ID_NODE *pid_node, int port_id) {
    FW_DBG("%s\n", __func__);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low = (1ull << 3) | ((u64)SDMA_SYS << 32) |
            ((u64)SDMA_SYS_NOP << 37);
        u64 high = 0;
    WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_SYS,
        SDMA_SYS_NOP, 0, pid_node, &high, &low, 1);
}

/* send or wait message */
static void bd_send_msg_gen_cmd(int32_t msg_id, int32_t wait_cnt,
                                CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "BD_SEND_MSG");
    pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    u64 low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) | ((u64)SYS << 41) | ((u64)BD_SYS_SEND_MSG << 45) | ((u64)bd_power_step() << 59);
    u64 high = ((u64)(msg_id & 0x1ff)) | ((u64)(wait_cnt & 0x7f) << 16);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_SEND_MSG, wait_cnt, pid_node, &high, &low, 1);
}
static void bd_wait_msg_gen_cmd(int32_t msg_id, int32_t send_cnt,
                                CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "BD_WAIT_MSG");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD_BD(MASTER_THREAD)
    u64 low = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) | ((u64)SYS << 41) | ((u64)BD_SYS_WAIT_MSG << 45) | ((u64)bd_power_step() << 59);
    u64 high = ((u64)(msg_id & 0x1ff)) | ((u64)(send_cnt & 0x7f) << 16);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_BD(pid_node)
  profile_time_set_node(ENGINE_BD, SYS,
      BD_SYS_WAIT_MSG, send_cnt, pid_node, &high, &low, 1);
}

static void gdma_send_msg_gen_cmd(int32_t msg_id,int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "GDMA_SEND_MSG");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
    u64 low = (1ull << 3) | ((u64)GDMA_SYS << 32) | ((u64)GDMA_SYS_SEND_MSG << 37);
    u64 high = ((u64)pid_node->bd_cmd_id & 0xfffff) |
               ((u64)(msg_id & 0x1ff) << 32) |
               ((u64)(wait_cnt & 0x7f) << 41);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_GDMA(pid_node)
  profile_time_set_node(ENGINE_GDMA, GDMA_SYS,
      GDMA_SYS_SEND_MSG, wait_cnt, pid_node, &high, &low, 1);
}
static void gdma_wait_msg_gen_cmd(int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "GDMA_WAIT_MSG");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
    u64 low = (1ull << 3) | ((u64)GDMA_SYS << 32) | ((u64)GDMA_SYS_WAIT_MSG << 37);
    u64 high = ((u64)pid_node->bd_cmd_id & 0xfffff) |
               ((u64)(msg_id & 0x1ff) << 32) |
               ((u64)(send_cnt & 0x7f) << 41);
    WRITE_CMD_EX(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_GDMA(pid_node)
  profile_time_set_node(ENGINE_GDMA, GDMA_SYS,
      GDMA_SYS_WAIT_MSG, send_cnt, pid_node, &high, &low, 1);
}

static void hau_send_msg_gen_cmd(int32_t msg_id, int32_t wait_cnt,
                                 CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "HAU_SEND_MSG");
    pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = HAU_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD(HAU)
    u64 high = ((u64)HAU_SEND_MSG) | // send message
               ((u64)(wait_cnt & 0x7f) << 8) |
               ((u64)(msg_id & 0x1ff) << 15);
    WRITE_CMD_EX(reg_addr, 2, high, 0ull);
  END_FAST_GEN_CMD(HAU, pid_node)
}
static void hau_wait_msg_gen_cmd(int32_t msg_id, int32_t send_cnt,
                                 CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
#ifdef SG_STAS_GEN
    strcpy(pid_node->cmd_name, "HAU_WAIT_MSG");
    pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = HAU_CMD_BASE_ADDR;
  BEGIN_FAST_GEN_CMD(HAU)
    u64 high = ((u64)HAU_WAIT_MSG) | // wait message
               ((u64)(send_cnt & 0x7f) << 8) |
               ((u64)(msg_id & 0x1ff) << 15);
    WRITE_CMD_EX(reg_addr, 2, high, 0ull);
  END_FAST_GEN_CMD(HAU, pid_node)
}

static void sdma_send_msg_gen_cmd(int32_t msg_id, int32_t wait_cnt,
                                  int port_id, CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "SDMA_SEND_MSG");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
  BEGIN_FAST_GEN_CMD_SDMA(port_id)
    u64 low = (1ull << 3) | ((u64)SDMA_SYS << 32) | ((u64)SDMA_SYS_SEND_MSG << 37);
    u64 high = ((u64)(wait_cnt & 0x7f) << 41) | ((u64)(msg_id & 0x1ff) << 32);
    WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_SDMA(pid_node)
  profile_time_set_node(fast_cmd->sdma_type, SDMA_SYS,
        SDMA_SYS_SEND_MSG, wait_cnt, pid_node, &high, &low, 1);
}

static void sdma_wait_msg_gen_cmd(int32_t msg_id, int32_t send_cnt,
                                  int port_id, CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
#ifdef SG_STAS_GEN
  strcpy(pid_node->cmd_name, "SDMA_WAIT_MSG");
  pid_node->cur_op_cycle = 0;
#endif

  const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
  BEGIN_FAST_GEN_CMD_SDMA(port_id)
    u64 low = (1ull << 3) | ((u64)SDMA_SYS << 32) | ((u64)SDMA_SYS_WAIT_MSG << 37);
    u64 high =((u64)(send_cnt & 0x7f) << 41) | ((u64)(msg_id & 0x1ff) << 32);
    WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
  END_FAST_GEN_CMD_SDMA(pid_node)
  profile_time_set_node(fast_cmd->sdma_type, SDMA_SYS,
        SDMA_SYS_WAIT_MSG, send_cnt, pid_node, &high, &low, 1);
}

void atomic_send_msg_gen_cmd(ENGINE_TYPE eng_type, int32_t msg_id,
                             int32_t wait_cnt, int thread_id, CMD_ID_NODE *pid_node) {
  ASSERT(msg_id >= 0 && msg_id < (1 << MSG_ID_WIDTH));
  ASSERT(wait_cnt > 0 && wait_cnt < (1 << MSG_CNT_BIT));
  //ASSERT(!pid_node->in_parallel_state);
  if (eng_type == ENGINE_BD) {
    bd_send_msg_gen_cmd(msg_id, wait_cnt, pid_node);
  } else if (eng_type == ENGINE_GDMA) {
    gdma_send_msg_gen_cmd(msg_id, wait_cnt, pid_node);
  } else if (eng_type == ENGINE_SDMA || eng_type == ENGINE_VSDMA) {
    sdma_send_msg_gen_cmd(msg_id, wait_cnt, thread_id, pid_node);
  } else if (eng_type == ENGINE_HAU) {
    hau_send_msg_gen_cmd(msg_id, wait_cnt, pid_node);
  } else {
    ASSERT(0);
  }
}

void atomic_wait_msg_gen_cmd(ENGINE_TYPE eng_type, int32_t msg_id,
                             int32_t send_cnt, int thread_id, CMD_ID_NODE *pid_node) {
  ASSERT(msg_id >= 0 && msg_id < (1 << MSG_ID_WIDTH));
  ASSERT(send_cnt > 0 && send_cnt < (1 << MSG_CNT_BIT));
  //ASSERT(!pid_node->in_parallel_state);
  if (eng_type == ENGINE_BD) {
    bd_wait_msg_gen_cmd(msg_id, send_cnt, pid_node);
  } else if (eng_type == ENGINE_GDMA) {
    gdma_wait_msg_gen_cmd(msg_id, send_cnt, pid_node);
  } else if (eng_type == ENGINE_SDMA || eng_type == ENGINE_VSDMA) {
    sdma_wait_msg_gen_cmd(msg_id, send_cnt, thread_id, pid_node);
  } else if (eng_type == ENGINE_HAU) {
    hau_wait_msg_gen_cmd(msg_id, send_cnt, pid_node);
  } else {
    ASSERT(0);
  }
}

void atomic_cdma_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node) {
  atomic_cdma_tx_send_msg_gen_cmd(port, msg_id, wait_cnt, pid_node);
  atomic_cdma_rx_send_msg_gen_cmd(port, msg_id, wait_cnt, pid_node);
}
void atomic_cdma_tx_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_TX_SEND_MSG << 8) |
              ((u64)(wait_cnt & 0x7f) << 48) | ((u64)(msg_id & 0x1ff) << 32);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_TX_SEND_MSG, (wait_cnt & 0x7f) | port << 7, pid_node, &high, &low, 1);
}
void atomic_cdma_rx_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, wait_cnt:%d\n", __func__, msg_id, wait_cnt);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_RX_SEND_MSG << 8) |
              ((u64)(wait_cnt & 0x7f) << 48) | ((u64)(msg_id & 0x1ff) << 32);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_RX_SEND_MSG, (wait_cnt & 0x7f) | port << 7, pid_node, &high, &low, 1);
}

void atomic_cdma_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node) {
  atomic_cdma_tx_wait_msg_gen_cmd(port, msg_id, send_cnt, pid_node);
  atomic_cdma_rx_wait_msg_gen_cmd(port, msg_id, send_cnt, pid_node);
}

void atomic_cdma_tx_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_TX_WAIT_MSG << 8) |
              ((u64)(send_cnt & 0x7f) << 48) | ((u64)(msg_id & 0x1ff) << 32);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_TX_WAIT_MSG, (send_cnt & 0x7f) | port << 7, pid_node, &high, &low, 1);
}

void atomic_cdma_rx_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node) {
  FW_DBG("%s: msg_id:%d, send_cnt:%d\n", __func__, msg_id, send_cnt);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_RX_WAIT_MSG << 8) |
              ((u64)(send_cnt & 0x7f) << 48) | ((u64)(msg_id & 0x1ff) << 32);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_RX_WAIT_MSG, (send_cnt & 0x7f) | port << 7, pid_node, &high, &low, 1);
}

void atomic_cdma_end_gen_cmd(int32_t port, CMD_ID_NODE *pid_node) {
  FW_DBG("%s: port:%d\n", __func__, port);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = 1ull | ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_END << 8);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_END, port << 7, pid_node, &high, &low, 1);
}

void atomic_cdma_nop_gen_cmd(int32_t port, CMD_ID_NODE *pid_node) {
  FW_DBG("%s: port:%d\n", __func__, port);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_NOP << 8);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_NOP, port << 7, pid_node, &high, &low, 1);
}

void atomic_set_cdma_id_gen_cmd(int32_t port, u32 cdma_id, CMD_ID_NODE *pid_node) {
  FW_DBG("%s: port:%d, cdma_id:%d\n", __func__, port, cdma_id);
  BEGIN_FAST_GEN_CMD_CDMA(port)
    u64 low = ((u64)CDMA_SYS << 4) | ((u64)CDMA_SYS_TRWR << 8) |((u64)(cdma_id) << 32);
    u64 high = 0;
    WRITE_CMD_CDMA(port, 0, high, low);
  END_FAST_GEN_CMD_CDMA(port, pid_node)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, CDMA_SYS,
      CDMA_SYS_TRWR,  port << 7, pid_node, &high, &low, 1);
}

void atomic_set_base_ddr(const int *base_idx, const u64 *base_addr, int count,
                         ENGINE_TYPE eng_type) {
  ASSERT(eng_type == ENGINE_GDMA || eng_type == ENGINE_HAU ||
    eng_type == ENGINE_SDMA || eng_type == ENGINE_CDMA || eng_type == ENGINE_VSDMA);

  const reg_id_t gdma_cfg_val = GDMA_ID_CFG_BASE_DDR0;
  const reg_id_t sort_cfg_val = SORT_ID_BASE_ADDR_ID0;
  const reg_id_t sdma_cfg_val = SDMA_ID_CFG_BASE_DDR0;
  const reg_id_t cdma_cfg_val = CDMA_ID_BASE_ADDR_ID0;
  for (int i = 0; i < count; i++) {
    CORE_PRINT("[%d] %d: set base_idx=%d, base_addr=0x%llx\n", CORE_ID, i, base_idx[i], base_addr[i]);
    ASSERT(base_idx[i] >= 0 && base_idx[i] <= (int)TAG_MASK);
    ASSERT((base_addr[i] & 0xff) == 0);
    if (eng_type == ENGINE_GDMA) {
      WRITE_REG((GDMA_ENGINE_MAIN_CTRL + gdma_cfg_val.where / 8 + base_idx[i] * 4), base_addr[i] >> 8, NODECHIP_REG);
    } else if (eng_type == ENGINE_HAU) {
      WRITE_REG((HAU_ENGINE_MAIN_CTRL + sort_cfg_val.where / 8 + base_idx[i] * 4), base_addr[i] >> 8, NODECHIP_REG);
    } else if (eng_type == ENGINE_SDMA) {
      WRITE_REG((SDMA_ENGINE_MAIN_CTRL + sdma_cfg_val.where / 8 + base_idx[i] * 4), base_addr[i] >> 8, NODECHIP_REG);
    } else if (eng_type == ENGINE_VSDMA) {
      WRITE_REG((VSDMA_ENGINE_MAIN_CTRL(CORE_ID) + sdma_cfg_val.where / 8 + base_idx[i] * 4), base_addr[i] >> 8, NODECHIP_REG);
    } else if (eng_type == ENGINE_CDMA) {
      for (int j = 0; j < CDMA_NUM; j++) {
        WRITE_REG((CDMA_ENGINE_MAIN_CTRL(j) + cdma_cfg_val.where / 8 + base_idx[i] * 4), base_addr[i] >> 8, NODECHIP_REG);
      }
    } else {
      ASSERT(0);
    }
  }
}

void atomic_cdma_port_set_base_ddr(const int *base_idx, const u64 *base_addr,
                                    int count, int cdma_port) {
  const reg_id_t cdma_cfg_val = CDMA_ID_BASE_ADDR_ID0;
  for (int i = 0; i < count; i++) {
    CORE_PRINT("%d: cdma[%d] set base_idx=%d, base_addr=0x%llx\n", i,
               cdma_port, base_idx[i], base_addr[i]);
    ASSERT(base_idx[i] >= 0 && base_idx[i] <= (int)TAG_MASK);
    ASSERT((base_addr[i] & 0xff) == 0);
    WRITE_REG((CDMA_ENGINE_MAIN_CTRL(cdma_port) + cdma_cfg_val.where / 8 +
               base_idx[i] * 4),
              base_addr[i] >> 8, NODECHIP_REG);
  }
}

void atomic_set_gdma_random_mask_seed(uint64_t seed) {
  u32 low = seed & 0xffffffff;
  u32 high = (seed >> 32) & 0xffffffff;
  reg_id_t l = GDMA_ID_CFG_SEED_L32;
  reg_id_t h = GDMA_ID_CFG_SEED_H32;
  reg_id_t e = GDMA_ID_CFG_SEED_EN;
  WRITE_REG(GDMA_ENGINE_MAIN_CTRL + l.where / 8, low, NODECHIP_REG);
  WRITE_REG(GDMA_ENGINE_MAIN_CTRL + h.where / 8, high, NODECHIP_REG);
  // enable always 1 align to ic
  WRITE_REG(GDMA_ENGINE_MAIN_CTRL + e.where / 8, 1, NODECHIP_REG);
}

void atomic_set_base_msg_id(int32_t base_idx, ENGINE_TYPE eng_type) {
#ifdef USING_EDA
  base_idx = 0;
#endif
  ASSERT(base_idx >=0 && base_idx < ((1 << MSG_ID_WIDTH) - 1));
  if (eng_type == ENGINE_BD) {
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_BASE_MSGID, base_idx);
  } else if (eng_type == ENGINE_GDMA) {
    FW_REG_ID_WRITE(GDMA_ENGINE_MAIN_CTRL, GDMA_ID_CFG_BASE_MSGID, base_idx);
  } else if (eng_type == ENGINE_HAU) {
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_BASE_MSGID, base_idx);
  } else if (eng_type == ENGINE_SDMA) {
    FW_REG_ID_WRITE(SDMA_ENGINE_MAIN_CTRL, SDMA_ID_CFG_BASE_MSGID, base_idx);
  } else if (eng_type == ENGINE_VSDMA) {
    FW_REG_ID_WRITE(VSDMA_ENGINE_MAIN_CTRL(CORE_ID), VSDMA_ID_CFG_BASE_MSGID, base_idx);
  } else if (eng_type == ENGINE_CDMA) {
#ifndef DISABLE_CDMA
    for (int i = 0; i < CDMA_NUM; i++) {
      FW_REG_ID_WRITE(CDMA_ENGINE_MAIN_CTRL(i), CDMA_ID_CFG_BASE_MSGID, base_idx);
    }
#endif
  } else {
    ASSERT_FS_INFO(0, "unsupported engine type:%d\n", eng_type);
  }
}
void atomic_cdma_port_set_base_msg_id(int32_t base_idx, int32_t port) {
#ifdef USING_EDA
  base_idx = 0;
#endif
  ASSERT(base_idx >=0 && base_idx < ((1 << MSG_ID_WIDTH) - 1));
  FW_REG_ID_WRITE(CDMA_ENGINE_MAIN_CTRL(port), CDMA_ID_CFG_BASE_MSGID, base_idx);
  printf("!!! cdma [%d] write base msg_id %d\n", port, base_idx);
}

void atomic_set_bd_random_gen_seed(uint64_t seed) {
  u32 low = seed & 0xFFFFFFFF;
  u32 high = (seed >> 32) & 0xFFFFFFFF;
  reg_id_t l = BD_ID_CFG_SEED_INITIAL_L32;
  reg_id_t h = BD_ID_CFG_SEED_INITIAL_H32;
  WRITE_REG(BD_ENGINE_MAIN_CTRL + l.where / 8, low, NODECHIP_REG);
  WRITE_REG(BD_ENGINE_MAIN_CTRL + h.where / 8, high, NODECHIP_REG);
}

int atomic_cdma_send_cmodel_gen_cmd(uint8_t dst_chipid,
                             uint32_t src_base_addr,
                             uint32_t src_n, uint32_t src_c, uint32_t src_h, uint32_t src_w,
                             uint32_t src_n_stride, uint32_t src_c_stride, uint32_t src_h_stride,
                             uint8_t opcode,
                             uint8_t dtype)
{
  return 0;
}

int atomic_cdma_receive_gen_cmd(uint8_t src_chipid,
                                uint32_t dst_base_addr,
                                uint32_t dst_n, uint32_t dst_c, uint32_t dst_h, uint32_t dst_w,
                                uint32_t dst_n_stride, uint32_t dst_c_stride, uint32_t dst_h_stride,
                                uint8_t dst_mem_type)
{
  return 0;
}

void atomic_gdma_debug(int32_t mode) {
  //mode:0 is no debug; 1 is singlestep debug mode; 2 is breakpoint debug mode;
  reg_id_t debug_mode = GDMA_ID_CFG_MST_DBG_MODE;
  reg_id_t exec = GDMA_ID_CFG_MST_DBG_EXEC_ENABLE;
  reg_id_t singlestep_disable = GDMA_ID_CFG_MST_SINGLESTEP_DISABLE;
  reg_id_t breakpoint_disable = GDMA_ID_CFG_MST_BREAKPOINT_DISABLE;
  reg_id_t singlestep_stat = GDMA_ID_CFG_MST_IRQ_SINGLESTEP;
  reg_id_t breakpoint_stat = GDMA_ID_CFG_MST_IRQ_BREAKPOINT;
  WRITE_REG(GDMA_ENGINE_MAIN_CTRL + debug_mode.where / 8, mode, NODECHIP_REG);
  WRITE_REG(GDMA_ENGINE_MAIN_CTRL + exec.where / 8, (int)(mode != 0), NODECHIP_REG);
  if (mode == 1){
    READ_REG(GDMA_ENGINE_MAIN_CTRL + singlestep_stat.where / 8);
    READ_REG(GDMA_ENGINE_MAIN_CTRL + singlestep_disable.where / 8);
  } else if (mode == 2){
    READ_REG(GDMA_ENGINE_MAIN_CTRL + breakpoint_stat.where / 8);
    READ_REG(GDMA_ENGINE_MAIN_CTRL + breakpoint_disable.where / 8);
  } else {
    READ_REG(GDMA_ENGINE_MAIN_CTRL + singlestep_stat.where / 8);
    READ_REG(GDMA_ENGINE_MAIN_CTRL + breakpoint_stat.where / 8);
    READ_REG(GDMA_ENGINE_MAIN_CTRL + singlestep_disable.where / 8);
    READ_REG(GDMA_ENGINE_MAIN_CTRL + breakpoint_disable.where / 8);
  }
}

void atomic_bd_debug(int32_t mode) {
  //mode:0 is no debug; 1 is singlestep debug mode; 2 is breakpoint debug mode;
  reg_id_t debug_mode = BD_ID_CFG_DBG_MODE;
  reg_id_t exec = BD_ID_CFG_DBG_EXEC;
  reg_id_t singlestep_stat = BD_ID_IRQ_DBG_SINGLE_STEP_STAT;
  reg_id_t breakpoint_stat = BD_ID_IRQ_DBG_BREAKPOINT_STAT;
  WRITE_REG(BD_ENGINE_MAIN_CTRL + debug_mode.where / 8, mode, NODECHIP_REG);
  WRITE_REG(BD_ENGINE_MAIN_CTRL + exec.where / 8, (int)(mode != 0), NODECHIP_REG);
  if (mode == 1){
    READ_REG(BD_ENGINE_MAIN_CTRL + singlestep_stat.where / 8);
  } else if (mode == 2){
    READ_REG(BD_ENGINE_MAIN_CTRL + breakpoint_stat.where / 8);
  } else {
    READ_REG(BD_ENGINE_MAIN_CTRL + singlestep_stat.where / 8);
    READ_REG(BD_ENGINE_MAIN_CTRL + breakpoint_stat.where / 8);
  }
}

void atomic_sdma_debug(int32_t mode, ENGINE_TYPE eng_type, int32_t port_id) {
  //mode:0 is no debug; 1 is singlestep debug mode; 2 is breakpoint debug mode;
  reg_id_t mst_debug_mode = SDMA_ID_CFG_MST_DBG_MODE;
  reg_id_t slv_debug_mode = SDMA_ID_CFG_SLV_DBG_MODE;
  reg_id_t mst_exec = SDMA_ID_CFG_MST_DBG_EXEC_ENABLE;
  reg_id_t slv_exec = SDMA_ID_CFG_SLV_DBG_EXEC_ENABLE;
  reg_id_t mst_singlestep_stat = SDMA_ID_CFG_MST_IRQ_SINGLESTEP;
  reg_id_t mst_breakpoint_stat = SDMA_ID_CFG_MST_IRQ_BREAKPOINT;
  reg_id_t slv_singlestep_stat = SDMA_ID_CFG_SLV_IRQ_SINGLESTEP;
  reg_id_t slv_breakpoint_stat = SDMA_ID_CFG_SLV_IRQ_BREAKPOINT;
  reg_id_t mst_singlestep_disable = SDMA_ID_CFG_MST_SINGLESTEP_DISABLE;
  reg_id_t mst_breakpoint_disable = SDMA_ID_CFG_SLV_SINGLESTEP_DISABLE;
  reg_id_t slv_singlestep_disable = SDMA_ID_CFG_MST_BREAKPOINT_DISABLE;
  reg_id_t slv_breakpoint_disable = SDMA_ID_CFG_SLV_BREAKPOINT_DISABLE;
  if (eng_type == ENGINE_SDMA) {
    WRITE_REG(SDMA_ENGINE_MAIN_CTRL + mst_debug_mode.where / 8, mode, NODECHIP_REG);
    WRITE_REG(SDMA_ENGINE_MAIN_CTRL + mst_exec.where / 8, (int)(mode != 0), NODECHIP_REG);
    if (mode == 1){
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_singlestep_stat.where / 8);
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_singlestep_disable.where / 8);
    } else if (mode == 2){
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_breakpoint_stat.where / 8);
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_breakpoint_disable.where / 8);
    } else {
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_singlestep_stat.where / 8);
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_breakpoint_stat.where / 8);
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_singlestep_disable.where / 8);
      READ_REG(SDMA_ENGINE_MAIN_CTRL + mst_breakpoint_disable.where / 8);
    }
  } else if (eng_type == ENGINE_VSDMA) {
    WRITE_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_debug_mode.where / 8, mode, NODECHIP_REG);
    WRITE_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_exec.where / 8, (int)(mode != 0), NODECHIP_REG);
    if (mode == 1){
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_singlestep_stat.where / 8);
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_singlestep_disable.where / 8);
    } else if (mode == 2){
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_breakpoint_stat.where / 8);
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_breakpoint_disable.where / 8);
    } else {
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_singlestep_stat.where / 8);
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_breakpoint_stat.where / 8);
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_singlestep_disable.where / 8);
      READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) + slv_breakpoint_disable.where / 8);
    }
  }
}
