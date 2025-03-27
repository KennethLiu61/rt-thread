#pragma once
#include "firmware_common.h"
#include "gen_cmd_utils.h"
#include "sort_reg_def.h"

static THREAD gen_cmd_t g_hau_gen_cmd;
static THREAD unsigned char g_hau_gen_cmd_initialized = __FALSE__;
static THREAD int g_hau_buf_depth = 0;

static inline P_COMMAND hau_cmd_buf(int thread_id) {
#ifdef USING_CMODEL
    int sz = HAU_REG_COUNT;
    poll_hau_engine_fifo();
    unsigned int *p_cmd = (unsigned int *)malloc(sz * sizeof(unsigned int));
    memset(p_cmd, 0, sz * sizeof(unsigned int));
    (void)(g_hau_buf_depth);
#else
    unsigned int *p_cmd;
    p_cmd = (unsigned int*)map_to_kaddr(HAU_CMD_BASE_ADDR);

    reg_id_t reg_value = SORT_ID_PIO_BUF_DEPTH;
    volatile u32 *rd_addr =
        (volatile u32 *)map_to_kaddr(HAU_ENGINE_MAIN_CTRL + ((reg_value.where >> 5) << 2));
    reg_value.where &= 31;
    u32 mask_val = (u32)(1 << reg_value.len) - 1;
    while (g_hau_buf_depth == 0) {
      volatile u32 rd_data = ((*rd_addr) >> reg_value.where) & mask_val;
      if (rd_data != 0) {
        g_hau_buf_depth = rd_data;
        break;
      }
    }
    g_hau_buf_depth -= 1;
#endif
    return (P_COMMAND)p_cmd;
}

static inline void hau_cmd_id_proc(unsigned int *id, CMD_ID_NODE *pid_node, int thread_id/*unused for hau*/) {
    HAU_CMD_ID_PROC(id[0], id[1], id[2]);
}

static inline void hau_common(cmd_t *cmd) {
    reg_pack_t pio_enable = SORT_PACK_PIO_ENABLE(__TRUE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &pio_enable);
    reg_pack_t int_enable = SORT_PACK_INT_ENABLE(__FALSE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &int_enable);
    reg_pack_t eod = SORT_PACK_EOD(__TRUE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &eod);
    reg_pack_t int_sts_en = SORT_PACK_INT_STS_EN(__TRUE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &int_sts_en);
    reg_pack_t enable_syncid = SORT_PACK_ENABLE_SYNCID(__FALSE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &enable_syncid);
}

static inline void hau_start(cmd_t *cmd) {
    reg_pack_t dscp_vld = SORT_PACK_DSCP_VLD(__TRUE__);
    write_cmd(HAU_CMD_BASE_ADDR, cmd, &dscp_vld);
}

static gen_cmd_t *hau_gen_cmd() {
    if (g_hau_gen_cmd_initialized == __FALSE__) {
        g_hau_gen_cmd.buf = hau_cmd_buf;
        g_hau_gen_cmd.id_proc = hau_cmd_id_proc;
        g_hau_gen_cmd.common = hau_common;
        g_hau_gen_cmd.start = hau_start;
        g_hau_gen_cmd_initialized = __TRUE__;
    }
    return &g_hau_gen_cmd;
}

static inline gen_cmd_t* HAU_begin_fast_gen_cmd(u32 ids[3], CMD_ID_NODE *pid_node) {
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, __TRUE__);
    gen_cmd_t *fast_cmd = hau_gen_cmd();
    fast_cmd->cmd.buf = hau_cmd_buf(0);
    hau_cmd_id_proc(ids, pid_node, 0);
    return fast_cmd;
}

static inline void HAU_end_fast_gen_cmd(gen_cmd_t *fast_cmd, CMD_ID_NODE *pid_node) {
    fast_cmd->start(&fast_cmd->cmd);
    call_atomic(fast_cmd->cmd.buf, ENGINE_HAU, pid_node, 0);
    fast_cmd->cmd.buf = NULL;
}
