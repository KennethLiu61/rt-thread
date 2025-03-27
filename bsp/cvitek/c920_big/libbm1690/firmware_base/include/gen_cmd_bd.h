#pragma once
#include "firmware_common.h"
#include "gen_cmd_utils.h"
#include "bd_reg_def.h"

static THREAD gen_cmd_t g_bd_gen_cmd;
static THREAD unsigned char g_bd_gen_cmd_initialized = __FALSE__;
static THREAD u64 g_bd_lane = { 0xffffffffffffffff };
static THREAD int g_bd_buf_depth = 0;

void record_fifo_depth(int index, uint32_t v);

static inline P_COMMAND bd_cmd_buf(int thread_id) {
#ifdef USING_CMODEL
    int sz = BD_REG_COUNT;
    unsigned int *p_cmd = (unsigned int *)malloc(sz * sizeof(unsigned int));
    memset(p_cmd, 0, sz * sizeof(unsigned int));
    (void)(g_bd_buf_depth);
#else
    unsigned int *p_cmd;
    p_cmd = (unsigned int *)map_to_kaddr(BDC_CMD_BASE_ADDR);
#endif
    return (P_COMMAND)p_cmd;
}

static inline void bd_poll_fifo() {
#ifndef USING_CMODEL
  if (g_bd_buf_depth == 0) {
    reg_id_t reg_value = BD_ID_CFG_IST_FIFO_DEPTH;
    volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(
        BD_ENGINE_MAIN_CTRL + ((reg_value.where >> 5) << 2));
    reg_value.where &= 31;
    u32 mask_val = (u32)(1 << reg_value.len) - 1;

#ifdef RECORD_FIFO_DEPTH
    record_fifo_depth(1, g_bd_buf_depth);
#endif

    while (g_bd_buf_depth <= 2) {
      volatile u32 rd_data = ((*rd_addr) >> reg_value.where) & mask_val;
      if (rd_data >= 2) {
        g_bd_buf_depth = rd_data;
        break;
      }
    }
  }
  g_bd_buf_depth -= 1;

#ifdef RECORD_FIFO_DEPTH
  record_fifo_depth(1, g_bd_buf_depth);
#endif

#endif
}

static inline void bd_cmd_id_proc(unsigned int *id, CMD_ID_NODE *pid_node, int thread_id) {
    BD_CMD_ID_PROC(id[0], id[1], id[2], id[3], thread_id);
}
static inline void bd_common(cmd_t *cmd) {
    reg_pack_t pack_lane = BD_PACK_DES_TSK_LANE_NUM(g_bd_lane);
    write_cmd(BDC_CMD_BASE_ADDR, cmd, &pack_lane);
}

static inline void bd_start(cmd_t *cmd) {
}

static gen_cmd_t *bd_gen_cmd() {
    if (g_bd_gen_cmd_initialized == __FALSE__) {
        g_bd_gen_cmd.buf = bd_cmd_buf;
        g_bd_gen_cmd.id_proc = bd_cmd_id_proc;
        g_bd_gen_cmd.common = bd_common;
        g_bd_gen_cmd.start = bd_start;
        g_bd_gen_cmd.thread_id = 0;
        g_bd_gen_cmd_initialized = __TRUE__;
    }
    return &g_bd_gen_cmd;
}
static u64 get_bd_lane() {
    return g_bd_lane;
}
static void set_bd_lane(u64 lane) {
    g_bd_lane = lane;
}

static inline gen_cmd_t* BD_begin_fast_gen_cmd(u32 ids[4], CMD_ID_NODE *pid_node,int thread_id) {
    gen_cmd_t *fast_cmd = bd_gen_cmd();
    if (fast_cmd->cmd.buf == NULL) {
        fast_cmd->cmd.buf = bd_cmd_buf(thread_id);
    }
    bd_poll_fifo();
    fast_cmd->thread_id = thread_id;
    bd_cmd_id_proc(ids, pid_node, thread_id);
    return fast_cmd;
}

static inline void BD_end_fast_gen_cmd(gen_cmd_t *fast_cmd, CMD_ID_NODE *pid_node) {
    call_atomic(fast_cmd->cmd.buf, ENGINE_BD, pid_node, fast_cmd->thread_id);
}
