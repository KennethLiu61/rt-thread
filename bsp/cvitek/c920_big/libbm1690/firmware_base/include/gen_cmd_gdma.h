#pragma once
#include "gen_cmd_utils.h"
#include "gdma_reg_def.h"

static THREAD gen_cmd_t g_gdma_gen_cmd;
static THREAD unsigned char g_gdma_gen_cmd_initialized = __FALSE__;
static THREAD unsigned int g_gdma_lane[2] = { 0xffffffff, 0xffffffff };
static THREAD unsigned char g_gdma_des_intr_flag = __FALSE__;
static THREAD int g_gdma_buf_depth = 0;

void record_fifo_depth(int index, uint32_t v);

static inline P_COMMAND gdma_cmd_buf(int thread_id) {
#ifdef USING_CMODEL
    int sz = GDMA_REG_COUNT;
    unsigned int *p_cmd = (unsigned int *)malloc(sz * sizeof(unsigned int));
    memset(p_cmd, 0, sz * sizeof(unsigned int));
    (void)(g_gdma_buf_depth);
#else
    unsigned int *p_cmd;
    if (thread_id == 0) {
        p_cmd = (unsigned int *)map_to_kaddr(GDMA_CMD_BASE_ADDR);
    } else {
        p_cmd = (unsigned int *)map_to_kaddr(GDMA_SLAVE_CMD_BASE_ADDR);
    }
#endif
    return (P_COMMAND)p_cmd;
}

static inline void gdma_poll_fifo() {
#ifndef USING_CMODEL
  if (g_gdma_buf_depth == 0) {
    reg_id_t reg_value = GDMA_ID_CFG_IST_FIFO_DEPTH;
    volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(
        GDMA_ENGINE_MAIN_CTRL + ((reg_value.where >> 5) << 2));
    reg_value.where &= 31;
    u32 mask_val = (u32)(1 << reg_value.len) - 1;
    while (g_gdma_buf_depth <= 2) {
      volatile u32 rd_data = ((*rd_addr) >> reg_value.where) & mask_val;
      if (rd_data >= 2) {
        g_gdma_buf_depth = rd_data;
        break;
      }
    }
  }

#ifdef RECORD_FIFO_DEPTH
  record_fifo_depth(0, g_gdma_buf_depth);
#endif

  g_gdma_buf_depth -= 1;
#endif
}
static inline void gdma_cmd_id_proc(unsigned int *id, CMD_ID_NODE *pid_node, int thread_id) {
    GDMA_CMD_ID_PROC(id[0], id[1], id[2], id[3], thread_id);
}

static inline void gdma_common(cmd_t *cmd) {
    reg_pack_t intr_en = GDMA_PACK_INTR_EN(g_gdma_des_intr_flag);
    write_cmd(GDMA_CMD_BASE_ADDR, cmd, &intr_en);

    // 0x0 indicate only sync with TPU
    reg_pack_t lane_mask_l32 = GDMA_PACK_LOCALMEM_MASK_L32(g_gdma_lane[0]);
    write_cmd(GDMA_CMD_BASE_ADDR, cmd, &lane_mask_l32);
    reg_pack_t lane_mask_h32 = GDMA_PACK_LOCALMEM_MASK_H32(g_gdma_lane[1]);
    write_cmd(GDMA_CMD_BASE_ADDR, cmd, &lane_mask_h32);
}

static inline void gdma_start(cmd_t *cmd) {
}

static gen_cmd_t *gdma_gen_cmd() {
    if (g_gdma_gen_cmd_initialized == __FALSE__) {
        g_gdma_gen_cmd.buf = gdma_cmd_buf;
        g_gdma_gen_cmd.id_proc = gdma_cmd_id_proc;
        g_gdma_gen_cmd.common = gdma_common;
        g_gdma_gen_cmd.start = gdma_start;
        g_gdma_gen_cmd.thread_id =0;
        g_gdma_gen_cmd_initialized = __TRUE__;
    }
    return &g_gdma_gen_cmd;
}
static unsigned int get_gdma_lane(int sec_idx) {
    return g_gdma_lane[sec_idx];
}
static void set_gdma_lane(unsigned int lane, int sec_idx) {
    g_gdma_lane[sec_idx] = lane;
}
static unsigned char get_gdma_des_intr_enable() {
    return g_gdma_des_intr_flag;
}
static void set_gdma_des_intr_enable(unsigned char flag) {
    g_gdma_des_intr_flag = flag;
}

static inline gen_cmd_t* GDMA_begin_fast_gen_cmd(u32 ids[4], CMD_ID_NODE *pid_node, int thread_id) {
    gen_cmd_t *fast_cmd = gdma_gen_cmd();
    if (fast_cmd->cmd.buf == NULL || fast_cmd->thread_id != thread_id) {
        g_gdma_gen_cmd.cmd.buf = gdma_cmd_buf(thread_id);
    }
    fast_cmd->thread_id = thread_id;
    gdma_cmd_id_proc(ids, pid_node, thread_id);
    gdma_poll_fifo();
    return fast_cmd;
}

static inline void GDMA_end_fast_gen_cmd(gen_cmd_t *fast_cmd, CMD_ID_NODE *pid_node) {
    call_atomic(fast_cmd->cmd.buf, ENGINE_GDMA, pid_node, fast_cmd->thread_id);
}
