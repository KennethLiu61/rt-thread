#pragma once
#include "gen_cmd_utils.h"
#include "sdma_reg_def.h"

static THREAD sdma_gen_cmd_t g_sdma_gen_cmd;
static THREAD unsigned char g_sdma_gen_cmd_initialized = __FALSE__;
static THREAD unsigned int g_sdma_lane[2] = { 0xffffffff, 0xffffffff };
static THREAD unsigned char g_sdma_des_intr_flag = __FALSE__;
static THREAD int g_sdma_master_buf_depth = 0;
static THREAD int g_sdma_slave_buf_depth = 0;

static inline P_COMMAND sdma_cmd_buf(int port_id) {
#ifdef USING_CMODEL
    int sz = SDMA_REG_COUNT;
    unsigned int *p_cmd = (unsigned int *)malloc(sz * sizeof(unsigned int));
    memset(p_cmd, 0, sz * sizeof(unsigned int));
    (void)(g_sdma_master_buf_depth);
    (void)(g_sdma_slave_buf_depth);
#else
    ENGINE_TYPE sdma_type = port_id == -1 ? ENGINE_SDMA : ENGINE_VSDMA;
    unsigned int *p_cmd;
    if (sdma_type == ENGINE_SDMA) {
        p_cmd = (unsigned int*)map_to_kaddr(SDMA_CMD_BASE_ADDR);
    } else {
        p_cmd = (unsigned int*)map_to_kaddr(VSDMA_CMD_BASE_ADDR(port_id));
    }

#endif
    return (P_COMMAND)p_cmd;
}

static inline void sdma_poll_fifo(int port_id) {
#ifndef USING_CMODEL
  ENGINE_TYPE sdma_type = port_id == -1 ? ENGINE_SDMA : ENGINE_VSDMA;
  if (sdma_type == ENGINE_SDMA) {
    if (g_sdma_master_buf_depth == 0) {
      reg_id_t reg_value = SDMA_ID_CFG_IST_FIFO_DEPTH;
      volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(
          SDMA_ENGINE_MAIN_CTRL + ((reg_value.where >> 5) << 2));
      reg_value.where &= 31;
      u32 mask_val = (u32)(1 << reg_value.len) - 1;
      while (g_sdma_master_buf_depth <= 2) {
        volatile u32 rd_data = ((*rd_addr) >> reg_value.where) & mask_val;
        if (rd_data >= 2) {
          g_sdma_master_buf_depth = rd_data;
          break;
        }
      }
    }
    g_sdma_master_buf_depth -= 1;
  } else {
    if (g_sdma_slave_buf_depth == 0) {
      reg_id_t reg_value = VSDMA_ID_CFG_IST_FIFO_DEPTH;
      volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(
          VSDMA_ENGINE_MAIN_CTRL(port_id) + ((reg_value.where >> 5) << 2));
      reg_value.where %= 32;
      u32 mask_val = (u32)(1 << reg_value.len) - 1;
      while (g_sdma_slave_buf_depth <= 2) {
        volatile u32 rd_data = ((*rd_addr) >> reg_value.where) & mask_val;
        if (rd_data >= 2) {
          g_sdma_slave_buf_depth = rd_data;
          break;
        }
      }
    }
    g_sdma_slave_buf_depth -= 1;
  }
#endif
}

static inline void sdma_cmd_id_proc(unsigned int *id, CMD_ID_NODE *pid_node, ENGINE_TYPE sdma_type, int port_id) {
    if (sdma_type == ENGINE_SDMA) {
        SDMA_CMD_ID_PROC_CORE();
    } else {
        VSDMA_CMD_ID_PROC_CORE(port_id);
    }
}

static inline void sdma_common(cmd_t *cmd) {
}

static inline void sdma_start(cmd_t *cmd) {
}

static sdma_gen_cmd_t *sdma_gen_cmd() {
    if (g_sdma_gen_cmd_initialized == __FALSE__) {
        g_sdma_gen_cmd.buf = sdma_cmd_buf;
        g_sdma_gen_cmd.id_proc = sdma_cmd_id_proc;
        g_sdma_gen_cmd.common = sdma_common;
        g_sdma_gen_cmd.start = sdma_start;
        g_sdma_gen_cmd.sdma_type = ENGINE_SDMA;
        g_sdma_gen_cmd.port_id = -1;
        g_sdma_gen_cmd_initialized = __TRUE__;
    }
    return &g_sdma_gen_cmd;
}

static unsigned int get_sdma_lane(int sec_idx) {
    return g_sdma_lane[sec_idx];
}

static void set_sdma_lane(unsigned int lane, int sec_idx) {
    g_sdma_lane[sec_idx] = lane;
}

static unsigned char get_sdma_des_intr_enable() {
    return g_sdma_des_intr_flag;
}

static void set_sdma_des_intr_enable(unsigned char flag) {
    g_sdma_des_intr_flag = flag;
}

static inline sdma_gen_cmd_t* SDMA_begin_fast_gen_cmd(u32 ids[3], CMD_ID_NODE *pid_node, int port_id) {
    sdma_gen_cmd_t *fast_cmd = sdma_gen_cmd();
    if (fast_cmd->cmd.buf == NULL || fast_cmd->port_id != port_id) {
        g_sdma_gen_cmd.cmd.buf = sdma_cmd_buf(port_id);
    }
    fast_cmd->sdma_type = port_id == -1 ? ENGINE_SDMA : ENGINE_VSDMA;
    fast_cmd->port_id = port_id;
    sdma_cmd_id_proc(ids, pid_node, fast_cmd->sdma_type, port_id);
    sdma_poll_fifo(port_id);
    return fast_cmd;
}

static inline void SDMA_end_fast_gen_cmd(sdma_gen_cmd_t *fast_cmd, CMD_ID_NODE *pid_node) {
    call_atomic(fast_cmd->cmd.buf, fast_cmd->sdma_type, pid_node, fast_cmd->port_id);
}
