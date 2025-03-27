#pragma once
#include "cdma_reg_def.h"
#include "gen_cmd_utils.h"

static THREAD cdma_gen_cmd_t g_cdma_gen_cmd;
static THREAD unsigned char g_cdma_gen_cmd_initialzed = __FALSE__;
static THREAD int g_cdma_buf_depth[MAX_CDMA_NUM] = {0};

static inline P_COMMAND cdma_cmd_buf(int port) {
#ifdef USING_CMODEL
    int sz = CDMA_REG_COUNT;
    poll_cdma_engine_fifo(port);
    unsigned int *p_cmd = (unsigned int *) malloc(sz * sizeof(unsigned int));
    memset(p_cmd, 0, sz * sizeof(unsigned int));
    (void) (g_cdma_buf_depth[port]);
#else
    unsigned int *p_cmd;
    p_cmd = (unsigned int *) map_to_kaddr(CDMA_CMD_BASE_ADDR(port));
    reg_id_t common_reg_value = CDMA_ID_CFG_IST_FIFO_DEPTH;
    volatile u32 *rd_addr = (volatile u32 *) map_to_kaddr(CDMA_ENGINE_MAIN_CTRL(port) + ((common_reg_value.where >> 5) << 2));
    common_reg_value.where &= 31;
    u32 mask_val = (u32) (1 << common_reg_value.len) - 1;
    while (g_cdma_buf_depth[port] <= 2) {
        volatile u32 rd_data = ((*rd_addr) >> common_reg_value.where) & mask_val;
        if (rd_data >= 2) {
            g_cdma_buf_depth[port] = rd_data;
            break;
        }
    }
    g_cdma_buf_depth[port] -= 1;
#endif
    return (P_COMMAND) p_cmd;
}

static inline void cdma_cmd_id_proc(int port, unsigned int *id, CMD_ID_NODE *pid_node) {
    CDMA_CMD_ID_PROC(port);
}


static inline void cdma_common(int port, cmd_t *cmd) {}

static inline void cdma_start(int port, cmd_t *cmd) {}

static cdma_gen_cmd_t *cdma_gen_cmd() {
    if (g_cdma_gen_cmd_initialzed == __FALSE__) {
        g_cdma_gen_cmd.buf = cdma_cmd_buf;
        g_cdma_gen_cmd.id_proc = cdma_cmd_id_proc;
        g_cdma_gen_cmd.common = cdma_common;
        g_cdma_gen_cmd.start = cdma_start;
        g_cdma_gen_cmd_initialzed = __TRUE__;
    }
    return &g_cdma_gen_cmd;
}

static inline cdma_gen_cmd_t* CDMA_begin_fast_gen_cmd(int port, u32 ids[3], CMD_ID_NODE *pid_node) {
    cdma_gen_cmd_t *fast_cmd = cdma_gen_cmd();
    fast_cmd->cmd.buf = cdma_cmd_buf(port);
    cdma_cmd_id_proc(port, ids, pid_node);
    return fast_cmd;
}

static inline void CDMA_end_fast_gen_cmd(int port, cdma_gen_cmd_t *fast_cmd, CMD_ID_NODE *pid_node) {
    call_atomic(fast_cmd->cmd.buf, ENGINE_CDMA, pid_node, port);
    fast_cmd->cmd.buf = NULL;
}