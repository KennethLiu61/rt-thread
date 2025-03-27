#ifndef FIRMWARE_COMMON_INLINE_H
#define FIRMWARE_COMMON_INLINE_H

#include "common.h"
#include "gdma_reg_def.h"
#include "bd_reg_def.h"
#include "sort_reg_def.h"
#include "cdma_reg_def.h"
#include "sdma_reg_def.h"
#include "cmd_id_proc.h"
#include "firmware_profile.h"
#include "firmware_common_macro.h"
#include "memmap.h"
#include "base_def.h"
#include "stas_gen_macro.h"

#ifdef USING_CMODEL
#include "cmodel_common.h"
#include "store_cmd.h"

#ifdef SG_STAS_GEN
#include "sg_stas_gen_util.h"
#include "get_atomic_profile.h"
#endif

#ifdef USING_MULTI_THREAD_ENGINE
#include "cmodel_multi_thread.h"
#endif

#ifdef SG_TV_GEN
#include "sg_tv_gen_util.h"
#endif

#else
#include "firmware_top.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

INLINE static void poll_bd_engine_fifo() {
  FW_READ32_WAIT_GE(BD_ENGINE_MAIN_CTRL + 4, 1, 12, 0);
}

INLINE static void poll_gdma_engine_fifo() {
  FW_READ32_WAIT_GE(GDMA_ENGINE_MAIN_CTRL, 1, 7, 8);
}

INLINE static void poll_hau_engine_fifo() {
  FW_REG_ID_WAIT_GE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_BUF_DEPTH, 1);
}
INLINE static void poll_sdma_engine_fifo() {
  FW_READ32_WAIT_GE(SDMA_ENGINE_MAIN_CTRL, 1, 7, 8);
}

INLINE static void poll_vsdma_engine_fifo(int port_id) {
  FW_READ32_WAIT_GE(VSDMA_ENGINE_MAIN_CTRL(port_id), 1, 7, 25);
}
INLINE static void poll_hau_engine_done(CMD_ID_NODE *id_node) {
//   profile_time_wait_sort(id_node->hau_cmd_id);
  reg_id_t id_pio_en = SORT_ID_PIO_ENABLE;
  u32 reg_data = READ_REG(HAU_ENGINE_MAIN_CTRL + id_pio_en.where / 32 * 4);
  int id_shift = id_pio_en.where % 32;
  int mask = id_pio_en.len < 32 ? ((1 << id_pio_en.len) - 1) : -1;
  if (((reg_data >> id_shift) & mask) == 1) {
    // pio mode
    FW_REG_ID_WAIT_EQ(HAU_ENGINE_MAIN_CTRL, SORT_ID_ENGINE_BUSY, 0);
  } else {
    // descriptor mode
    FW_REG_ID_WAIT_EQ(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 0);
  }
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
  int nodechip_idx = get_cur_nodechip_idx();
  wait_engine_done(nodechip_idx, ENGINE_HAU, id_node->hau_cmd_id);
#endif
//   profile_time_wait_sort(id_node->hau_cmd_id);
}
inline static u32 read_reg_u32(u64 base, reg_id_t reg)
{
    u32 mask = 0xffffffff >> (32 - (reg.len));
    u32 val = READ_REG(base + ((reg.where >> 5) << 2));
    return (val >> (reg.where & 31)) & mask;
}
inline static u32 read_bd_command_id()
{
    const reg_id_t reg = BD_ID_CFG_CURRENT_CMD_ID;
    return read_reg_u32(BD_ENGINE_MAIN_CTRL, reg);
}
inline static void poll_bd_engine_des_done()
{
    const reg_id_t reg = BD_ID_CFG_DES_MODE;
    FW_READ32_WAIT_EQ(
        BD_ENGINE_MAIN_CTRL + ((reg.where >> 5) << 2),
        0,
        reg.len,
        reg.where & 31);
    reg_id_t inst_fifo = BD_ID_CFG_IST_FIFO_DEPTH;
    UNUSED(inst_fifo);
    FW_WAIT_INST_INFO_EMPTY(inst_fifo, BD_ENGINE_MAIN_CTRL, 0x80);
}
inline static void poll_bd_engine_done(CMD_ID_NODE *id_node) {
  const reg_id_t reg_bd_id = BD_ID_CFG_CURRENT_CMD_ID;
  UNUSED(reg_bd_id);
  FW_READ32_WAIT_GE(BD_ENGINE_MAIN_CTRL + ((reg_bd_id.where >> 5) << 2),
                    id_node->bd_cmd_id, reg_bd_id.len, reg_bd_id.where & 31);
#ifndef USING_CMODEL
  id_node->bd_cmd_id = __rd_data;
#endif
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
  int core_idx = get_cur_nodechip_idx();
  wait_engine_done(core_idx, ENGINE_BD, id_node->bd_cmd_id);
#endif
}
inline static void poll_gdma_engine_des_done()
{
#if 0
    const reg_id_t int_reg = GDMA_ID_CFG_DES_MODE_END;
    FW_READ32_WAIT_EQ(
        GDMA_ENGINE_MAIN_CTRL + ((int_reg.where >> 5) << 2),
        1,
        int_reg.len,
        int_reg.where & 31);
    return;
#endif

    const reg_id_t reg = GDMA_ID_CFG_DES_ENABLE;
    FW_READ32_WAIT_EQ(
        GDMA_ENGINE_MAIN_CTRL + ((reg.where >> 5) << 2),
        0,
        reg.len,
        reg.where & 31);
    reg_id_t inst_fifo = GDMA_ID_CFG_IST_FIFO_DEPTH;
    UNUSED(inst_fifo);
    FW_WAIT_INST_INFO_EMPTY(inst_fifo, GDMA_ENGINE_MAIN_CTRL, 0x40);
}
inline static void poll_gdma_engine_done(CMD_ID_NODE *id_node) {
  const reg_id_t reg_gdma_id = GDMA_ID_CFG_CURRENT_CMD_ID;
  UNUSED(reg_gdma_id);
  FW_READ32_WAIT_GE(GDMA_ENGINE_MAIN_CTRL + ((reg_gdma_id.where >> 5) << 2),
                    id_node->gdma_cmd_id, reg_gdma_id.len,
                    reg_gdma_id.where & 31);
#ifndef USING_CMODEL
  id_node->gdma_cmd_id = __rd_data;
#endif
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
  int core_idx = get_cur_nodechip_idx();
  wait_engine_done(core_idx, ENGINE_GDMA, id_node->gdma_cmd_id);
#endif
}

INLINE static void poll_sdma_engine_done(CMD_ID_NODE *id_node) {
  const reg_id_t reg_sdma_id = SDMA_ID_CFG_CURRENT_CMD_ID;
  UNUSED(reg_sdma_id);
  FW_READ32_WAIT_GE(SDMA_ENGINE_MAIN_CTRL + ((reg_sdma_id.where >> 5) << 2),
                    id_node->sdma_cmd_id, reg_sdma_id.len,
                    reg_sdma_id.where & 31);
#ifndef USING_CMODEL
  id_node->sdma_cmd_id = __rd_data;
#endif
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
  int core_idx = get_cur_nodechip_idx();
  wait_engine_done(core_idx, ENGINE_SDMA, id_node->sdma_cmd_id);
#endif
}

inline static void poll_sdma_engine_des_done()
{
    const reg_id_t reg = SDMA_ID_CFG_DES_ENABLE;
    FW_READ32_WAIT_EQ(
        SDMA_ENGINE_MAIN_CTRL + ((reg.where >> 5) << 2),
        0,
        reg.len,
        reg.where & 31);
    reg_id_t inst_fifo = SDMA_ID_CFG_IST_FIFO_DEPTH;
    UNUSED(inst_fifo);
    FW_WAIT_INST_INFO_EMPTY(inst_fifo, SDMA_ENGINE_MAIN_CTRL, 0x40);
}

inline static void poll_vsdma_engine_des_done(int port)
{
    const reg_id_t reg = VSDMA_ID_CFG_DES_ENABLE;
    FW_READ32_WAIT_EQ(
        VSDMA_ENGINE_MAIN_CTRL(port) + ((reg.where >> 5) << 2),
        0,
        reg.len,
        reg.where & 31);
    const reg_id_t clr_reg = SDMA_ID_CFG_DES_CLR;
    FW_READ32_WAIT_EQ(
        VSDMA_ENGINE_MAIN_CTRL(port) + ((clr_reg.where >> 5) << 2),
        0,
        clr_reg.len,
        clr_reg.where & 31);
}

inline static void poll_cdma_engine_des_done(int port) {
    const reg_id_t reg = CDMA_ID_CFG_DES_ENABLE;
    FW_READ32_WAIT_EQ(
        CDMA_ENGINE_MAIN_CTRL(port) + ((reg.where >> 5) << 2),
        0,
        reg.len,
        reg.where & 31);
    const reg_id_t clr_reg = CDMA_ID_CFG_DES_CLR;
    FW_READ32_WAIT_EQ(
        CDMA_ENGINE_MAIN_CTRL(port) + ((clr_reg.where >> 5) << 2),
        0,
        clr_reg.len,
        clr_reg.where & 31);
}

INLINE static void poll_vsdma_engine_done(int port, CMD_ID_NODE *id_node) {
    const reg_id_t reg_vsdma_id = VSDMA_ID_CFG_CURRENT_CMD_ID;
    UNUSED(reg_vsdma_id);

    FW_READ32_WAIT_GE(
        VSDMA_ENGINE_MAIN_CTRL(port) + ((reg_vsdma_id.where >> 5) << 2),
        id_node->vsdma_cmd_id[port], reg_vsdma_id.len,
        reg_vsdma_id.where & 31);
    #if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
        wait_engine_done(port, ENGINE_VSDMA, id_node->vsdma_cmd_id[port]);
    #endif
}

INLINE static void poll_cdma_engine_fifo(int port) {
    // CDMA_CSR_8, reg_ins_buf_status[7:0]
    FW_READ32_WAIT_GE(CDMA_ENGINE_MAIN_CTRL(port)+0x28, 1, 7, 0);
}

INLINE static void poll_cdma_engine_done(int port, CMD_ID_NODE *id_node) {
    const reg_id_t reg_send_cmd_id = CDMA_ID_CFG_CURRENT_SEND_CMD_ID;
    const reg_id_t reg_rcv_cmd_id = CDMA_ID_CFG_CURRENT_RCV_CMD_ID;
    (void)reg_send_cmd_id;
    (void)reg_rcv_cmd_id;
    // we always have system command in the end
    // so the last send and rcv id are the same
    FW_CDMA_READ32_WAIT_GE(
        CDMA_ENGINE_MAIN_CTRL(port) + ((reg_send_cmd_id.where >> 5) << 2),
        CDMA_ENGINE_MAIN_CTRL(port) + ((reg_rcv_cmd_id.where >> 5) << 2),
        id_node->cdma_cmd_id[port], reg_send_cmd_id.len,
        reg_send_cmd_id.where & 31, port);

#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    wait_engine_done(port, ENGINE_CDMA, id_node->cdma_cmd_id[port]);
#endif
}

INLINE static void poll_cdma_perf_engine_done(int *chips, int *ports, int* actions, unsigned long long *info_addr, CMD_ID_NODE *id_node) {
    const reg_id_t reg_send_cmd_id = CDMA_ID_CFG_CURRENT_SEND_CMD_ID;
    const reg_id_t reg_rcv_cmd_id = CDMA_ID_CFG_CURRENT_RCV_CMD_ID;
    (void)reg_send_cmd_id;
    (void)reg_rcv_cmd_id;
    // we always have system command in the end
    // so the last send and rcv id are the same
    FW_CDMA_PERF_READ32_WAIT_GE(
        CDMA_ENGINE_MAIN_CTRL(ports[0]) + ((reg_send_cmd_id.where >> 5) << 2),
        CDMA_ENGINE_MAIN_CTRL(ports[0]) + ((reg_rcv_cmd_id.where >> 5) << 2),
        id_node->cdma_cmd_id[ports[0]],
        CDMA_ENGINE_MAIN_CTRL(ports[1]) + ((reg_send_cmd_id.where >> 5) << 2),
        CDMA_ENGINE_MAIN_CTRL(ports[1]) + ((reg_rcv_cmd_id.where >> 5) << 2),
        id_node->cdma_cmd_id[ports[1]],
        reg_send_cmd_id.len,
        reg_send_cmd_id.where & 31,
        info_addr,
        chips,
        ports,
        actions
    );
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    wait_engine_done(ports[0], ENGINE_CDMA, id_node->cdma_cmd_id[ports[0]]);
    wait_engine_done(ports[1], ENGINE_CDMA, id_node->cdma_cmd_id[ports[1]]);
#endif
}

static inline int fw_read32_less(unsigned long long addr, u32 cmp_data, u8 bits, u8 shift) {
#ifdef USING_CMODEL
  ASSERT(false);
  return 0;
#else
  volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(addr);
  u32 mask_val = (0xffffffff >> (32 - bits));
  volatile u32 rd_data = ((*rd_addr) >> shift) & mask_val;
  return (rd_data < cmp_data);
#endif
}

static inline int fw_reg_id_less(unsigned long long base, int* reg_id, u32 cmp_data) {
#ifdef USING_CMODEL
  ASSERT(false);
  return 0;
#else
  int id_offset = reg_id[0] / REG_WORD_WIDTH * REG_WORD_SIZE;
  int id_shift = reg_id[0] % REG_WORD_WIDTH;
  int id_len = reg_id[1];
  return fw_read32_less(base + id_offset, cmp_data, id_len, id_shift);
#endif
}

static __attribute__((optimize("O0"))) INLINE int check_engine_busy_internal(CMD_ID_NODE * id_node, ENGINE_TYPE engine_type) {
    int res = 0;
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    int node_idx = get_cur_nodechip_idx();
    u32 engine_id = get_engine_id(node_idx, engine_type);
    if (engine_type == ENGINE_GDMA) {
        res = (engine_id < id_node->gdma_cmd_id);
    } else if (engine_type == ENGINE_BD) {
        res = (engine_id < id_node->bd_cmd_id);
    } else if (engine_type == ENGINE_HAU) {
        res = (engine_id < id_node->hau_cmd_id);
    } else if (engine_type == ENGINE_SDMA) {
        res = (engine_id < id_node->sdma_cmd_id);
    } else {
        ASSERT(0);
    }
#elif defined(USING_CMODEL)
    ASSERT(0);
#else
    if (engine_type == ENGINE_GDMA) {
        const reg_id_t reg_gdma_id = GDMA_ID_CFG_CURRENT_CMD_ID;
        res = fw_read32_less(GDMA_ENGINE_MAIN_CTRL + ((reg_gdma_id.where >> 5) << 2),
                            id_node->gdma_cmd_id, reg_gdma_id.len, reg_gdma_id.where & 31);
    } else if (engine_type == ENGINE_BD) {
        const reg_id_t reg_bd_id = BD_ID_CFG_CURRENT_CMD_ID;
        res = fw_read32_less(BD_ENGINE_MAIN_CTRL + ((reg_bd_id.where >> 5) << 2),
                            id_node->bd_cmd_id, reg_bd_id.len, reg_bd_id.where & 31);
    } else if (engine_type == ENGINE_HAU) {
        reg_id_t id_pio_en = SORT_ID_PIO_ENABLE;
        u32 reg_data = READ_REG(HAU_ENGINE_MAIN_CTRL + id_pio_en.where / 32 * 4);
        int id_shift = id_pio_en.where % 32;
        int mask = id_pio_en.len < 32 ? ((1 << id_pio_en.len) - 1) : -1;
        if (((reg_data >> id_shift) & mask) == 1) {
            // pio mode
            int sort_busy_id[2] = SORT_ID_ENGINE_BUSY;
            res = fw_reg_id_less(HAU_ENGINE_MAIN_CTRL, sort_busy_id, 0);
        } else {
            // descriptor mode
            int sort_busy_id[2] = SORT_ID_DSCRP_START;
            res = fw_reg_id_less(HAU_ENGINE_MAIN_CTRL, sort_busy_id, 0);
        }
    } else if (engine_type == ENGINE_SDMA) {
        const reg_id_t reg_sdma_id = SDMA_ID_CFG_CURRENT_CMD_ID;
        res = fw_read32_less(SDMA_ENGINE_MAIN_CTRL + ((reg_sdma_id.where >> 5) << 2),
                            id_node->sdma_cmd_id, reg_sdma_id.len,
                            reg_sdma_id.where & 31);
        const reg_id_t reg_vsdma_id = VSDMA_ID_CFG_CURRENT_CMD_ID;
        res |= fw_read32_less(VSDMA_ENGINE_MAIN_CTRL(CORE_ID) + ((reg_vsdma_id.where >> 5) << 2),
                            id_node->vsdma_cmd_id[CORE_ID], reg_vsdma_id.len,
                            reg_vsdma_id.where & 31);
    } else {
        ASSERT(0);
    }
#endif
    return res;
}

INLINE static void assert_reset(int ip_id) {
#ifndef USING_CMODEL
    u32 reset = 0;
    reset = READ_REG(TPU_SYS_SOFT_RESET);
    WRITE_REG(TPU_SYS_SOFT_RESET, (reset | ip_id), NODECHIP_REG);
#else
    UNUSED(ip_id);
#endif
}

INLINE static void deassert_reset(int ip_id) {
#ifndef USING_CMODEL
    u32 reset = 0;
    reset = READ_REG(TPU_SYS_SOFT_RESET);
    WRITE_REG(TPU_SYS_SOFT_RESET, (reset & (~ip_id)), NODECHIP_REG);
#else
    UNUSED(ip_id);
#endif
}

INLINE static void enable_clk(int clk_id) {
#ifndef USING_CMODEL
    u32 clk = 0;
    clk = READ_REG(TPU_SYS_BASE_ADDR);
    WRITE_REG(TPU_SYS_BASE_ADDR, (clk | clk_id), NODECHIP_REG);
#else
    UNUSED(clk_id);
#endif
}

INLINE static void disable_clk(int clk_id) {
#ifndef USING_CMODEL
    u32 clk = 0;
    clk = READ_REG(TPU_SYS_BASE_ADDR);
    WRITE_REG(TPU_SYS_BASE_ADDR, (clk & (~clk_id)), NODECHIP_REG);
#else
    UNUSED(clk_id);
#endif
}

INLINE static void clk_hwlock_lock(void) {
    FW_READ32_WAIT_EQ(TOP_REG_DEVICE_LOCK_CLK, 0x0, 1, 0);
}

INLINE static void clk_hwlock_unlock(void) {
    WRITE_REG(TOP_REG_DEVICE_LOCK_CLK, 0x0, NODECHIP_REG);
}

#ifdef USING_CMODEL
#define call_atomic(cmd, eng_id, id_node, port_id)                                                \
    do {                                                                                    \
        store_cmd(cmd, eng_id, id_node,port_id, 0);                                                          \
        if (get_atomic_cmodel_enable()) {                                                         \
            cmodel_call_atomic(get_cur_nodechip_idx(), cmd, eng_id, port_id);                     \
        }                                                                                         \
        if (cmd != NULL) {                                                                        \
            free(cmd);                                                                            \
            cmd = NULL;                                                                           \
        }                                                                                         \
    } while (0)
#else
#define call_atomic(cmd, eng_id, id_node, port)       \
    (void)(cmd); (void)(eng_id);
#endif

INLINE static void resync_cmd_id(CMD_ID_NODE * p_cmd_id) {
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_CMD_ID_RESET, 1);
    FW_REG_ID_WRITE(GDMA_ENGINE_MAIN_CTRL, GDMA_ID_CFG_CMD_ID_RESET, 1);
    FW_REG_ID_WRITE(SDMA_ENGINE_MAIN_CTRL, SDMA_ID_CFG_CMD_ID_RESET, 1);
    p_cmd_id->bd_cmd_id = 0;
    p_cmd_id->gdma_cmd_id = 0;
    p_cmd_id->hau_cmd_id = 0;
    p_cmd_id->sdma_cmd_id = 0;
    p_cmd_id->in_parallel_state = false;
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    reset_all_engine_cmd_id(get_cur_nodechip_idx());
#endif
#ifdef SG_STAS_GEN
    p_cmd_id->cycle_count = 0;
    strcpy(p_cmd_id->name_prefix, "\0");
    sg_stas_reset();
#endif
}

INLINE static void resync_cdma_port_subsys_cmd_id(CMD_ID_NODE *p_cmd_id, int port) {
    FW_REG_ID_WRITE(CDMA_ENGINE_MAIN_CTRL(port), CDMA_ID_CFG_CMD_ID_RESET, 1);
    p_cmd_id->cdma_cmd_id[port] = 0;
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    reset_cdma_port_subsys_cmd_id(port);
#endif
}

INLINE static void resync_vsdma_port_subsys_cmd_id(CMD_ID_NODE *p_cmd_id, int port) {
    FW_REG_ID_WRITE(VSDMA_ENGINE_MAIN_CTRL(port), VSDMA_ID_CFG_CMD_ID_RESET, 1);
    p_cmd_id->vsdma_cmd_id[port] = 0;
#if defined(USING_MULTI_THREAD_ENGINE) && defined(USING_CMODEL)
    reset_vsdma_port_subsys_cmd_id(port);
#endif
}

static __attribute__((optimize("O0"))) INLINE void
poll_all_engine_done_internal(CMD_ID_NODE *id_node) {
  poll_bd_engine_done(id_node);
  poll_gdma_engine_done(id_node);
  poll_sdma_engine_done(id_node);
  poll_hau_engine_done(id_node);
}

INLINE static void poll_all_engine_done(CMD_ID_NODE * id_node) {
#ifndef SG_TV_GEN
    // profile_time_wait_node(0);
#endif
    poll_all_engine_done_internal(id_node);
#ifdef  SG_TV_GEN
    sg_wr_tv_dump_sync();
#endif
#ifdef SG_STAS_GEN
    sg_stas_show(id_node->bd_cmd_id, id_node->gdma_cmd_id, id_node->cycle_count);
#endif
#ifndef SG_TV_GEN
    // profile_time_wait_node(0);
#endif
}

#ifndef USING_CMODEL
    INLINE static void send_msg_done_interrupt() {
    }
#else
    __attribute__((weak)) void send_msg_done_interrupt() {
        FW_ERR(
            "Weak symbol send_msg_done_interrupt has been invoked!\n"
            "If you see this message, it means your calling executable does not properly provide send_msg_done_interrupt symbol.\n"
            "The calling executable MUST:\n"
            "1. have a strong symbol send_msg_done_interrupt;\n"
            "2. linked with option -Wl,--export-dynamic;\n"
            "3. not linked with -static option.\n");
        exit(-1);
    }
#endif

#define CACHE_LINE_SIZE 64
INLINE static void invalidate_cache(u64 address, u64 size) {
#if 0
    ASSERT(size % CACHE_LINE_SIZE == 0);
    ASSERT(address % CACHE_LINE_SIZE == 0);
#ifndef USING_CMODEL
    extern void invalidate_dcache_range(unsigned long start, unsigned long size);
    invalidate_dcache_range(address, size);
#endif
#endif
}
INLINE static void flush_cache(u64 address, u64 size) {
#if 0
    ASSERT(size % CACHE_LINE_SIZE == 0);
    ASSERT(address % CACHE_LINE_SIZE == 0);
#ifndef USING_CMODEL
    extern void flush_dcache_range(unsigned long start, unsigned long size);
    flush_dcache_range(address, size);
#endif
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* FIRMWARE_COMMON_H */
