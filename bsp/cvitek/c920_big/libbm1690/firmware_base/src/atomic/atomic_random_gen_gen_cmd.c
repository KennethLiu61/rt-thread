#include "firmware_common.h"
#include "atomic_random_gen_gen_cmd.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

#define CHECK_RG_STRIDE(p_stride) \
      ASSERT((p_stride[0] < (((int)1) << 18)) && (p_stride[0] >= 0)); \
      ASSERT((p_stride[1] < (((int)1) << 18)) && (p_stride[1] >= 0)); \
      ASSERT((p_stride[2] < (((int)1) << 18)) && (p_stride[2] >= 0)); \
      ASSERT((p_stride[3] < (((int)1) << 18)) && (p_stride[3] >= 0)); \

#define UNUSED(x) (void)(x)

void atomic_random_gen_gen_cmd(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    unsigned int load_state_addr,
    unsigned int store_state_addr,
    int need_store,
    RAND_OP op_type,
    int thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG(
        "%s: "
        "addr = 0x%08x, "
        "n = %d, "
        "c = %d, "
        "h = %d, "
        "w = %d, "
        "stride = [%d, %d, %d, %d], "
        "short_str = %d, "
        "prec = %d, "
        "load_state_addr = 0x%08x, "
        "store_state_addr = 0x%08x, "
        "op_type = %d, "
        "bd_cmd_id = %u, gdma_cmd_id = %u\n",
        __func__,
        addr,
        n, c, h, w,
        stride == NULL ? 0 : stride[0],
        stride == NULL ? 0 : stride[1],
        stride == NULL ? 0 : stride[2],
        stride == NULL ? 0 : stride[3],
        short_str,
        prec,
        load_state_addr,
        store_state_addr,
        op_type,
        pid_node->bd_cmd_id,
        pid_node->gdma_cmd_id
    );
#ifdef USING_CMODEL
    ASSERT(short_str == 0);
    ASSERT(addr % ALIGN_BYTES == 0);
    ASSERT(prec != INT4 && prec != FP32 && prec != FP16 && prec != BFP16 && prec != FP8);
    ASSERT(n == 1);
    ASSERT(c < (((int)1) << 16) && (c > 0));
    ASSERT(h == 1);
    ASSERT(w < (((int)1) << 16) && (w > 0));
    ASSERT(w % get_eu_num(prec) == 0);
    if(need_store) {
        ASSERT(addr / LOCAL_MEM_SIZE == store_state_addr / LOCAL_MEM_SIZE);
    }
    if (op_type == PRNG_WITH_LOADED_STATES) {
        ASSERT(addr / LOCAL_MEM_SIZE == load_state_addr / LOCAL_MEM_SIZE);
    }
    ASSERT(op_type == PRNG || op_type == PRNG_WITH_LOADED_STATES);
    if(op_type == PRNG) {
        ASSERT(c <= NPU_NUM && c > 0);
    }
#endif
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    bool need_stride = ((short_str != 0)&&(short_str != 1));
    u32 N_STR = need_stride ? stride[0]&0x0000FFFF : 0;
    u32 C_STR = need_stride ? stride[1]&0x0000FFFF : 0;
    u32 H_STR = need_stride ? (((stride[0]&0x30000) << 2) | (stride[2]&0x0003FFFF)) : 0;
    u32 W_STR = need_stride ? (((stride[1]&0x30000) << 2) | (stride[3]&0x0003FFFF)) : 0;
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)RANDOM_GEN << 41) |
              ((u64)op_type << 45) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)need_store) |
               ((u64)prec << 8) |
               ((u64)short_str << 23);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)n) |
              ((u64)c << 16) |
              ((u64)h << 32) |
              ((u64)w << 48);
        high[3] = ((u64)N_STR) |
               ((u64)C_STR << 16);
        high[4] = ((u64)addr) |
               ((u64)load_state_addr << 32);
        high[5] = ((u64)H_STR) |
               ((u64)W_STR << 32);
        high[7] = store_state_addr;
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 3;
    u64 low[3] = {0}, high[3] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = ((u64)1) |
                (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
                ((u64)1ull << 37) |
                ((u64)RANDOM_GEN << 41) |
                ((u64)op_type << 45) |
                ((u64) need_store << 55) |
                ((u64)prec << 56) |
                ((u64)bd_power_step() << 59);
        high[0] = bd_get_lane_mask();
        low[1] = ((u64)c) |
                ((u64)w << 16);
        high[1] = ((u64)addr) |
                ((u64)load_state_addr << 32);
        low[2] = ((u64)store_state_addr);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, RANDOM_GEN,
      op_type, prec, pid_node, high, low, elt);
}

void atomic_random_gen_init_seed_gen_cmd(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    int jump_cnt,
    int c_offset,
    unsigned int store_state_addr,
    int need_store,
    int thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG(
        "%s: "
        "addr = 0x%08x, "
        "n = %d, "
        "c = %d, "
        "h = %d, "
        "w = %d, "
        "stride = [%d, %d, %d, %d], "
        "short_str = %d, "
        "prec = %d, "
        "jump_cnt = %d,"
        "c_offset = %d,"
        "store_state_addr = 0x%08x, "
        "bd_cmd_id = %u, gdma_cmd_id = %u\n",
        __func__,
        addr,
        n, c, h, w,
        stride == NULL ? 0 : stride[0],
        stride == NULL ? 0 : stride[1],
        stride == NULL ? 0 : stride[2],
        stride == NULL ? 0 : stride[3],
        short_str,
        prec,
        jump_cnt,
        c_offset,
        store_state_addr,
        pid_node->bd_cmd_id,
        pid_node->gdma_cmd_id
    );
#ifdef USING_CMODEL
    ASSERT((short_str == 0));
    ASSERT(addr % ALIGN_BYTES == 0);
    ASSERT(prec != INT4 && prec != FP32 && prec != FP16 && prec != BFP16 && prec != FP8);
    ASSERT(n == 1);
    ASSERT(c < (((int)1) << 16) && (c > 0));
    ASSERT(h == 1);
    ASSERT(w < (((int)1) << 16) && (w > 0));
    ASSERT(w % get_eu_num(prec) == 0);
    ASSERT(c_offset < (((int)1) << 16) && (c_offset >= 0));
    ASSERT((short_str == 0));
    if(need_store) {
        ASSERT(addr / LOCAL_MEM_SIZE == store_state_addr / LOCAL_MEM_SIZE);
    }
#endif
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    bool need_stride = ((short_str != 0)&&(short_str != 1));
    u32 N_STR = need_stride ? stride[0]&0x0000FFFF : 0;
    u32 C_STR = need_stride ? stride[1]&0x0000FFFF : 0;
    u32 H_STR = need_stride ? (((stride[0]&0x30000) << 2) | (stride[2]&0x0003FFFF)) : 0;
    u32 W_STR = need_stride ? (((stride[1]&0x30000) << 2) | (stride[3]&0x0003FFFF)) : 0;
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)RANDOM_GEN << 41) |
              ((u64)PRNG_WITH_INTIAL_SEED << 45) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)need_store) |
               ((u64)prec << 8) |
               ((u64)short_str << 23);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)n) |
              ((u64)c << 16) |
              ((u64)h << 32) |
              ((u64)w << 48);
        high[3] = ((u64)N_STR) |
               ((u64)C_STR << 16);
        low[4] =  ((u64)c_offset << 32) |
               ((u64)jump_cnt << 48);
        high[4] = ((u64)addr);
        high[5] = ((u64)H_STR) |
               ((u64)W_STR << 32);
        high[7] = store_state_addr;
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 3;
    u64 low[3] = {0}, high[3] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = ((u64)1) |
                (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
                ((u64)1ull << 37) |
                ((u64)RANDOM_GEN << 41) |
                ((u64)PRNG_WITH_INTIAL_SEED << 45) |
                ((u64) need_store << 55) |
                ((u64)prec << 56) |
                ((u64)bd_power_step() << 59);
        high[0] = bd_get_lane_mask();
        low[1] = ((u64)c) |
                ((u64)w << 16) |
                ((u64)c_offset << 32) |
                ((u64)jump_cnt << 48);
        high[1] = ((u64)addr);
        low[2] = ((u64)store_state_addr);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, RANDOM_GEN,
      PRNG_WITH_INTIAL_SEED, prec, pid_node, high, low, elt);
}
