#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"
#include "stas_gen_macro.h"

// src_shape=[N,1,H,W], dst_shape=[N,C,H,W]
// storage aligned in local memory
void atomic_lane_broad_gen_cmd(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node)
{
  FW_DBG("%s,  src_addr:%u,  dst_addr:%u,  N:%d,  "
         "H:%d,  W:%d,  dst_C:%d,  lane_mask:%llu\n",
         __func__, src_addr, dst_addr, N,
         H, W, dst_C, lane_mask);

#ifdef USING_CMODEL
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(dst_C < (1 << 16) && dst_C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + dst_C <= NPU_NUM);
#endif

  LANE_BC_GET_PROFILE(N, dst_C, H, W, prec, src_addr, dst_addr, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)LANE_BROAD << 45) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec << 8);
    high[1] = lane_mask;
    low[2] = ((u64)N) |
          ((u64)dst_C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[2] = (1ull << 16) |
           ((u64)W << 48);
    high[4] = ((u64)dst_addr) |
           ((u64)src_addr << 32);

    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    low[0] = (1ull) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)LANE_BROAD << 45) |
          ((u64)prec << 50) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)N) |
           ((u64)dst_C << 16) |
           ((u64)H << 32) |
           ((u64)W << 48);
    low[1] = (1ull) |
          ((u64)W << 16);
    high[1] = ((u64)dst_addr) | ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, TRANS_BC,
      LANE_BROAD, prec, pid_node, high, low, elt);
}

// src_shape=dst_shape=[N,C,H,W]
// storage aligned in local memory
void atomic_lane_copy_gen_cmd(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int C,
    int H,
    int W,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node)
{
  FW_DBG("%s,  src_addr:%u,  dst_addr:%u,  N:%d,  "
         "C:%d,  H:%d,  W:%d\n",
         __func__, src_addr, dst_addr, N,
         C, H, W);

#ifdef USING_CMODEL
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(dst_addr != src_addr);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
#endif

  LANE_CPY_GET_PROFILE(N, C, H, W, prec, src_addr, dst_addr, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 8;
    u64 high[8] = {0};
    u64 low[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)LANE_COPY << 45) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec << 8);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[2] = ((u64)C << 16) |
           ((u64)W << 48);
    high[4] = ((u64)dst_addr) |
           ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    low[0] = (1ull) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)LANE_COPY << 45) |
          ((u64)prec << 50) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)N) |
           ((u64)C << 16) |
           ((u64)H << 32) |
           ((u64)W << 48);
    low[1] = ((u64)C) |
          ((u64)W << 16);
    high[1] = ((u64)dst_addr) | ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, TRANS_BC,
      LANE_COPY, prec, pid_node, high, low, elt);
}

// src_shape=[1,1,1,W], dst_shape=[1,C,1,W]
// storage aligned in local memory
void atomic_static_broad_gen_cmd(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node)
{
  FW_DBG("%s,  src_addr:%u,  dst_addr:%u,  c:%d,  W:%d,  "
         "lane_mask:%llu,  prec:%d\n",
         __func__, src_addr, dst_addr, C, W,
         lane_mask, prec);

#ifdef USING_CMODEL
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + C <= NPU_NUM);
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
#endif

  STATIC_BC_GET_PROFILE(C, W, prec, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 8;
    u64 high[8] = {0};
    u64 low[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)STATIC_BROAD << 45) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec << 8);
    high[1] = lane_mask;
    low[2] = ((u64)1) |
          ((u64)C << 16) |
          ((u64)1 << 32) |
          ((u64)W << 48);
    high[2] = ((u64)C << 16) |
           ((u64)W << 48);
    high[4] = ((u64)dst_addr) |
           ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    low[0] = (1ull) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)STATIC_BROAD << 45) |
          ((u64)prec << 50) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)1) |
           ((u64)C << 16) |
           ((u64)1 << 32) |
           ((u64)W << 48);
    low[1] = ((u64)C) |
          ((u64)W << 16);
    high[1] = ((u64)dst_addr) | ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, TRANS_BC,
      STATIC_BROAD, prec, pid_node, high, low, elt);
}

// src_shape=dst_shape=[1,C,1,1]
// storage compacted in local memory
void atomic_static_distribute_gen_cmd(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node)
{
  FW_DBG("%s,  src_addr:%u,  dst_addr:%u,  c:%d,  lane_mask:%llu,  prec:%d\n",
         __func__, src_addr, dst_addr, C, lane_mask, prec);

#ifdef USING_CMODEL
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu == 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % get_bytesize(prec) == 0);
#endif

  DIS_BC_GET_PROFILE(C, prec, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 8;
    u64 high[8] = {0};
    u64 low[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)STATIC_DISTRIBUTE << 45) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec << 8);
    high[1] = lane_mask;
    low[2] = ((u64)1) |
          ((u64)C << 16) |
          ((u64)1 << 32) |
          ((u64)1 << 48);
    high[2] = ((u64)C << 16) |
           ((u64)1 << 48);
    high[4] = ((u64)dst_addr) |
           ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    low[0] = (1ull) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)TRANS_BC << 41) |
          ((u64)STATIC_DISTRIBUTE << 45) |
          ((u64)prec << 50) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)1) |
           ((u64)C << 16) |
           ((u64)1 << 32) |
           ((u64)1 << 48);
    low[1] = ((u64)C) |
          ((u64)1 << 16);
    high[1] = ((u64)dst_addr) | ((u64)src_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, TRANS_BC,
      STATIC_DISTRIBUTE, prec, pid_node, high, low, elt);
}
