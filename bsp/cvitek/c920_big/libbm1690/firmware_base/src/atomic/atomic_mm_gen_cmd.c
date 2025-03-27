#include "atomic_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

void atomic_mm_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int is_bias_const,
  int add_result,
  int do_relu,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s:  L_addr:0x%08x  R_addr:0x%08x  Y_addr:0x%08x  bias_addr:0x%08x  "
          "L_tensor_W:%d  L_tensor_C:%d  R_tensor_W:%d  R_tensor_C:%d  "
          "L_row_num:%d  L_col_num:%d  is_L_trans:%d  is_L_const:%d  "
          "is_bias_const:%d  add_result:%d  do_relu:%d\n",
          __func__, L_addr, R_addr, Y_addr, bias_addr, L_tensor_W, L_tensor_C,
          R_tensor_W, R_tensor_C, L_row_num, L_col_num, is_L_trans, is_L_const,
          is_bias_const, add_result, do_relu);

  int L_last_W = (is_L_trans ? L_row_num : L_col_num) % L_tensor_W;
  if (L_last_W == 0) L_last_W = L_tensor_W;

#ifdef USING_CMODEL
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
  }
  ASSERT(R_addr % ALIGN_BYTES == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  if (!is_bias_const) {
    ASSERT(bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Y_addr) == get_npu_index(bias_addr));
  }
  ASSERT(get_npu_index(Y_addr) == get_npu_index(R_addr));
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(is_L_trans < (1 << 1) && is_L_trans >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(R_tensor_C < (1 << 16) && R_tensor_C > 0);
  ASSERT(R_tensor_W < (1 << 16) && R_tensor_W > 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_tensor_C < (1 << 16) && L_tensor_C > 0);
  ASSERT(L_tensor_W < (1 << 16) && L_tensor_W > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(L_last_W < (1 << 16) && L_last_W > 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
#endif
  MM_GET_PROFILE(L_row_num, L_col_num, R_tensor_C, R_tensor_W, L_last_W, L_tensor_C, L_tensor_W, FP32, Y_addr, !is_bias_const, is_L_trans, add_result, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)MM << 41) |
            ((u64)MM_NORMAL << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)add_result) |
             ((u64)do_relu << 1) |
             ((u64)is_L_trans << 2) |
             ((u64)FP32 << 8) |
             ((u64)FP32 << 11) |
             ((u64)is_L_const << 20) |
             ((u64)is_bias_const << 22);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)R_tensor_C << 16) |
            ((u64)R_tensor_W << 48);
      high[2] = ((is_L_trans ? (u64)L_col_num : (u64)L_row_num)) |
             ((u64)L_tensor_C << 16) |
             ((u64)L_tensor_W << 48);
      low[3] = (u64)L_last_W << 48;
      high[4] = ((u64)Y_addr) | ((u64)L_addr << 32);
      low[5] = ((u64)R_addr) | ((u64)bias_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            ((u64)do_relu << 4) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)MM << 41) |
            ((u64)MM_NORMAL << 45) |
            ((u64)add_result << 50) |
            ((u64)is_L_trans << 51) |
            ((u64)is_L_const << 52) |
            ((u64)is_bias_const << 53) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)FP32 << 3) |
             ((u64)FP32 << 6) |
             ((u64)R_tensor_C << 48);
      low[1] = ((u64)R_tensor_W) |
            ((is_L_trans ? (u64)L_col_num : (u64)L_row_num) << 16) |
            ((u64)L_tensor_C << 32) |
            ((u64)L_tensor_W << 48);
      high[1] = ((u64)L_last_W << 16) |
             ((u64)Y_addr << 32);
      low[2] = ((u64)L_addr) | ((u64)R_addr << 32);
      high[2] = ((u64)bias_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, MM,
      MM_NORMAL, FP32, pid_node, high, low, elt);
}

void atomic_mm_fixed_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int L_sign,
  int R_sign,
  int bias_sign,
  int Res_sign,
  int is_bias_const,
  int add_result,
  int if_relu,
  int sym_range,
  int do_rq,
  s32 multiplier,
  s8 shift,
  s16 yzp,
  PREC Y_prec,
  PREC LR_prec,
  ROUND_MODE round_mode,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s:  L_addr:0x%08x  R_addr:0x%08x  Y_addr:0x%08x  bias_addr:0x%08x  "
          "L_tensor_W:%d  L_tensor_C:%d  R_tensor_W:%d  R_tensor_C:%d  "
          "L_row_num:%d  L_col_num:%d  is_L_trans:%d  is_L_const:%d  "
          "L_sign:%d  R_sign:%d  bias_sign:%d  Res_sign:%d  is_bias_const:%d  "
          "add_result:%d  if_relu:%d  sym_range:%d  do_rq:%d  "
          "multiplier:%d  shift:%d  yzp:%d  LR_prec:%d  Y_prec:%d  round_mode:%d\n",
          __func__, L_addr, R_addr, Y_addr, bias_addr,
          L_tensor_W, L_tensor_C, R_tensor_W, R_tensor_C,
          L_row_num, L_col_num, is_L_trans, is_L_const,
          L_sign, R_sign, bias_sign, Res_sign, is_bias_const,
          add_result, if_relu, sym_range, do_rq,
          multiplier, shift, yzp, LR_prec, Y_prec, round_mode);

  int L_last_W = (is_L_trans ? L_row_num : L_col_num) % L_tensor_W;
  if (L_last_W == 0) L_last_W = L_tensor_W;
  if (is_bias_const && bias_addr == 0) bias_sign = 0;

#ifdef USING_CMODEL
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
  }
  ASSERT(R_addr % ALIGN_BYTES == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  if (!is_bias_const) {
    ASSERT(bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(bias_addr) == get_npu_index(Y_addr));
  }
    ASSERT(get_npu_index(R_addr) == get_npu_index(Y_addr));
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(is_L_trans < (1 << 1) && is_L_trans >= 0);
  ASSERT(L_sign < (1 << 1) && L_sign >= 0);
  ASSERT(R_sign < (1 << 1) && R_sign >= 0);
  ASSERT(bias_sign < (1 << 1) && bias_sign >= 0);
  ASSERT(Res_sign < (1 << 1) && Res_sign >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(R_tensor_C < (1 << 16) && R_tensor_C > 0);
  ASSERT(R_tensor_W < (1 << 16) && R_tensor_W > 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_tensor_C < (1 << 16) && L_tensor_C > 0);
  ASSERT(L_tensor_W < (1 << 16) && L_tensor_W > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(L_last_W < (1 << 16) && L_last_W > 0);
  ASSERT(if_relu == 0 || if_relu == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(round_mode < 7 && round_mode >= 0);
  ASSERT(LR_prec == INT8 || LR_prec == INT16 || LR_prec == INT32);
  ASSERT(Y_prec == INT8 || Y_prec == INT16 || Y_prec == INT32);
#endif

  //write tgcr
  if (do_rq) {
    u32 value[3] = {(u32)multiplier, (u32)shift, (u32)yzp};
    int indice[3] = {6, 32, 33};
    atomic_bd_trwr_gen_cmd(value, indice, 3, MASTER_THREAD, pid_node);
  }

  MM_GET_PROFILE(L_row_num,L_col_num, R_tensor_C, R_tensor_W, L_last_W, L_tensor_C, L_tensor_W, INT8, Y_addr, !is_bias_const, is_L_trans, add_result, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)MM << 41) |
            ((u64)MM_NORMAL << 45) |
            ((u64)(do_rq) << 50) |
            ((u64)(Res_sign) << 55) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)add_result) |
             ((u64)if_relu << 1) |
             ((u64)is_L_trans << 2) |
             ((u64)L_sign << 5) |
             ((u64)R_sign << 6) |
             ((u64)bias_sign << 7) |
             ((u64)Y_prec << 8) |
             ((u64)LR_prec << 11) |
             ((u64)is_L_const << 20) |
             ((u64)is_bias_const << 22) |
             ((u64)sym_range << 61);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)R_tensor_C << 16) |
            ((u64)R_tensor_W << 48);
      high[2] = ((is_L_trans ? (u64)L_col_num : (u64)L_row_num)) |
             ((u64)L_tensor_C << 16) |
             ((u64)L_tensor_W << 48);
      low[3] = (u64)L_last_W << 48;
      low[4] = ((u64)round_mode << 32);
      high[4] = ((u64)Y_addr) | ((u64)L_addr << 32);
      low[5] = ((u64)R_addr) | ((u64)bias_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            ((u64)sym_range << 1) |
            ((u64)if_relu << 4) |
            ((u64)do_rq << 5) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)MM << 41) |
            ((u64)MM_NORMAL << 45) |
            ((u64)add_result << 50) |
            ((u64)is_L_trans << 51) |
            ((u64)is_L_const << 52) |
            ((u64)is_bias_const << 53) |
            ((u64)Res_sign << 55) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)L_sign) |
             ((u64)R_sign << 1) |
             ((u64)bias_sign << 2) |
             ((u64)Y_prec << 3) |
             ((u64)LR_prec << 6) |
             ((u64)round_mode << 9) |
             ((u64)R_tensor_C << 48);
      low[1] = ((u64)R_tensor_W) |
            ((is_L_trans ? (u64)L_col_num : (u64)L_row_num) << 16) |
            ((u64)L_tensor_C << 32) |
            ((u64)L_tensor_W << 48);
      high[1] = ((u64)L_last_W << 16) |
             ((u64)Y_addr << 32);
      low[2] = ((u64)L_addr) | ((u64)R_addr << 32);
      high[2] = ((u64)bias_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, MM,
      MM_NORMAL, Y_prec, pid_node, high, low, elt);
}
