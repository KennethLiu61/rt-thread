#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

/* requantize with fp32 scale and zero-point
 * A is aligned in local memory, support S8/U8/S16/U16/S32
 * B is fp32, and aligned in local memory,
 * B=[0,C,1,2], B(0,c,0,0) is scale, B(0,c,0,1) is zp
 * R is aligned in local memory, support S4/U4/S8/U8/S16/U16
 */
void atomic_rq_f32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  float zp_value,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode,
  ROUND_MODE f2i_mode,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr%u, B_addr:%u, R_addr:%u, N:%d, C:%d, H:%d,"
         "W:%d, B_is_const:%d, scale_val:%f, zp_val:%f, A_sign:%d,"
         "res_sign:%d, sym_range:%d, A_prec:%d, R_prec:%d, i2f_mode:%d, f2i_mode:%d\n",
         __func__, A_addr, B_addr, R_addr, N, C, H, W, B_is_const,
         scale_value, zp_value, A_sign, R_sign, sym_range, A_prec, R_prec,
         i2f_mode, f2i_mode);

#ifdef USING_CMODEL
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(A_prec == INT16 || A_prec == INT32 || A_prec == INT8);
  ASSERT(R_prec == INT8 || R_prec == INT16 || R_prec == INT4);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(i2f_mode < (1 << 3) && i2f_mode >= 0);
  ASSERT(f2i_mode < (1 << 3) && f2i_mode >= 0);
  ASSERT(i2f_mode != ROUND_HALF_UP && i2f_mode != ROUND_HALF_DOWN);
  ASSERT(f2i_mode != ROUND_HALF_UP && f2i_mode != ROUND_HALF_DOWN);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(B_addr));
  }

#endif
  RQDQ_GET_PROFILE(A_addr, B_addr, R_addr, N, C, H, W, B_is_const, pid_node);

  u32 opd1_addr = B_addr;
  u32 opd2_addr = 0;
  if (B_is_const) {
    memcpy(&opd1_addr, &scale_value, sizeof(int));
    memcpy(&opd2_addr, &zp_value, sizeof(int));
  }
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd2_n_str = ((int)f2i_mode << 3) | (int)i2f_mode;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)RQ_0 << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)A_sign << 5) |
             ((u64)R_sign << 7) |
             ((u64)R_prec << 8) |
             ((u64)A_prec << 11) |
             ((u64)B_is_const << 21) |
             ((u64)sym_range << 61);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      low[4] = ((u64)opd2_n_str << 32);
      high[4] = ((u64)R_addr) | ((u64)A_addr << 32);
      low[5] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 2;
      u64 low[2] = {0}, high[2] = {0};
      low[0] = (1ull) |
            ((u64)sym_range << 1) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)RQ_0 << 45) |
            ((u64)A_sign << 50) |
            ((u64)R_sign << 51) |
            ((u64)R_prec << 52) |
            ((u64)bd_power_step() << 59) |
            ((u64)B_is_const << 63);
      high[0] = ((u64)A_prec) |
             ((u64)opd2_n_str << 3) |
             ((u64)R_addr << 10) |
             ((u64)A_addr << 36);
      low[1] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      high[1] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, RQDQ,
      RQ_0, R_prec, pid_node, high, low, elt);
}

/* RQ_1 with int scale-factor, right-shift and zp,
 * A is aligned with shape=[N, C, H, W], support S32/S16/U16/S8/U8
 * B is aligned with shape=[1, C, 1, 3], storaged as S32
 * scale_factor=B(0,c,0,0) support S8/S16/S32,
 * shift_num=B(0,c,0,1) support S8,
 * zp=B(0,c,0,2) support U8/U4, zp_sign=res_sign
 * R is aligned with shape=[N, C, H, W], support S4/U4/S8/U8/S16/U16
 */
void atomic_rq_i32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  int scale_val,
  char shift_val, // negative: right shift, positive: left shift
  short zp_val,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr:%u, B_addr;%u, R_addr:%u, N:%d, C:%d, H:%d, W:%d,"
         "B_is_const:%d, scale_val:%d, shift_va:%dm zp_val:%d,"
         "A_sign:%d, res_sign:%d, sym_range:%d, A_prec:%d, R_prec:%d, rd:%d\n",
         __func__, A_addr, B_addr, R_addr, N, C, H, W, B_is_const,
         scale_val, shift_val, zp_val, A_sign, R_sign, sym_range, A_prec,
         R_prec, shift_rd);

#ifdef USING_CMODEL
  ASSERT(shift_rd < (1 << 3) && shift_rd >= 0);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_prec == INT16 || A_prec == INT32 || A_prec == INT8);
  ASSERT(R_prec == INT4 || R_prec == INT8 || R_prec == INT16);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  }
#endif
  RQDQ_GET_PROFILE(A_addr, B_addr, R_addr, N, C, H, W, B_is_const, pid_node);

  u32 opd2_n_str = (u32)shift_rd;
  u32 opd1_addr = B_addr, opd2_addr = 0;
  if (B_is_const) {
    memcpy(&opd1_addr, &scale_val, sizeof(int));
    opd2_addr = (zp_val << 16) | (shift_val & 0xff);
  }
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)RQ_1 << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)A_sign << 5) |
             ((u64)R_sign << 7) |
             ((u64)R_prec << 8) |
             ((u64)A_prec << 11) |
             ((u64)B_is_const << 21) |
             ((u64)sym_range << 61);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      low[4] = ((u64)opd2_n_str << 32);
      high[4] = ((u64)R_addr) | ((u64)A_addr << 32);
      low[5] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 2;
      u64 low[2] = {0}, high[2] = {0};
      low[0] = (1ull) |
            ((u64)sym_range << 1) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)RQ_1 << 45) |
            ((u64)A_sign << 50) |
            ((u64)R_sign << 51) |
            ((u64)R_prec << 52) |
            ((u64)bd_power_step() << 59) |
            ((u64)B_is_const << 63);
      high[0] = ((u64)A_prec) |
             ((u64)opd2_n_str << 3) |
             ((u64)R_addr << 10) |
             ((u64)A_addr << 36);
      low[1] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      high[1] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, RQDQ,
      RQ_1, R_prec, pid_node, high, low, elt);
}

/* Dequantize with scale and output are float32
 * A=[N,C,H,W] with aligned in local memory, support INT4/INT8/INT16
 * B=[1,C,1,2] and B(0,c,0,0) with aligned in local memory, PREC is same with A,
 * R=[N,C,H,W] with aligned in local memory
 */
void atomic_dq_f32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  short zp_value, // INT8/INT16
  int A_sign,
  int R_sign,
  PREC A_prec,
  ROUND_MODE i2f_mode,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr:%u, B_addr:%u, R_addr:%u, N,%d, C:%d, H:%d, W:%d,"
         "B_is_const:%d, scale_val:%f, zp_value:%d, A_sign:%d,"
         "R_sign:%d, A_prec:%d, i2f_mode:%d\n",
         __func__, A_addr, B_addr, R_addr, N, C, H, W, B_is_const,
         scale_value, zp_value, A_sign, R_sign, A_prec, i2f_mode);

#ifdef USING_CMODEL
  ASSERT(A_prec == INT4 || A_prec == INT8 || A_prec == INT16);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(i2f_mode < (1 << 3) && i2f_mode >= 0);
  ASSERT(i2f_mode != ROUND_HALF_UP && i2f_mode != ROUND_HALF_DOWN);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  }
#endif
  RQDQ_GET_PROFILE(A_addr, B_addr, R_addr, N, C, H, W, B_is_const, pid_node);

  u32 opd2_n_str = (u32)i2f_mode;
  u32 opd1_addr = B_addr, opd2_addr = 0;
  if (B_is_const) {
    opd2_addr = zp_value & 0xffff;
    memcpy(&opd1_addr, &scale_value, sizeof(int));
  }
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)DQ_0 << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)A_sign << 5) |
             ((u64)R_sign << 7) |
             ((u64)FP32 << 8) |
             ((u64)A_prec << 11) |
             ((u64)B_is_const << 21);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      low[4] = ((u64)opd2_n_str << 32);
      high[4] = ((u64)R_addr) | ((u64)A_addr << 32);
      low[5] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
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
            ((u64)RQDQ << 41) |
            ((u64)DQ_0 << 45) |
            ((u64)A_sign << 50) |
            ((u64)R_sign << 51) |
            ((u64)FP32 << 52) |
            ((u64)bd_power_step() << 59) |
            ((u64)B_is_const << 63);
      high[0] = ((u64)A_prec) |
             ((u64)opd2_n_str << 3) |
             ((u64)R_addr << 10) |
             ((u64)A_addr << 36);
      low[1] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      high[1] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, RQDQ,
      DQ_0, FP32, pid_node, high, low, elt);
}

/* Dequantize for fixed output and zp ...
 * A=[N, C, H, W] with aligned in local memory, support INT4/INT8/INT16
 * B=[1, C, 1, 3] with aligned in local memory, B(0,c,0,0) is zp,
 * B(0,c,0,1) is scale_factor, B(0,c,0,2) is shift_num,
 * the prec is the same with A, but storaged as int32 in local mem
 * R=[N, C, H, W] with aligned in local memory, support S8/U8/S16/U16/S32
 */
void atomic_dq_i32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  short zp_value, // s16/u16
  int scale_factor, // S8/S16/S32
  char shift_num, // negative: right shift, positive: left shift
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr:%x, B_addr:%x, R_addr:%x, N:%d, C:%d,"
         "H:%d, W:%d, B_is_const:%d, zp_value:%d, scale_factor:%d,"
         "shift_num:%d, A_sign:%d, R_sign:%d, sym_range:%d, A_prec:%d, R_prec:%d, shift_rd:%d\n",
         __func__, A_addr, B_addr, R_addr, N, C, H, W, B_is_const,
         zp_value, scale_factor, shift_num, A_sign, R_sign, sym_range,
         A_prec, R_prec, shift_rd);

#ifdef USING_CMODEL
  ASSERT(shift_rd < (1 << 3) && shift_rd >= 0);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_prec == INT4 || A_prec == INT8 || A_prec == INT16);
  ASSERT(R_prec == INT32 || R_prec == INT16 || R_prec == INT8);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(B_addr));
  }
#endif
  RQDQ_GET_PROFILE(A_addr, B_addr, R_addr, N, C, H, W, B_is_const, pid_node);

  u32 opd1_addr = B_addr, opd2_addr = 0;
  if (B_is_const) {
    memcpy(&opd1_addr, &scale_factor, sizeof(int));
    opd2_addr = (zp_value << 16) | (shift_num & 0xff);
  }
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)DQ_1 << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)A_sign << 5) |
             ((u64)R_sign << 7) |
             ((u64)R_prec << 8) |
             ((u64)A_prec << 11) |
             ((u64)B_is_const << 21) |
             ((u64)sym_range << 61);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      low[4] = ((u64)shift_rd << 32);
      high[4] = ((u64)R_addr) | ((u64)A_addr << 32);
      low[5] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 2;
      u64 low[2] = {0}, high[2] = {0};
      low[0] = (1ull) |
            ((u64)sym_range << 1) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)RQDQ << 41) |
            ((u64)DQ_1 << 45) |
            ((u64)A_sign << 50) |
            ((u64)R_sign << 51) |
            ((u64)R_prec << 52) |
            ((u64)bd_power_step() << 59) |
            ((u64)B_is_const << 63);
      high[0] = ((u64)A_prec) |
             ((u64)shift_rd << 3) |
             ((u64)R_addr << 10) |
             ((u64)A_addr << 36);
      low[1] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
      high[1] = ((u64)opd1_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
          WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, RQDQ,
      DQ_1, R_prec, pid_node, high, low, elt);
}
