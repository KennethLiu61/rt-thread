#include "atomic_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

void atomic_mm2_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 Bias_addr,
  u32 RQ_addr, //save re_scale when const
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_bias_const,
  int add_result,
  int do_relu,
  int do_rq,
  int is_rq_const,
  PREC LR_prec,
  PREC B_prec,
  PREC Y_prec,
  FP8_TYPE L_fp8_type,
  FP8_TYPE R_fp8_type,
  FP8_TYPE Y_fp8_type,
  int thread_id,
  int tf32_mode,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s:  L_addr:0x%08x  R_addr:0x%08x  Y_addr:0x%08x  Bias_addr:0x%08x  RQ_addr:0x%08x  "
          "L_row_num:%d  L_col_num:%d  R_col_num:%d  is_L_trans:%d  is_R_trans:%d  "
          "is_L_const:%d  is_R_const:%d  is_bias_const:%d  add_result:%d  do_relu:%d  "
          "do_rq:%d  is_rq_const:%d "
          "LR_prec:%d  B_prec:%d  Y_prec:%d  "
          "L_fp8_type:%d  R_fp8_type:%d  Y_fp8_type:%d, tf32_mode:%d\n",
          __func__, L_addr, R_addr, Y_addr, Bias_addr, RQ_addr,
          L_row_num, L_col_num, R_col_num, is_L_trans, is_R_trans,
          is_L_const, is_R_const, is_bias_const, add_result, do_relu,
          do_rq, is_rq_const,
          LR_prec, B_prec, Y_prec,
          L_fp8_type, R_fp8_type, Y_fp8_type, tf32_mode);

  MM_OP mm_op = MM_NN;
  if (!is_L_trans && is_R_trans) mm_op = MM_NT;
  if (is_L_trans && is_R_trans) mm_op = MM_TT;
  if (tf32_mode != 0) {
    if (mm_op == MM_NN) mm_op = MM_NN_TF32;
    else if (mm_op == MM_NT) mm_op = MM_NT_TF32;
    else if (mm_op == MM_TT) mm_op = MM_TT_TF32;
  }
#ifdef USING_CMODEL
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(L_addr) == 0);
  }
  if (!is_R_const) {
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(R_addr) == 0);
  }
  if (!is_bias_const) {
    if (mm_op == MM_TT || mm_op == MM_TT_TF32) ASSERT(Bias_addr % get_bytesize(FP32) == 0);
    else ASSERT(Bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Bias_addr) == 0);
  }
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(Y_addr) == 0);

  if (!is_rq_const) {
    ASSERT(RQ_addr % get_bytesize(FP32) == 0);
    ASSERT(get_npu_index(RQ_addr) == 0);
  }

  if (!is_R_trans) ASSERT(!is_L_trans);
  ASSERT(LR_prec == FP32 || LR_prec == FP16 ||
         LR_prec == BFP16 || LR_prec == FP8);
  if (LR_prec == FP32) {
    ASSERT(Y_prec == FP32);
    ASSERT(B_prec == FP32);
  }
  if (LR_prec == FP16) {
    ASSERT(Y_prec == FP16 || Y_prec == FP32);
    ASSERT(B_prec == FP16 || B_prec == FP32);
  }
  if (LR_prec == BFP16) {
    ASSERT(Y_prec == BFP16 || Y_prec == FP32);
    ASSERT(B_prec == BFP16 || B_prec == FP32);
  }
  if (LR_prec == FP8) {
    ASSERT(Y_prec == FP8 || Y_prec == FP16 || Y_prec == FP32);
    ASSERT(B_prec == FP16 || B_prec == FP32);
  }
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_R_const < (1 << 1) && is_R_const >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
//   ASSERT(do_rq == 0);
  ASSERT(is_rq_const == 0 || is_rq_const == 1);

  ASSERT(L_fp8_type == FP8E4M3 || L_fp8_type == FP8E5M2);
  ASSERT(R_fp8_type == FP8E4M3 || R_fp8_type == FP8E5M2);
  ASSERT(Y_fp8_type == FP8E4M3 || Y_fp8_type == FP8E5M2);
#endif

  //write tgcr
  u32 value[1] = {RQ_addr};
  int indice[1] = {6};
  atomic_bd_trwr_gen_cmd(value, indice, 1, 0/*thread-id*/, pid_node);

  MM2_GET_PROFILE(L_row_num, L_col_num, R_col_num, LR_prec, mm_op, add_result, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)MM << 41) |
        ((u64)mm_op << 45) |
        ((u64)do_rq << 50) |
        ((u64)Y_fp8_type << 55) |
        ((u64)bd_power_step() << 59);
  high[0] = ((u64)add_result) |
          ((u64)do_relu << 1) |
          ((u64)L_fp8_type << 5) |
          ((u64)R_fp8_type << 6) |
          ((u64)Y_prec << 8) |
          ((u64)LR_prec << 11) |
          ((u64)LR_prec << 14) |
          ((u64)B_prec << 17) |
          ((u64)is_L_const << 20) |
          ((u64)is_R_const << 21) |
          ((u64)is_bias_const << 22) |
          ((u64)is_rq_const << 63);
  high[1] = bd_get_lane_mask();
  low[2] = ((u64)((mm_op == MM_TT || mm_op == MM_TT_TF32) ? R_col_num : L_row_num) << 16) |
        ((u64)((mm_op == MM_TT || mm_op == MM_TT_TF32) ? L_row_num : R_col_num) << 48);
  low[3] = ((u64)((mm_op == MM_NN || mm_op == MM_NN_TF32) ? L_col_num : R_col_num) << 16) |
        ((u64)((mm_op == MM_NN || mm_op == MM_NN_TF32) ? R_col_num : L_col_num) << 48);
  high[4] = ((u64)Y_addr) | ((u64)L_addr << 32);
  low[5] = ((u64)R_addr) | ((u64)Bias_addr << 32);
  BEGIN_FAST_GEN_CMD_BD(thread_id)
  for (int i = 0; i < 8; ++i) {
    WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 2;
  u64 low[2] = {0}, high[2] = {0};
  low[0] = (1ull) |
        ((u64)is_rq_const << 3) |
        ((u64)do_relu << 4) |
        ((u64)do_rq << 5) |
        ((u64)B_prec << 6) |
        ((u64)LR_prec << 9) |
        (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)MM << 41) |
        ((u64)mm_op << 45) |
        ((u64)L_fp8_type << 50) |
        ((u64)R_fp8_type << 51) |
        ((u64)Y_prec << 52) |
        ((u64)Y_fp8_type << 55) |
        ((u64)bd_power_step() << 59) |
        ((u64)is_bias_const << 63);
  high[0] = ((u64)((mm_op == MM_TT || mm_op == MM_TT_TF32) ? R_col_num : L_row_num)) |
          ((u64)((mm_op == MM_TT || mm_op == MM_TT_TF32) ? L_row_num : R_col_num) << 16) |
          ((u64)((mm_op == MM_NN || mm_op == MM_NN_TF32) ? L_col_num : R_col_num) << 32) |
          ((u64)((mm_op == MM_NN || mm_op == MM_NN_TF32) ? R_col_num : L_col_num) << 48);
  low[1] = ((u64)LR_prec) |
        ((u64)is_L_const << 3) |
        ((u64)is_R_const << 4) |
        ((u64)add_result << 5) |
        ((u64)Y_addr << 6) |
        ((u64)L_addr << 32);
  high[1] = ((u64)R_addr) | ((u64)Bias_addr << 32);
  BEGIN_FAST_GEN_CMD_BD(thread_id)
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, MM,
      mm_op, Y_prec, pid_node, high, low, elt);
}

/* if (!L_trans) zp=[1,NPU_NUM,1,R_col_num],and its stride is[0,0,W,1]
 * if (L_trans && R_trans) zp=[1,R_col_num,1,1], and it is compacted in local memory
 */
void atomic_mm2_fixed_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 rzp_addr,
  u32 Bias_addr,
  u32 RQ_addr,
  s8 shift_val,
  s16 yzp_val,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_zp_const,
  int L_sign,
  int R_sign,
  int add_result,
  int Res_sign,
  int Bias_sign,
  int is_bias_const,
  int is_rq_const,
  int do_relu,
  int sym_range,
  int do_rq,
  ROUND_MODE rshift_rd,
  PREC L_prec,
  PREC R_prec,
  PREC Y_prec,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s:  L_addr:0x%08x  R_addr:0x%08x  Y_addr:0x%08x  rzp_addr:0x%08x  Bias_addr:0x%08x  RQ_addr:0x%08x  "
          "shift_val:%d  yzp_val:%d  L_row_num:%d  L_col_num:%d  R_col_num:%d  is_L_trans:%d  is_R_trans:%d  "
          "is_L_const:%d  is_R_const:%d  is_zp_const:%d  L_sign:%d  R_sign:%d  add_result:%d  Res_sign:%d  "
          "Bias_sign:%d  is_bias_const:%d  is_rq_const:%d  do_relu:%d  sym_range:%d  do_rq:%d  "
          "rshift_rd:%d  L_prec:%d  R_prec:%d  Y_prec:%d\n",
          __func__, L_addr, R_addr, Y_addr, rzp_addr, Bias_addr, RQ_addr,
          shift_val, yzp_val, L_row_num, L_col_num, R_col_num, is_L_trans, is_R_trans,
          is_L_const, is_R_const, is_zp_const, L_sign, R_sign, add_result, Res_sign,
          Bias_sign, is_bias_const, is_rq_const, do_relu, sym_range, do_rq,
          rshift_rd, L_prec, R_prec, Y_prec);


  MM_OP mm_op = MM_NN;
  if (!is_L_trans && is_R_trans) mm_op = MM_NT;
  if (is_L_trans && is_R_trans) mm_op = MM_TT;
  if (is_bias_const && Bias_addr == 0) Bias_sign = 0;

#ifdef USING_CMODEL
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(L_addr) == 0);
  }
  if (!is_R_const) {
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(R_addr) == 0);
  }
  if (!is_bias_const) {
    if (mm_op == MM_TT) ASSERT(Bias_addr % get_bytesize(INT32) == 0);
    else ASSERT(Bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Bias_addr) == 0);
  }
  if (!is_zp_const) {
    if (mm_op == MM_TT) ASSERT(rzp_addr % get_bytesize(INT16) == 0);
    else ASSERT(rzp_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(rzp_addr) == 0);
  }
  if (do_rq && !is_rq_const) {
    if (mm_op == MM_TT) ASSERT(RQ_addr % (sizeof(int) * 2) == 0);
    else ASSERT(RQ_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(RQ_addr) == 0);
  }

  ASSERT(Y_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(Y_addr) == 0);

  ASSERT(L_prec == INT8);
  ASSERT(R_prec == INT8);
  ASSERT(Y_prec == INT8 || Y_prec == INT16 || Y_prec == INT32);
  if (!is_R_trans) ASSERT(!is_L_trans);
  ASSERT(L_sign < (1 << 1) && L_sign >= 0);
  ASSERT(R_sign < (1 << 1) && R_sign >= 0);
  ASSERT(Bias_sign < (1 << 1) && Bias_sign >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(is_zp_const < (1 << 1) && is_zp_const >= 0);
  ASSERT(is_rq_const < (1 << 1) && is_rq_const >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_R_const < (1 << 1) && is_R_const >= 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(Res_sign < (1 << 1) && Res_sign >= 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
  ASSERT(sym_range < (1 << 1) && sym_range >= 0);
  ASSERT(do_rq < (1 << 1) && do_rq >= 0);
  ASSERT(rshift_rd < 7 && rshift_rd >= 0);
#endif
  //write tgcr
  u32 value[4] = {rzp_addr, RQ_addr, (u32)shift_val, (u32)yzp_val};
  int indice[4] = {5, 6, 32, 33};
  atomic_bd_trwr_gen_cmd(value, indice, !do_rq ? 1 : (is_rq_const ? 4 : 2), thread_id, pid_node);

  MM2_GET_PROFILE(L_row_num, L_col_num, R_col_num, L_prec, mm_op, add_result, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)MM << 41) |
        ((u64)mm_op << 45) |
        ((u64)do_rq << 50) |
        ((u64)Res_sign << 55) |
        ((u64)bd_power_step() << 59);
  high[0] = ((u64)add_result) |
          ((u64)do_relu << 1) |
          ((u64)is_zp_const << 3) |
          ((u64)L_sign << 5) |
          ((u64)R_sign << 6) |
          ((u64)Bias_sign << 7) |
          ((u64)Y_prec << 8) |
          ((u64)L_prec << 11) |
          ((u64)R_prec << 14) |
          ((u64)is_L_const << 20) |
          ((u64)is_R_const << 21) |
          ((u64)is_bias_const << 22) |
          ((u64)sym_range << 61) |
          ((u64)is_rq_const << 63);
  high[1] = bd_get_lane_mask();
  low[2] = ((u64)(mm_op == MM_TT ? R_col_num : L_row_num) << 16) |
        ((u64)(mm_op == MM_TT ? L_row_num : R_col_num) << 48);
  low[3] = ((u64)(mm_op == MM_NN ? L_col_num : R_col_num) << 16) |
        ((u64)(mm_op == MM_NN ? R_col_num : L_col_num) << 48);
  low[4] = ((u64)rshift_rd << 32);
  high[4] = ((u64)Y_addr) | ((u64)L_addr << 32);
  low[5] = ((u64)R_addr) | ((u64)Bias_addr << 32);
  BEGIN_FAST_GEN_CMD_BD(thread_id)
  for (int i = 0; i < 8; ++i) {
    WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 2;
  u64 low[2] = {0}, high[2] = {0};
  low[0] = (1ull) |
        ((u64)sym_range << 1) |
        ((u64)is_zp_const << 2) |
        ((u64)is_rq_const << 3) |
        ((u64)do_relu << 4) |
        ((u64)do_rq << 5) |
        ((u64)R_prec << 9) |
        ((u64)Bias_sign << 14) |
        (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)MM << 41) |
        ((u64)mm_op << 45) |
        ((u64)L_sign << 50) |
        ((u64)R_sign << 51) |
        ((u64)Y_prec << 52) |
        ((u64)Res_sign << 55) |
        ((u64)rshift_rd << 56) |
        ((u64)bd_power_step() << 59) |
        ((u64)is_bias_const << 63);
  high[0] = ((u64)(mm_op == MM_TT ? R_col_num : L_row_num)) |
          ((u64)(mm_op == MM_TT ? L_row_num : R_col_num) << 16) |
          ((u64)(mm_op == MM_NN ? L_col_num : R_col_num) << 32) |
          ((u64)(mm_op == MM_NN ? R_col_num : L_col_num) << 48);
  low[1] = ((u64)L_prec) |
        ((u64)is_L_const << 3) |
        ((u64)is_R_const << 4) |
        ((u64)add_result << 5) |
        ((u64)Y_addr << 6) |
        ((u64)L_addr << 32);
  high[1] = ((u64)R_addr) | ((u64)Bias_addr << 32);
  BEGIN_FAST_GEN_CMD_BD(thread_id)
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, MM,
      mm_op, Y_prec, pid_node, high, low, elt);
}
