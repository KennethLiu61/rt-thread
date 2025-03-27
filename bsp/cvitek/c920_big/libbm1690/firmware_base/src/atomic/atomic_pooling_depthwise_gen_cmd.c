#include "atomic_pooling_depthwise_gen_cmd.h"
#include "atomic_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"
#include "firmware_common.h"
#include "bd_reg_def.h"

void atomic_max_min_pooling_gen_cmd(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 output_addr,
    u32 index_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int ins_const_val,
    int input_sign,
    PREC input_prec,
    PREC out_index_prec,
    PAD_MODE pad_mode,
    int do_relu,
    PD_OP pool_op,
    int thread_id,
    CMD_ID_NODE *pid_node) {

  FW_DBG("%s: "
         "input_addr = 0x%08x, pad_ins_addr = 0x%08x, output_addr = 0x%08x, index_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "kh = %d, kw = %d, "
         "stride_h = %d, stride_w = %d, "
         "ins_h = %d, ins_w = %d, "
         "dh = %d, dw = %d, "
         "pad_h_t = %d, pad_h_b = %d, pad_w_l = %d, pad_w_r = %d, "
         "pad_ins_is_const = %d, ins_const_val = 0x%08x, "
         "input_sign = %d, input_prec = %d, pad_mode = %d, out_index_prec = %d, "
         "do_relu = %d, pool_op = %d\n",
         __func__,
         input_addr, pad_ins_addr, output_addr, index_addr,
         input_n, input_c, input_h, input_w,
         kh, kw,
         stride_h, stride_w,
         ins_h, ins_w,
         dh, dw,
         pad_h_t, pad_h_b, pad_w_l, pad_w_r,
         pad_ins_is_const, ins_const_val,
         input_sign, input_prec, pad_mode, out_index_prec,
         do_relu, pool_op);

  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;

#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT((index_addr == 0xFFFFFFFF) || (index_addr / LOCAL_MEM_SIZE == start_npu_idx));
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT((index_addr == 0xFFFFFFFF) || (index_addr % ALIGN_BYTES == 0));
  ASSERT(input_prec == INT8 || input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(pool_op == PD_MIN_POOLING || pool_op == PD_MAX_POOLING);
#endif

  PorD_GET_PROFILE(input_n, input_c, output_h, output_w, kh, kw, stride_h, stride_w, output_addr, input_prec, PD_MAX_POOLING, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 res1_addr = (u32)ins_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)pool_op << 45) |
          ((u64)pad_mode << 53) |
          ((u64)input_sign << 55) | //output_sign = input_sign;
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_sign << 5) |
           ((u64)input_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)out_index_prec << 17) |
          //  ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)pad_ins_is_const << 62); // des_opt_opd3_const
    low[1] = ((u64)ins_w << 0) |
          ((u64)ins_h << 4) |
          ((u64)(dw-1) << 8) |
          ((u64)(dh-1) << 12) |
          ((u64)pad_h_t << 16) |
          ((u64)pad_h_b << 20) |
          ((u64)pad_w_l << 24) |
          ((u64)pad_w_r << 28) |
          ((u64)stride_w << 32) |
          ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)index_addr << 32);  //opd2 addr
    low[6] = low[5];
    high[7] = ((u64)res1_addr << 0) | // des_res1_addr
           ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)

#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) |
          ((u64)do_relu << 4) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)pool_op << 45) |
          ((u64)input_sign << 51) |
          ((u64)pad_mode << 53) |
          ((u64)input_sign << 55) | //res_sign = input_sign
          ((u64)out_index_prec << 56) | //des_opd2_addr = out_index_prec
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)pad_ins_is_const << 1) | // des_opt_opd3_const
           ((u64)input_prec << 2) | //res_prec = input_prec
           ((u64)input_prec << 5) |
           ((u64)ins_w << 8) |
           ((u64)ins_h << 12) |
           ((u64)(dw-1) << 16) |
           ((u64)(dh-1) << 20) |
           ((u64)pad_h_t << 24) |
           ((u64)pad_h_b << 28) |
           ((u64)pad_w_l << 32) |
           ((u64)pad_w_r << 36) |
           ((u64)stride_w << 40) |
           ((u64)stride_h << 44);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)index_addr << 32);
    low[3] = ((u64)res1_addr << 0) | // des_res1_addr
          ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      pool_op, input_prec, pid_node, high, low, elt);
}

//only for float
void atomic_avg_pooling_gen_cmd(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int kernel_const_val,
    int ins_const_val,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int sym_range,
    float re_scale,
    PREC input_prec,
    PREC output_prec,
    FP8_TYPE input_fp8_prec,
    FP8_TYPE kernel_fp8_prec,
    FP8_TYPE output_fp8_prec,
    PAD_MODE pad_mode,
    ROUND_MODE round_mode,
    int thread_id,
    CMD_ID_NODE *pid_node) {

  FW_DBG("%s: "
         "input_addr = 0x%08x, pad_ins_addr = 0x%08x, rq_addr = 0x%08x, output_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "kh = %d, kw = %d, "
         "stride_h = %d, stride_w = %d, "
         "ins_w = %d, ins_h = %d, "
         "dh = %d, dw = %d, "
         "pad_h_t = %d, pad_h_b = %d, pad_w_l = %d, pad_w_r = %d, "
         "pad_ins_is_const = %d, kernel_const_val = 0x%08x, "
         "ins_const_val = 0x%08x,  do_relu = %d, do_rq = %d, rq_is_const = %d, re_scale = %f, sym_range = %d,"
         "input_prec = %d, output_prec = %d, input_fp8_prec = %d, kernel_fp8_prec = %d, output_fp8_prec = %d,"
         "pad_mode = %d, round_mode = %d\n",
         __func__,
         input_addr, pad_ins_addr, rq_addr, output_addr,
         input_n, input_c, input_h, input_w,
         kh, kw,
         stride_h, stride_w,
         ins_h, ins_w,
         dh, dw,
         pad_h_t, pad_h_b, pad_w_l, pad_w_r,
         pad_ins_is_const, kernel_const_val,
         ins_const_val, do_relu, do_rq, rq_is_const, re_scale, sym_range,
         input_prec, output_prec, input_fp8_prec, kernel_fp8_prec, output_fp8_prec,
         pad_mode, round_mode);

  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
  ASSERT(output_prec == FP16 || output_prec == FP32 || output_prec == BFP16 || output_prec == FP8);
  if (input_prec == FP8) {
      ASSERT((do_rq && (output_prec == FP16 || output_prec == FP8)) ||
             (!do_rq && (output_prec == FP16 || output_prec == FP32)));
  } else {
      ASSERT(output_prec == FP32 || output_prec == input_prec);
      ASSERT(do_rq == 0);
  }
  if (do_rq && !rq_is_const) {
    ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
    ASSERT(rq_addr % get_bytesize(INT32) == 0);
  }
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(pad_mode >= 0 && pad_mode < (((int)1) << 2));
  ASSERT(round_mode == ROUND_HALF_TO_EVEN);
  ASSERT(input_fp8_prec == 0 || input_fp8_prec == 1);
  ASSERT(kernel_fp8_prec == 0 || kernel_fp8_prec == 1);
  ASSERT(output_fp8_prec == 0 || output_fp8_prec == 1);
#endif

  //write tgcr
  u32 T6_value;
  int indice = 6;
  if (rq_is_const) memcpy(&T6_value, &re_scale, sizeof(float));
  else memcpy(&T6_value, &rq_addr, sizeof(u32));
  atomic_bd_trwr_gen_cmd(&T6_value, &indice, do_rq ? 1 : 0, thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, output_h, output_w, kh, kw, stride_h, stride_w, output_addr, output_prec, PD_AVG_POOLING, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd1_addr = (u32)kernel_const_val;
  u32 res1_addr = (u32)ins_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_AVG_POOLING << 45) |
          ((u64)do_rq << 50) |
          ((u64)pad_mode << 53) |
          ((u64)output_fp8_prec << 55) | // fp8 output type
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_fp8_prec << 5) | // fp8 input type
           ((u64)kernel_fp8_prec << 6) | // fp8 kernel type
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)sym_range << 61) |
           ((u64)pad_ins_is_const << 62) | // des_opt_opd3_const
           ((u64)rq_is_const << 63);
    low[1] = ((u64)ins_w << 0) |
          ((u64)ins_h << 4) |
          ((u64)(dw-1) << 8) |
          ((u64)(dh-1) << 12) |
          ((u64)pad_h_t << 16) |
          ((u64)pad_h_b << 20) |
          ((u64)pad_w_l << 24) |
          ((u64)pad_w_r << 28) |
          ((u64)stride_w << 32) |
          ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)round_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)opd1_addr << 0); // des_opd1_addr;
    high[7] = ((u64)res1_addr << 0) | // des_res1_addr
           ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) |
          ((u64)sym_range << 1) |
          ((u64)rq_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_rq << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_AVG_POOLING << 45) |
          ((u64)input_fp8_prec << 51) | //input fp8_prec
          ((u64)kernel_fp8_prec << 52) | // kernerl fp8_prec
          ((u64)pad_mode << 53) |
          ((u64)output_fp8_prec << 55) | //output fp8_prec
          ((u64)bd_power_step() << 59) |
          ((u64)1ull << 63); // des_opt_opd1_const
    high[0] = ((u64)pad_ins_is_const << 1) | // des_opt_opd3_const
           ((u64)output_prec << 2) | //output_prec
           ((u64)input_prec << 5) | //input_prec
           ((u64)ins_w << 8) |
           ((u64)ins_h << 12) |
           ((u64)(dw-1) << 16) |
           ((u64)(dh-1) << 20) |
           ((u64)pad_h_t << 24) |
           ((u64)pad_h_b << 28) |
           ((u64)pad_w_l << 32) |
           ((u64)pad_w_r << 36) |
           ((u64)stride_w << 40) |
           ((u64)stride_h << 44) |
           ((u64)round_mode << 48);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)opd1_addr << 0); // des_opd1_addr
    low[3] = ((u64)res1_addr << 0) | // des_res1_addr
          ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_AVG_POOLING, output_prec, pid_node, high, low, elt);
}

//for fixed
void atomic_avg_pooling_fixed_gen_cmd(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int kernel_const_val,
    int ins_const_val,
    int input_sign,
    int output_sign,
    int kernel_sign,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int sym_range,
    int mul,
    s8  shift,
    s16 yzp,
    ROUND_MODE round_mode,
    PREC input_prec,
    PREC output_prec,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE *pid_node) {

  FW_DBG("%s: "
         "input_addr = 0x%08x, pad_ins_addr = 0x%08x, output_addr = 0x%08x, rq_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "kh = %d, kw = %d, "
         "stride_h = %d, stride_w = %d, "
         "ins_w = %d, ins_h = %d, "
         "dh = %d, dw = %d, "
         "pad_h_t = %d, pad_h_b = %d, pad_w_l = %d, pad_w_r = %d, "
         "pad_ins_is_const = %d, kernel_const_val = 0x%08x, "
         "ins_const_val = 0x%08x, input_sign = %d, output_sign = %d, kernel_sign = %d, "
         "do_relu = %d, do_rq = %d, rq_is_const = %d, sym_range = %d, "
         "mul = %d, shift = %d, yzp = %d, round_mode = %d, "
         "input_prec = %d, output_prec = %d, pad_mode = %d\n",
         __func__,
         input_addr, pad_ins_addr, output_addr, rq_addr,
         input_n, input_c, input_h, input_w,
         kh, kw,
         stride_h, stride_w,
         ins_h, ins_w,
         dh, dw,
         pad_h_t, pad_h_b, pad_w_l, pad_w_r,
         pad_ins_is_const, kernel_const_val,
         ins_const_val, input_sign, output_sign, kernel_sign,
         do_relu, do_rq, rq_is_const, sym_range,
         mul, shift, yzp, round_mode,
         input_prec, output_prec, pad_mode);

  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  if (do_rq && !rq_is_const) ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  if (do_rq && !rq_is_const) ASSERT(rq_addr % (2 * get_bytesize(INT32)) == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(pad_ins_is_const == 0 || pad_ins_is_const == 1);
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(output_sign == 0 || output_sign == 1);
  ASSERT(kernel_sign == 0 || kernel_sign == 1);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(round_mode < 7 && round_mode >= 0);
  ASSERT(pad_mode >= 0 && pad_mode < (((int)1) << 2));
#endif
  u32 opd1_addr = (u32)kernel_const_val;
  u32 opd3_addr = (u32)pad_ins_addr;
  u32 res1_addr = (u32)ins_const_val;
  //write tgcr
  u32 value[3] = {rq_is_const ? (u32)mul : rq_addr, (u32)shift, (u32)yzp};
  int indice[3] = {6, 32, 33};
  atomic_bd_trwr_gen_cmd(value, indice, !do_rq ? 0 : (rq_is_const ? 3 : 1), thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, output_h, output_w, kh, kw, stride_h, stride_w, output_addr, output_prec, PD_AVG_POOLING, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_AVG_POOLING << 45) |
          ((u64)do_rq << 50) |
          ((u64)pad_mode << 53) |
          ((u64)output_sign << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_sign << 5) |
           ((u64)kernel_sign << 6) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)sym_range << 61) |
           ((u64)pad_ins_is_const << 62) | // des_opt_opd3_const
           ((u64)rq_is_const << 63);
    low[1] = ((u64)ins_w << 0) |
          ((u64)ins_h << 4) |
          ((u64)(dw-1) << 8) |
          ((u64)(dh-1) << 12) |
          ((u64)pad_h_t << 16) |
          ((u64)pad_h_b << 20) |
          ((u64)pad_w_l << 24) |
          ((u64)pad_w_r << 28) |
          ((u64)stride_w << 32) |
          ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)round_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)opd1_addr << 0); // des_opd1_addr
    high[7] = ((u64)res1_addr << 0) | // des_res1_addr
           ((u64)opd3_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) |
          ((u64)sym_range << 1) |
          ((u64)rq_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_rq << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_AVG_POOLING << 45) |
          ((u64)input_sign << 51) |
          ((u64)kernel_sign << 52) |
          ((u64)pad_mode << 53) |
          ((u64)output_sign << 55) |
          ((u64)bd_power_step() << 59) |
          ((u64)1ull << 63); // des_opt_opd1_const
    high[0] = ((u64)pad_ins_is_const << 1) | // des_opt_opd3_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5) |
           ((u64)ins_w << 8) |
           ((u64)ins_h << 12) |
           ((u64)(dw-1) << 16) |
           ((u64)(dh-1) << 20) |
           ((u64)pad_h_t << 24) |
           ((u64)pad_h_b << 28) |
           ((u64)pad_w_l << 32) |
           ((u64)pad_w_r << 36) |
           ((u64)stride_w << 40) |
           ((u64)stride_h << 44) |
           ((u64)round_mode << 48);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)opd1_addr << 0); // des_opd1_addr
    low[3] = ((u64)res1_addr << 0) | // des_res1_addr
          ((u64)opd3_addr << 32);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_AVG_POOLING, output_prec, pid_node, high, low, elt);
}

void atomic_depthwise_gen_cmd(
    u32 input_addr,
    u32 weight_addr,  // if kernel_is_const, store weight value
    u32 bias_addr,    // if bias_is_const, store bias value
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int ins_const_val,
    int kernel_rotate,
    int do_relu,
    int do_rq,
    int rq_is_const,
    PREC in_prec,
    PREC out_prec,
    FP8_TYPE input_type,
    FP8_TYPE kernel_type,
    FP8_TYPE res_type,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE *pid_node)
{

      FW_DBG("%s: "
             "input_addr = 0x%08x, weight_addr = 0x%08x, bias_addr = 0x%08x, "
             "pad_ins_addr = 0x%08x, output_addr = 0x%08x, "
             "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
             "kh = %d, kw = %d, "
             "stride_h = %d, stride_w = %d, "
             "ins_h = %d, ins_w = %d, "
             "dh = %d, dw = %d, "
             "pad_h_t = %d, pad_h_b = %d, pad_w_l = %d, pad_w_r = %d, "
             "kernel_is_const = %d, bias_is_const = %d, "
             "pad_ins_is_const = %d, ins_const_val = 0x%08x, "
             "kernel_rotate = %d, if_relu = %d, "
             "input_prec = %d, input_sign = %d ,res_sign = %d, pad_mode = %d\n",
             __func__,
             input_addr, weight_addr, bias_addr,
             pad_ins_addr, output_addr,
             input_n, input_c, input_h, input_w,
             kh, kw,
             stride_h, stride_w,
             ins_h, ins_w,
             dh, dw,
             pad_h_t, pad_h_b, pad_w_l, pad_w_r,
             kernel_is_const, bias_is_const,
             pad_ins_is_const, ins_const_val,
             kernel_rotate, do_relu,
             in_prec, input_type, res_type, pad_mode);

      // compute the output_h, output_w
      int kh_ext = dh * (kh - 1) + 1;
      int kw_ext = dw * (kw - 1) + 1;
      int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
      int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
      int output_h = (ih_ext - kh_ext) / stride_h + 1;
      int output_w = (iw_ext - kw_ext) / stride_w + 1;

#ifdef USING_CMODEL
      int start_npu_idx = get_npu_index(input_addr);
      ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
      ASSERT(bias_is_const || get_npu_index(bias_addr) == start_npu_idx);
      ASSERT(pad_ins_is_const || get_npu_index(pad_ins_addr) == start_npu_idx);
      ASSERT(get_npu_index(output_addr) == start_npu_idx);
      ASSERT(input_addr % ALIGN_BYTES == 0);
      ASSERT(kernel_is_const || weight_addr % get_bytesize(in_prec) == 0);
      if (in_prec == FP8) {
            ASSERT(bias_is_const || (do_rq && (bias_addr % sizeof(float)) == 0)
                   || (!do_rq && (bias_addr % get_bytesize(out_prec)) == 0));
            ASSERT((!do_rq && (out_prec == FP16 || out_prec == FP32)) ||
                   (do_rq && (out_prec == FP16 || out_prec == FP8)));
            if (do_rq && !rq_is_const) {
                  ASSERT(rq_addr % sizeof(float) == 0 && get_npu_index(rq_addr) == start_npu_idx);
            }
      } else {
            ASSERT(do_rq == 0);
            ASSERT(bias_is_const || bias_addr % get_bytesize(out_prec) == 0);
      }
      ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(in_prec)) == 0);
      ASSERT(is_float_prec(in_prec) && is_float_prec(out_prec));
      ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
      ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
      ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
      ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
      ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
      ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
      ASSERT(kh < (((int)1) << 16) && (kh > 0));
      ASSERT(kw < (((int)1) << 16) && (kw > 0));
      ASSERT(stride_h > 0 && stride_h < 16);
      ASSERT(stride_w > 0 && stride_w < 16);
      ASSERT(ins_h >= 0 && ins_h < 8);
      ASSERT(ins_w >= 0 && ins_w < 8);
      ASSERT(dh > 0 && dh < 16);
      ASSERT(dw > 0 && dw < 16);
      ASSERT(pad_h_t >= 0 && pad_h_t < 16);
      ASSERT(pad_h_b >= 0 && pad_h_b < 16);
      ASSERT(pad_w_r >= 0 && pad_w_r < 16);
      ASSERT(pad_w_l >= 0 && pad_w_l < 16);
      ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
      ASSERT(bias_is_const >= 0 && bias_is_const < 2);
      ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
      ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
      ASSERT(do_relu >= 0 && do_relu < 2);
      ASSERT(do_rq >=0 && do_rq < 2);
      ASSERT(rq_is_const >=0 && rq_is_const < 2);
      ASSERT(input_type >=0 && input_type < 2);
      ASSERT(kernel_type >= 0 && kernel_type < 2);
      ASSERT(res_type >= 0 && res_type < 2);
#endif

      if (in_prec == FP8 && do_rq) {
            int indice = 6;
            atomic_bd_trwr_gen_cmd(&rq_addr, &indice, 1, thread_id, pid_node);
      }
      PorD_GET_PROFILE(input_n, input_c, output_h, output_w, kh, kw, stride_h,
                       stride_w, output_addr, in_prec, PD_DEPTHWISE, pid_node);
      const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
      u32 res1_addr = (u32)ins_const_val;
#ifndef FAST_GEN_CMD
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      BEGIN_FAST_GEN_CMD_BD(thread_id)
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)PD << 41) |
            ((u64)PD_DEPTHWISE << 45) |
            ((u64)do_rq << 50) |
            ((u64)pad_mode << 53) |
            ((u64)res_type << 55) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)do_relu << 1) |
             ((u64)kernel_rotate << 4) |
             ((u64)input_type << 5) |       // op0 sign
             ((u64)kernel_type << 6) |       // op1 sign
             ((u64)out_prec << 8)    |         // res0 data type
             ((u64)in_prec << 11)   |         // op0 data type
             ((u64)kernel_is_const << 21) | // des_opt_opd1_const
             ((u64)bias_is_const << 22) |   // des_opt_opd2_const
             ((u64)pad_ins_is_const << 62) | // des_opt_opd3_const
             ((u64)rq_is_const << 63);
      low[1] = ((u64)ins_w << 0) |
            ((u64)ins_h << 4) |
            ((u64)(dw - 1) << 8) |
            ((u64)(dh - 1) << 12) |
            ((u64)pad_h_t << 16) |
            ((u64)pad_h_b << 20) |
            ((u64)pad_w_l << 24) |
            ((u64)pad_w_r << 28) |
            ((u64)stride_w << 32) |
            ((u64)stride_h << 36);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)input_n << 0) |
            ((u64)input_c << 16) |
            ((u64)output_h << 32) |
            ((u64)output_w << 48);
      high[2] = ((u64)input_h << 32) |
             ((u64)input_w << 48);
      low[3] = ((u64)kh << 32) |
            ((u64)kw << 48);
      high[4] = ((u64)output_addr << 0) |
             ((u64)input_addr << 32);
      low[5] = ((u64)weight_addr << 0) | // des_opd1_addr
            ((u64)bias_addr << 32);   // des_opd2_addr
      high[7] = ((u64)res1_addr << 0) |    // des_res1_addr
             ((u64)pad_ins_addr << 32); // des_opd3_addr
      for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
      END_FAST_GEN_CMD_BD(pid_node)
#else
      int elt = 4;
      u64 low[4] = {0}, high[4] = {0};
      BEGIN_FAST_GEN_CMD_BD(thread_id)
      low[0] = ((u64)1ull << 0) | // des_short_cmd
            ((u64)rq_is_const << 3) |
            ((u64)do_relu << 4) |
            ((u64)do_rq << 5) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)PD << 41) |
            ((u64)PD_DEPTHWISE << 45)  |
            ((u64)kernel_rotate << 50) |
            ((u64)input_type << 51)    |
            ((u64)kernel_type << 52)    |
            ((u64)pad_mode << 53)      |
            ((u64)res_type << 55)      |    //res sign
            ((u64)bd_power_step() << 59) |
            ((u64)kernel_is_const << 63);   // des_opt_opd1_const
      high[0] = ((u64)bias_is_const << 0) |    // des_opt_opd2_const
             ((u64)pad_ins_is_const << 1) | // des_opt_opd3_const
             ((u64)out_prec << 2) |             // output_prec
             ((u64)in_prec << 5) |             // input_prec
             ((u64)ins_w << 8) |
             ((u64)ins_h << 12) |
             ((u64)(dw - 1) << 16) |
             ((u64)(dh - 1) << 20) |
             ((u64)pad_h_t << 24) |
             ((u64)pad_h_b << 28) |
             ((u64)pad_w_l << 32) |
             ((u64)pad_w_r << 36) |
             ((u64)stride_w << 40) |
             ((u64)stride_h << 44);
      low[1] = ((u64)input_n << 0) |
            ((u64)input_c << 16) |
            ((u64)output_h << 32) |
            ((u64)output_w << 48);
      high[1] = ((u64)input_h << 0) |
             ((u64)input_w << 16) |
             ((u64)kh << 32) |
             ((u64)kw << 48);
      low[2] = ((u64)output_addr << 0) |
            ((u64)input_addr << 32);
      high[2] = ((u64)weight_addr << 0) | // des_opd1_addr
             ((u64)bias_addr << 32);   // des_opd2_addr
      low[3] = ((u64)res1_addr << 0) |    // des_res1_addr
            ((u64)pad_ins_addr << 32); // des_opd3_addr
      for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
      END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_DEPTHWISE, out_prec, pid_node, high, low, elt);
}

void atomic_depthwise_quant_gen_cmd(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const, store weight value
    u32 bias_addr, // if bias_is_const, store bias value
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 requant_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int ins_const_val,
    int kernel_rotate,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int output_sign,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC output_prec,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE *pid_node) {

  FW_DBG("%s: "
         "input_addr = 0x%08x, weight_addr = 0x%08x, bias_addr = 0x%08x, "
         "pad_ins_addr = 0x%08x, requant_addr = 0x%08x, output_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "kh = %d, kw = %d, "
         "stride_h = %d, stride_w = %d, "
         "ins_h = %d, ins_w = %d, "
         "dh = %d, dw = %d, "
         "pad_h_t = %d, pad_h_b = %d, pad_w_l = %d, pad_w_r = %d, "
         "kernel_is_const = %d, bias_is_const = %d, "
         "pad_ins_is_const = %d, ins_const_val = 0x%08x, "
         "kernel_rotate = %d, input_sign = %d, kernel_sign = %d, "
         "bias_sign = %d, output_sign = %d, do_relu = %d,"
         "sym_saturate = %d, do_requant = %d, requant_is_const = %d, "
         "shift_num = %d, ozp = %d, rm_mode = %d, "
         "input_prec = %d, output_prec = %d, pad_mode = %d\n",
         __func__,
         input_addr, weight_addr, bias_addr,
         pad_ins_addr, requant_addr, output_addr,
         input_n, input_c, input_h, input_w,
         kh, kw,
         stride_h, stride_w,
         ins_h, ins_w,
         dh, dw,
         pad_h_t, pad_h_b, pad_w_l, pad_w_r,
         kernel_is_const, bias_is_const,
         pad_ins_is_const, ins_const_val,
         kernel_rotate, input_sign, weight_sign,
         bias_sign, output_sign, do_relu, sym_saturate,
         do_requant, requant_is_const, shift_num, ozp,
         rm_mode, input_prec, output_prec, pad_mode);

  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
  if (bias_is_const && bias_addr == 0) {
      bias_sign = 0;
  }

#ifdef USING_CMODEL
  int start_npu_idx = get_npu_index(input_addr);
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(bias_is_const || get_npu_index(bias_addr) == start_npu_idx);
  ASSERT(pad_ins_is_const || get_npu_index(pad_ins_addr) == start_npu_idx);
  ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == start_npu_idx);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0 && output_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(input_prec) == 0);
  ASSERT(bias_is_const || bias_addr % sizeof(int) == 0);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(!do_requant || requant_is_const || requant_addr % (sizeof(int) * 2) == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(bias_is_const >= 0 && bias_is_const < 2);
  ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(input_sign >= 0 && input_sign < 2);
  ASSERT(weight_sign >= 0 && weight_sign < 2);
  ASSERT(bias_sign >= 0 && bias_is_const < 2);
  ASSERT(output_sign >= 0 && output_sign < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(sym_saturate >= 0 && sym_saturate < 2);
  ASSERT(do_requant >= 0 && do_requant < 2);
  ASSERT(requant_is_const >= 0 && requant_is_const < 2);
  ASSERT(shift_num >= -128 && shift_num < 128);
  ASSERT(ozp >= -32768 && ozp < 32768);
#endif


  // write tgcr first
  u32 value[3] = {requant_addr, (u32)shift_num, (u32)ozp};
  int indice[3] = {6, 32, 33};
  atomic_bd_trwr_gen_cmd(
      value,
      indice,
      !do_requant ? 0 : (requant_is_const ? 3 : 1), thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, output_h, output_w, kh, kw, stride_h, stride_w, output_addr, output_prec, PD_DEPTHWISE, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 res1_addr = (ins_const_val & 0xff);
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_DEPTHWISE << 45) |
          ((u64)do_requant << 50) |
          ((u64)pad_mode << 53) |
          ((u64)output_sign << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)kernel_rotate << 4) |
           ((u64)input_sign << 5) |
           ((u64)weight_sign << 6) |
           ((u64)bias_sign << 7) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)kernel_is_const << 21) | // des_opt_opd1_const
           ((u64)bias_is_const << 22) | // des_opt_opd2_const
           ((u64)sym_saturate << 61) |
           ((u64)pad_ins_is_const << 62) | // des_opt_opd3_const
           ((u64)requant_is_const << 63);
    low[1] = ((u64)ins_w << 0) |
          ((u64)ins_h << 4) |
          ((u64)(dw-1) << 8) |
          ((u64)(dh-1) << 12) |
          ((u64)pad_h_t << 16) |
          ((u64)pad_h_b << 20) |
          ((u64)pad_w_l << 24) |
          ((u64)pad_w_r << 28) |
          ((u64)stride_w << 32) |
          ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)rm_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)weight_addr << 0) | // des_opd1_addr
          ((u64)bias_addr << 32); // des_opd2_addr
    high[7] = ((u64)res1_addr << 0) | // des_res1_addr
           ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)sym_saturate << 1) |
          ((u64)requant_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_requant << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_DEPTHWISE << 45) |
          ((u64)kernel_rotate << 50) |
          ((u64)input_sign << 51) |
          ((u64)weight_sign << 52) |
          ((u64)pad_mode << 53) |
          ((u64)output_sign << 55) |
          ((u64)bd_power_step() << 59) |
          ((u64)kernel_is_const << 63); // des_opt_opd1_const
    high[0] = ((u64)bias_is_const << 0) | // des_opt_opd2_const
           ((u64)pad_ins_is_const << 1) | // des_opt_opd3_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5) |
           ((u64)ins_w << 8) |
           ((u64)ins_h << 12) |
           ((u64)(dw-1) << 16) |
           ((u64)(dh-1) << 20) |
           ((u64)pad_h_t << 24) |
           ((u64)pad_h_b << 28) |
           ((u64)pad_w_l << 32) |
           ((u64)pad_w_r << 36) |
           ((u64)stride_w << 40) |
           ((u64)stride_h << 44) |
           ((u64)rm_mode << 48) |
           ((u64)bias_sign << 62);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)output_h << 32) |
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)weight_addr << 0) | // des_opd1_addr
           ((u64)bias_addr << 32); // des_opd2_addr
    low[3] = ((u64)res1_addr << 0) | // des_res1_addr
          ((u64)pad_ins_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_DEPTHWISE, output_prec, pid_node, high, low, elt);
}

void atomic_roi_max_min_pooling_gen_cmd(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int input_sign,
    PREC input_prec,
    int do_relu,
    PD_OP pool_op,
    int thread_id,
    CMD_ID_NODE *pid_node) {
  FW_DBG("%s: "
         "input_addr = 0x%08x, roi_addr = 0x%08x, output_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "output_w = %d, "
         "kh = %d, kw = %d, "
         "imm_const_val = 0x%08x, "
         "input_sign = %d, input_prec = %d, do_relu = %d, pool_op = %d\n",
         __func__,
         input_addr, roi_addr, output_addr,
         input_n, input_c, input_h, input_w,
         output_w,
         kh, kw,
         imm_const_val,
         input_sign, input_prec, do_relu, pool_op);
PREC output_prec = input_prec;
#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(input_prec == INT8 || input_prec == FP8 ||  input_prec == FP16 || input_prec == FP32 || input_prec == BFP16);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(pool_op == PD_ROI_MAX_POOLING || pool_op == PD_ROI_MIN_POOLING);
#endif

  PorD_GET_PROFILE(input_n, input_c, 1, output_w, kh, kw, 0, 0, output_addr, input_prec, PD_ROI_MAX_POOLING, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd2_addr = (u32)imm_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)pool_op << 45) |
          ((u64)input_sign << 55) | //output_sign = input_sign
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_sign << 5) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
          //  ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)1ull << 22); // des_opt_opd2_const
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)opd2_addr << 32); // des_opd2_addr
    high[7] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)do_relu << 4) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)pool_op << 45) |
          ((u64)input_sign << 51) |
          ((u64)input_sign << 55) | //output_sign = input_sign
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)1ull << 0) | // des_opt_opd2_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)opd2_addr << 32); // imm_const_val
    low[3] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      pool_op, output_prec, pid_node, high, low, elt);
}

//only for float
void atomic_roi_avg_pooling_gen_cmd(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int kernel_const_val,
    int imm_const_val,
    int do_rq,
    int rq_is_const,
    float re_scale,
    int sym_range,
    int do_relu,
    FP8_TYPE input_fp8_type,
    FP8_TYPE kernel_fp8_type,
    FP8_TYPE res_fp8_type,
    PREC input_prec,
    PREC output_prec,
    ROUND_MODE round_mode,
    int thread_id,
    CMD_ID_NODE *pid_node) {
  FW_DBG("%s: "
         "input_addr = 0x%08x, roi_addr = 0x%08x, output_addr = 0x%08x, rq_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "output_w = %d, "
         "kh = %d, kw = %d, "
         "kernel_const_val = 0x%08x, imm_const_val = 0x%08x, "
         "do_rq = %d, rq_is_const = %d, re_scale = 0x%f, sym_range = %d, "
         "do_relu = %d, input_fp8_type = %d, kernel_fp8_type = %d, res_fp8_type = %d, "
         "input_prec = %d, output_prec = %d, round_mode = %d\n",
         __func__,
         input_addr, roi_addr, output_addr, rq_addr,
         input_n, input_c, input_h, input_w,
         output_w,
         kh, kw,
         kernel_const_val, imm_const_val,
         do_rq, rq_is_const, re_scale, sym_range,
         do_relu, input_fp8_type, kernel_fp8_type, res_fp8_type,
         input_prec, output_prec, round_mode);
#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  if (do_rq && !rq_is_const) {
      ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
      ASSERT(rq_addr % get_bytesize(INT32) == 0);
  }
  ASSERT(is_float_prec(input_prec));
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(round_mode == ROUND_HALF_TO_EVEN);
  ASSERT(input_fp8_type == 0 || input_fp8_type == 1);
  ASSERT(kernel_fp8_type == 0 || kernel_fp8_type == 1);
  ASSERT(res_fp8_type == 0 || res_fp8_type == 1);
  if (input_prec == FP8) {
      ASSERT((do_rq && (output_prec == FP16 || output_prec == FP8)) ||
             (!do_rq && (output_prec == FP16 || output_prec == FP32)));
  } else {
      ASSERT(output_prec == FP32 || output_prec == input_prec);
      ASSERT(do_rq == 0);
  }
#endif
  //write tgcr
  u32 T6_value;
  int indice = 6;
  if (rq_is_const) memcpy(&T6_value, &re_scale, sizeof(float));
  else memcpy(&T6_value, &rq_addr, sizeof(u32));
  atomic_bd_trwr_gen_cmd(&T6_value, &indice, do_rq ? 1 : 0, thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, 1, output_w, kh, kw, 0, 0, output_addr, output_prec, PD_ROI_AVG_POOLING, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd1_addr = (u32)kernel_const_val;
  u32 opd2_addr = (u32)imm_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_AVG_POOLING << 45) |
          ((u64)do_rq << 50) |
          ((u64)res_fp8_type << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_fp8_type << 5) |
           ((u64)kernel_fp8_type << 6) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)1ull << 22) | // des_opt_opd2_const
           ((u64)sym_range << 61) |
           ((u64)rq_is_const << 63);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1 << 32) | // output_h
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)round_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)opd1_addr << 0) | // des_opd1_addr
          ((u64)opd2_addr << 32); // des_opd2_addr
    high[7] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)sym_range << 1) |
          ((u64)rq_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_rq << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_AVG_POOLING << 45) |
          ((u64)input_fp8_type << 51) | //input fp8_prec
          ((u64)kernel_fp8_type << 52) | //kernel fp8_prec
          ((u64)res_fp8_type << 55) | //res fp8_prec
          ((u64)bd_power_step() << 59) |
          ((u64)1ull << 63); // des_opt_opd1_const
    high[0] = ((u64)1ull << 0) | // des_opt_opd2_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5) |
           ((u64)round_mode << 48);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)opd1_addr << 0) | // des_opd1_addr
           ((u64)opd2_addr << 32); // imm_const_val
    low[3] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_ROI_AVG_POOLING, output_prec, pid_node, high, low, elt);
}

//for fixed
void atomic_roi_avg_pooling_quant_gen_cmd(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int kernel_const_val,
    int imm_const_val,
    int input_sign,
    int output_sign,
    int kernel_sign,
    PREC input_prec,
    PREC output_prec,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int mul,
    s8 shift,
    s16 yzp,
    int sym_range,
    ROUND_MODE round_mode,
    int thread_id,
    CMD_ID_NODE *pid_node) {
  FW_DBG("%s: "
         "input_addr = 0x%08x, roi_addr = 0x%08x, output_addr = 0x%08x, rq_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "output_w = %d, "
         "kh = %d, kw = %d, "
         "kernel_const_val = 0x%08x, imm_const_val = 0x%08x, "
         "input_sign = %d, output_sign = %d, kernel_sign = %d, input_prec = %d, output_prec = %d, "
         "do_relu = %d, do_rq = %d, rq_is_const = %d, "
         "mul = %d, shift = %d, yzp = %d, "
         "sym_range = %d, round_mode = %d\n",
         __func__,
         input_addr, roi_addr, output_addr, rq_addr,
         input_n, input_c, input_h, input_w,
         output_w,
         kh, kw,
         kernel_const_val, imm_const_val,
         input_sign, output_sign, kernel_sign, input_prec, output_prec,
         do_relu, do_rq, rq_is_const,
         mul, shift, yzp,
         sym_range, round_mode);
#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  if (do_rq && !rq_is_const) {
      ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
      ASSERT(rq_addr % (2 * get_bytesize(INT32)) == 0);
  }
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(output_sign == 0 || output_sign == 1);
  ASSERT(kernel_sign == 0 || kernel_sign == 1);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(round_mode < 7 && round_mode >= 0);
#endif
  u32 opd1_addr = (u32)kernel_const_val;
  u32 opd2_addr = (u32)imm_const_val;
  // write tgcr
  u32 value[3] = {rq_is_const ? (u32)mul : rq_addr, (u32)shift, (u32)yzp};
  int indice[3] = {6, 32, 33};
  atomic_bd_trwr_gen_cmd(value, indice, !do_rq ? 0 : (rq_is_const ? 3 : 1), thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, 1, output_w, kh, kw, 0, 0, output_addr, output_prec, PD_ROI_AVG_POOLING, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_AVG_POOLING << 45) |
          ((u64)do_rq << 50) |
          ((u64)output_sign << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)input_sign << 5) |
           ((u64)kernel_sign << 6) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)1ull << 21) | // des_opt_opd1_const
           ((u64)1ull << 22) | // des_opt_opd2_const
           ((u64)sym_range << 61) |
           ((u64)rq_is_const << 63);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1 << 32) | // output_h
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)round_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)opd1_addr << 0) | // des_opd1_addr
          ((u64)opd2_addr << 32); // des_opd2_addr
    high[7] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)sym_range << 1) |
          ((u64)rq_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_rq << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_AVG_POOLING << 45) |
          ((u64)input_sign << 51) |
          ((u64)kernel_sign << 52) |
          ((u64)output_sign << 55) | // des_cmd_id_en
          ((u64)bd_power_step() << 59) |
          ((u64)1ull << 63); // des_opt_opd1_const
    high[0] = ((u64)1ull << 0) | // des_opt_opd2_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5) |
           ((u64)round_mode << 48);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)opd1_addr << 0) | // des_opd1_addr
           ((u64)opd2_addr << 32); // des_opd2_addr
    low[3] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_ROI_AVG_POOLING, output_prec, pid_node, high, low, elt);
}

void atomic_roi_depthwise_gen_cmd(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const store weight value
    u32 roi_addr, // roi pairs
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int kernel_is_const,
    int kernel_rotate,
    int do_relu,
    int do_requant,
    int rq_is_const,
    PREC in_prec,
    PREC out_prec,
    FP8_TYPE in_type,
    FP8_TYPE kernel_type,
    FP8_TYPE res_type,
    int thread_id,
    CMD_ID_NODE *pid_node) {
  FW_DBG("%s: "
         "input_addr = 0x%08x, weight_addr = 0x%08x, roi_addr = 0x%08x, output_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "output_w = %d, "
         "kh = %d, kw = %d, "
         "imm_const_val = 0x%08x, "
         "kernel_is_const = %d, kernel_rotate = %d, "
         "do_relu = %d, prec = %d\n",
         __func__,
         input_addr, weight_addr, roi_addr, output_addr,
         input_n, input_c, input_h, input_w,
         output_w,
         kh, kw,
         imm_const_val,
         kernel_is_const, kernel_rotate,
         do_relu, in_prec);

#ifdef USING_CMODEL
  int start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(in_prec) == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(is_float_prec(in_prec) && is_float_prec(out_prec));
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(do_requant == 0 || do_requant == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(in_type == 0 || in_type == 1);
  ASSERT(kernel_type == 0 || kernel_type == 1);
  ASSERT(res_type == 0 || res_type == 1);
  if (in_prec == FP8) {
      ASSERT((do_requant && (out_prec == FP16 || out_prec == FP8)) ||
             (!do_requant && (out_prec == FP16 || out_prec == FP32)));
      if (do_requant && !rq_is_const) {
            ASSERT(get_npu_index(rq_addr) == start_npu_idx && rq_addr % sizeof(float) == 0);
      }
  } else {
      ASSERT(out_prec == FP32 || out_prec == in_prec);
      ASSERT(do_requant == 0);
  }
#endif

   if (in_prec == FP8 && do_requant) {
      int indice = 6;
      atomic_bd_trwr_gen_cmd(&rq_addr, &indice, 1, thread_id, pid_node);
   }
  PorD_GET_PROFILE(input_n, input_c, 1, output_w, kh, kw, 0, 0,
                   output_addr, in_prec, PD_ROI_DEPTHWISE, pid_node);

  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd2_addr = (u32)imm_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_DEPTHWISE << 45) |
          ((u64)do_requant << 50) |
          ((u64)res_type << 55) |     // res0 sign
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)kernel_rotate << 4) |
           ((u64)in_type << 5) |    // op0 sign
           ((u64)kernel_type << 6) |    // op1 sign
           ((u64)out_prec << 8)  |
           ((u64)in_prec << 11) |
           ((u64)kernel_is_const << 21) | // des_opt_opd1_const
           ((u64)1ull << 22) | // des_opt_opd2_const
           ((u64)rq_is_const << 63);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)weight_addr << 0) | // des_opd1_addr
          ((u64)opd2_addr << 32); // des_opd2_addr
    high[7] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)rq_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_requant << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_DEPTHWISE << 45) |
          ((u64)kernel_rotate << 50) |
          ((u64)in_type << 51) | // op0 sign
          ((u64)kernel_type << 52) | // op1 sign
          ((u64)res_type << 55) | // res0 sign
          ((u64)bd_power_step() << 59) |
          ((u64)kernel_is_const << 63); // des_opt_opd1_const
    high[0] = ((u64)1ull << 0) | // des_opt_opd2_const
           ((u64)out_prec << 2) | //output_prec
           ((u64)in_prec << 5); //input_prec
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)weight_addr << 0) | // des_opt_opd1_addr
           ((u64)opd2_addr << 32); // des_opt_opd2_addr
    low[3] = ((u64)roi_addr << 32); // des_opt_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_ROI_DEPTHWISE, out_prec, pid_node, high, low, elt);
}

void atomic_roi_depthwise_quant_gen_cmd(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const store weight value
    u32 roi_addr, // roi pairs
    u32 requant_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int kernel_is_const,
    int kernel_rotate,
    int input_sign,
    int kernel_sign,
    int res_sign,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC output_prec,
    int thread_id,
    CMD_ID_NODE *pid_node) {
  FW_DBG("%s: "
         "input_addr = 0x%08x, weight_addr = 0x%08x, roi_addr = 0x%08x, "
         "requant_addr = 0x%08x, output_addr = 0x%08x, "
         "input_n = %d, input_c = %d, input_h = %d, input_w = %d, "
         "output_w = %d, "
         "kh = %d, kw = %d, "
         "imm_const_val = 0x%08x, "
         "kernel_is_const = %d, kernel_rotate = %d, "
         "input_sign = %d, kernel_sign = %d, res_sign = %d, "
         "do_relu = %d, sym_saturate = %d, do_rq = %d, rq_is_const = %d, "
         "shift_num = %d, ozp = %d, rm_mode = %d, "
         "input_prec = %d, output_prec = %d\n",
         __func__,
         input_addr, weight_addr, roi_addr, requant_addr, output_addr,
         input_n, input_c, input_h, input_w,
         output_w,
         kh, kw,
         imm_const_val,
         kernel_is_const, kernel_rotate,
         input_sign, kernel_sign, res_sign,
         do_relu, sym_saturate, do_requant, requant_is_const,
         shift_num, ozp, rm_mode,
         input_prec, output_prec);

#ifdef USING_CMODEL
  int start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == start_npu_idx);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(input_prec) == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(!do_requant || requant_is_const || requant_addr % (2 * sizeof(int)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(sym_saturate >= 0 && sym_saturate < 2);
  ASSERT(do_requant >= 0 && do_requant < 2);
  ASSERT(requant_is_const >= 0 && requant_is_const < 2);
  ASSERT(shift_num >= -128 && shift_num < 128);
  ASSERT(ozp >= -32768 && ozp < 32768);
  ASSERT(input_sign >= 0 && input_sign < 2);
  ASSERT(kernel_sign >= 0 && kernel_sign < 2);
  ASSERT(res_sign >= 0 && res_sign < 2);
#endif

  // write tgcr first
  u32 value[3] = {requant_addr, (u32)shift_num, (u32)ozp};
  int indice[3] = {6, 32, 33};
  atomic_bd_trwr_gen_cmd(
      value,
      indice,
      !do_requant ? 0 : (requant_is_const ? 3 : 1), thread_id, pid_node);

  PorD_GET_PROFILE(input_n, input_c, 1, output_w, kh, kw, 0, 0, output_addr, output_prec, PD_ROI_DEPTHWISE, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
  u32 opd2_addr = (u32)imm_const_val;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 low[8] = {0}, high[8] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_DEPTHWISE << 45) |
          ((u64)do_requant << 50) |
          ((u64)res_sign << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)do_relu << 1) |
           ((u64)kernel_rotate << 4) |
           ((u64)input_sign << 5) |
           ((u64)kernel_sign << 6) |
           ((u64)output_prec << 8) |
           ((u64)input_prec << 11) |
           ((u64)kernel_is_const << 21) | // des_opt_opd1_const
           ((u64)1ull << 22) | // des_opt_opd2_const
           ((u64)sym_saturate << 61) |
           ((u64)requant_is_const << 63);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[2] = ((u64)input_h << 32) |
           ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
          ((u64)kw << 48);
    low[4] = ((u64)rm_mode << 32);
    high[4] = ((u64)output_addr << 0) |
           ((u64)input_addr << 32);
    low[5] = ((u64)weight_addr << 0) | // des_opd1_addr
          ((u64)opd2_addr << 32); // des_opd2_addr
    high[7] = ((u64)roi_addr << 32); // des_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = ((u64)1ull << 0) | // des_short_cmd
          ((u64)sym_saturate << 1) |
          ((u64)requant_is_const << 3) |
          ((u64)do_relu << 4) |
          ((u64)do_requant << 5) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)PD << 41) |
          ((u64)PD_ROI_DEPTHWISE << 45) |
          ((u64)kernel_rotate << 50) |
          ((u64)input_sign << 51) |
          ((u64)kernel_sign << 52) |
          ((u64)res_sign << 55) |
          ((u64)bd_power_step() << 59) |
          ((u64)kernel_is_const << 63); // des_opt_opd1_const
    high[0] = ((u64)1ull << 0) | // des_opt_opd2_const
           ((u64)output_prec << 2) |
           ((u64)input_prec << 5) |
           ((u64)rm_mode << 48);
    low[1] = ((u64)input_n << 0) |
          ((u64)input_c << 16) |
          ((u64)1ull << 32) | // output_h
          ((u64)output_w << 48);
    high[1] = ((u64)input_h << 0) |
           ((u64)input_w << 16) |
           ((u64)kh << 32) |
           ((u64)kw << 48);
    low[2] = ((u64)output_addr << 0) |
          ((u64)input_addr << 32);
    high[2] = ((u64)weight_addr << 0) | // des_opt_opd1_addr
           ((u64)opd2_addr << 32); // des_opt_opd2_addr
    low[3] = ((u64)roi_addr << 32); // des_opt_opd3_addr
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, PD,
      PD_ROI_DEPTHWISE, output_prec, pid_node, high, low, elt);
}
