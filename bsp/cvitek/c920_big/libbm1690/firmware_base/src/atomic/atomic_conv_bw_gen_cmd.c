#include "atomic_gen_cmd.h"
#include "atomic_conv_bw_gen_cmd.h"
#include "conv_util.h"

void atomic_conv_bw_gen_cmd(
  u32 input_addr,   //opad0
  u32 grad_addr,    //opad1
  u32 pad_ins_addr, //opad2
  u32 res_addr,
  int n,
  int ic,
  int ih,
  int iw,
  int oc,
  int oh,
  int ow,
  int kh,
  int kw,
  int ins_h,
  int ins_w,
  int dh,
  int dw,
  int stride_h,
  int stride_w,
  int pad_h_t,
  int pad_h_b,
  int pad_w_l,
  int pad_w_r,
  int pad_ins_is_const,
  int result_add,
  u32 insert_const_val,
  int *input_stride,
  PAD_MODE pad_mode,
  PREC input_prec,
  PREC res_prec,
  FP8_TYPE input_fp8_type,
  FP8_TYPE grad_fp8_type,
  FP8_TYPE res_fp8_type,
  int thread_id,
  CMD_ID_NODE * pid_node) {

  FW_DBG("%s: "
          "input_addr = 0x%08x, grad_addr = 0x%08x, "
          "pad_ins_addr = 0x%08x, res_addr = 0x%08x, "
          "n = %d, ic = %d, ih = %d, iw = %d, oc = %d, oh = %d, ow = %d, "
          "kh = %d, kw = %d, ins_h = %d, ins_w = %d, dh = %d, dw = %d, "
          "stride_h = %d, stride_w = %d, pad = (%d %d %d %d), "
          "pad_ins_is_const = %d, result_add = %d, insert_const_val = %08x, "
          "pad_mode = %d, input_prec = %d, res_prec = %d, "
          "input_fp8_type = %d, grad_fp8_type = %d, res_fp8_type = %d\n ",
          __func__,
          input_addr, grad_addr,
          pad_ins_addr, res_addr,
          n, ic, ih, iw, oc, oh, ow,
          kh, kw, ins_h, ins_w, dh, dw,
          stride_h, stride_w, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
          pad_ins_is_const, result_add, insert_const_val,
          pad_mode, input_prec, res_prec,
          input_fp8_type, grad_fp8_type, res_fp8_type);
  int str[4] = {0};
  int tsk_eu_typ = CONV_BW;
  if (input_stride) {
    memcpy(str, input_stride, 4 * sizeof(int));
  }
  if (input_prec == TF32) {
    input_prec = FP32;
    res_prec = FP32;
    tsk_eu_typ = CONV_DW_TF32;
  }

#ifdef USING_CMODEL
  if (input_stride) {
    ASSERT(input_addr % get_bytesize(input_prec) == 0);
  } else {
    ASSERT(input_addr % ALIGN_BYTES == 0);
  }

  ASSERT(get_npu_index(grad_addr) == get_npu_index(res_addr));
  ASSERT((grad_addr % ALIGN_BYTES == 0) && (res_addr % ALIGN_BYTES == 0));

  if (!pad_ins_is_const) {
    ASSERT(get_npu_index(pad_ins_addr) == get_npu_index(input_addr));
    ASSERT(pad_ins_addr % (get_bytesize(input_prec) * 2) == 0);
  }

  ASSERT(n < (((int)1) << 16) && (n > 0));
  ASSERT(ic < (((int)1) << 16) && (ic > 0));
  ASSERT(ih < (((int)1) << 16) && (ih > 0));
  ASSERT(iw < (((int)1) << 16) && (iw > 0));
  int ih_ext = cal_conv2d_input_ext(ih, ins_h, pad_h_t, pad_h_b);
  int iw_ext = cal_conv2d_input_ext(iw, ins_w, pad_w_r, pad_w_l);
  ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
  ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
  ASSERT(oc < (((int)1) << 16) && (oc > 0));
  ASSERT(oh < (((int)1) << 16) && (oh > 0));
  ASSERT(ow < (((int)1) << 16) && (ow > 0));
  ASSERT(kh < (((int)1) << 16) && (kw > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(ins_h >= 0 && ins_h < 15);
  ASSERT(ins_w >= 0 && ins_w < 15);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(pad_ins_is_const == 0 || pad_ins_is_const == 1);
  ASSERT(pad_mode >= 0 && pad_mode < 4);
  ASSERT((input_prec == FP32 || input_prec == TF32 || input_prec == FP16 || input_prec == BFP16 || input_prec == FP8) && res_prec == FP32);
  // only support FP32 output
  ASSERT(res_prec == FP32);
  ASSERT(input_fp8_type == FP8E5M2 || input_fp8_type == FP8E4M3);
  ASSERT(grad_fp8_type == FP8E5M2 || grad_fp8_type == FP8E4M3);
  ASSERT(res_fp8_type == 0 || res_fp8_type == 1);
  // do not support result_add
  ASSERT(result_add == 0);
#endif
  //TODO CONV_BW_GET_PROFILE
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  int elt = 8;
  u64 high[8] = {0};
  u64 low[8] = {0};
  int input_short_str = input_stride ? 3 : 0;
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)CONV << 41) |
          ((u64)tsk_eu_typ << 45) |
          ((u64)pad_mode << 53) |
          ((u64)res_fp8_type << 55) |
          ((u64)bd_power_step() << 59);
      high[0] = (u64)result_add |
          ((u64)input_fp8_type << 5) |
          ((u64)grad_fp8_type << 6) |
          ((u64)res_prec << 8) |
          ((u64)input_prec << 11) |
          ((u64)input_prec << 14) |
          ((u64)input_short_str << 26) |
          ((u64)pad_ins_is_const << 62);
      low[1] = ((u64)ins_w) |
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
      low[2] = ((u64)oc << 16) |
            ((u64)kh << 32) |
            ((u64)kw << 48);
      high[2] = ((u64)n) |
            ((u64)ic << 16) |
            ((u64)ih << 32) |
            ((u64)iw << 48);
      low[3] = ((u64)oc << 16) |
            ((u64)oh << 32) |
            ((u64)ow << 48);
      high[3] = ((u64)str[0] << 32) |
          ((u64)str[1] << 48);
      low[4] = low[3];
      high[4] = (u64)res_addr |
          ((u64)input_addr << 32);
      low[5] = (u64)grad_addr;
      low[6] = (u64)str[2] |
          ((u64)str[3] << 32);
      high[7] = (u64)insert_const_val |
          ((u64)pad_ins_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  int elt = 4;
  u64 low[4] = {0}, high[4] = {0};
  int input_short_str = input_stride ? 3 : 0;
  BEGIN_FAST_GEN_CMD_BD(thread_id)
    low[0] = 1ull |
          ((u64)input_prec << 9) |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)CONV << 41) |
          ((u64)tsk_eu_typ << 45) |
          ((u64)input_fp8_type << 50) |
          ((u64)grad_fp8_type << 51) |
          ((u64)pad_mode << 53) |
          ((u64)res_fp8_type << 55) |
          ((u64)bd_power_step() << 59) |
          ((u64)result_add << 63);
    high[0] = ((u64)res_prec << 1) |
          ((u64)input_prec << 4) |
          ((u64)pad_h_t << 9) |
          ((u64)pad_h_b << 13) |
          ((u64)pad_w_l << 17) |
          ((u64)pad_w_r << 21) |
          ((u64)stride_w << 25) |
          ((u64)stride_h << 29) |
          ((u64)pad_ins_is_const << 33) |
          ((u64)input_short_str << 34) |
          ((u64)res_addr << 37);
    low[1] = ((u64)ins_w) |
            ((u64)ins_h << 4) |
            ((u64)(dw - 1) << 8) |
            ((u64)(dh - 1) << 12) |
            ((u64)str[0] << 16) |
            ((u64)n << 32) |
            ((u64)oc << 48);
    high[1] = ((u64)kh) |
            ((u64)kw << 16) |
            ((u64)ic << 32) |
            ((u64)ih << 48);
    low[2] = ((u64)iw) |
          ((u64)oh << 16) |
          ((u64)ow << 32) |
          ((u64)str[1] << 48);
    high[2] = (u64)input_addr |
          ((u64)grad_addr << 32);
    low[3] = ((u64)insert_const_val << 32);
    high[3] = ((u64)pad_ins_addr) |
        ((u64)str[2] << 32) |
        ((u64)str[3] << 48);
    for (int i = 0; i < elt; ++i) {
      WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, CONV,
      tsk_eu_typ, res_prec, pid_node, high, low, elt);
}
