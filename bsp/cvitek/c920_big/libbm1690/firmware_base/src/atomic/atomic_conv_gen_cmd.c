#include "firmware_common.h"
#include "gen_cmd.h"
#include "atomic_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"

void atomic_conv_quant_gen_cmd(
    u32 input_addr,
    u32 weight_addr, // or weight const value
    u32 bias_addr, // or bias const value
    u32 pad_ins_addr, // pad const value
    u32 kzp_addr, // kzp const value
    u32 requant_addr, // multipiler const value
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kzp_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int res_sign,
    int *input_stride,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC weight_prec,
    PREC output_prec,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE * pid_node) {

    FW_DBG("%s: "
           "input_addr = 0x%08x, weight_addr = 0x%08x, kzp_addr = 0x%08x, "
           "pad_ins_addr = 0x%08x, output_addr = 0x%08x, "
           "iN = %d, iC = %d, iH = %d, iW = %d, oc = %d, "
           "kh = %d, kw = %d, stride_h = %d, stride_w = %d, ins_h = %d, ins_w = %d, "
           "dilation_h = %d, dilation_w = %d, pad = (%d %d %d %d), "
           "kernel_is_const = %d, kzp_is_const = %d, pad_ins_is_const = %d, kernel_rotate = %d, "
           "input_sign = %d, weight_sign = %d, out_prec = %d, pad_mode = %d, round_mode = %d, thread_id = %d\n",
           __func__,
           input_addr, weight_addr, kzp_addr,
           pad_ins_addr, output_addr,
           input_n, input_c, input_h, input_w, output_c,
           kh, kw, stride_h, stride_w, ins_h, ins_w,
           dilation_h, dilation_w, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
           kernel_is_const, kzp_is_const, pad_ins_is_const, kernel_rotate,
           input_sign, weight_sign, output_prec, pad_mode, rm_mode, thread_id);

    int kh_ext = dilation_h * (kh - 1) + 1;
    int kw_ext = dilation_w * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_r + pad_w_l + 1;
    int output_h = (ih_ext - kh_ext) / stride_h + 1;
    int output_w = (iw_ext - kw_ext) / stride_w + 1;
    int input_short_str = input_stride == NULL ? 0 : 3;
    u32 str[4] = {0};
    if (input_stride != NULL) {
        memcpy(str, input_stride, 4 * sizeof(int));
        str[0] = input_n == 1 ? 0 : str[0];
        str[1] = input_c <= NPU_NUM ? 0 : str[1];
    }
    if (bias_is_const && bias_addr == 0) {
        bias_sign = 0;
    }

#ifdef USING_CMODEL
    ASSERT(input_addr < LOCAL_MEM_SIZE);
    ASSERT(pad_ins_is_const || (pad_ins_addr < LOCAL_MEM_SIZE));
    ASSERT(output_addr < LOCAL_MEM_SIZE * NPU_NUM);
    ASSERT(bias_is_const || get_npu_index(bias_addr) == get_npu_index(output_addr));
    ASSERT(kernel_is_const || (get_npu_index(weight_addr) == get_npu_index(output_addr)));
    ASSERT(kzp_is_const || get_npu_index(kzp_addr) == get_npu_index(output_addr));
    ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == get_npu_index(output_addr));
    ASSERT(input_stride || input_addr % ALIGN_BYTES == 0);
    ASSERT(kernel_is_const || weight_addr % ALIGN_BYTES == 0);
    ASSERT(bias_is_const || bias_addr % sizeof(int) == 0);
    ASSERT(pad_ins_is_const || pad_ins_addr % (sizeof(char) * 2) == 0);
    ASSERT(kzp_is_const || kzp_addr % sizeof(short) == 0);
    ASSERT(!do_requant || requant_is_const || requant_addr % (sizeof(int) * 2) == 0);
    ASSERT(output_addr % ALIGN_BYTES == 0);
    ASSERT(input_prec == INT8);
    ASSERT(weight_prec == INT8);
    ASSERT(is_fixed_prec(output_prec) && output_prec != INT4);
    ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
    ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
    ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
    ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
    ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
    ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
    ASSERT(output_c < (((int)1) << 16) && (output_c > 0));
    ASSERT(output_h > 0 && output_h < (((int)1) << 16));
    ASSERT(output_w > 0 && output_w < (((int)1) << 16));
    ASSERT(stride_h > 0 && stride_h < 16);
    ASSERT(stride_w > 0 && stride_w < 16);
    ASSERT(pad_h_t >= 0 && pad_h_t < 16);
    ASSERT(pad_h_b >= 0 && pad_h_b < 16);
    ASSERT(pad_w_r >= 0 && pad_w_r < 16);
    ASSERT(pad_w_l >= 0 && pad_w_l < 16);
    ASSERT(dilation_h > 0 && dilation_h < 16);
    ASSERT(dilation_w > 0 && dilation_w < 16);
    ASSERT(ins_h >= 0 && ins_h < 15);
    ASSERT(ins_w >= 0 && ins_w < 15);
    ASSERT(kh > 0 && kh < 65536 && kw > 0 && kw < 65536);
    ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
    ASSERT(bias_is_const >= 0 && bias_is_const < 2);
    ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
    ASSERT(kzp_is_const >= 0 && kzp_is_const < 2);
    ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
    ASSERT(input_sign >= 0 && input_sign < 2);
    ASSERT(weight_sign >= 0 && weight_sign < 2);
    ASSERT(bias_sign >= 0 && bias_is_const < 2);
    ASSERT(res_sign >= 0 && res_sign < 2);
    ASSERT(do_relu >= 0 && do_relu < 2);
    ASSERT(sym_saturate >= 0 && sym_saturate < 2);
    ASSERT(do_requant >= 0 && do_requant < 2);
    ASSERT((requant_is_const >= 0 && requant_is_const < 2));
    ASSERT(shift_num >= -128 && shift_num < 128);
    ASSERT(ozp >= -32768 && ozp < 32768);
    ASSERT(str[0] < (1 << 16) && str[1] < (1 << 16) && str[2] < (1 << 16) && str[3] < (1 << 16));
    ASSERT(input_prec == INT8);
#endif

    // write tgcr first
    u32 value[4] = {kzp_addr, requant_addr, (u32)shift_num, (u32)ozp};
    int indice[4] = {5, 6, 32, 33};
    atomic_bd_trwr_gen_cmd(
        value,
        indice,
        !do_requant ? 1 : (requant_is_const ? 4 : 2), thread_id,
        pid_node);

    CONV_GET_PROFILE(input_n, output_c, output_h, output_w, input_c, kh, kw, INT8, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)CONV << 41) |
        ((u64)CONV_NORMAL << 45) |
        ((u64)do_requant << 50) |
        ((u64)pad_mode << 53) |
        ((u64)res_sign << 55) |
        ((u64)bd_power_step() << 59);
    high[0] = (u64)result_add |
        ((u64)do_relu << 1) |
        ((u64)kzp_is_const << 3) |
        ((u64)kernel_rotate << 4) |
        ((u64)input_sign << 5) |
        ((u64)weight_sign << 6) |
        ((u64)bias_sign << 7) |
        ((u64)output_prec << 8) |
        ((u64)input_prec << 11) |
        ((u64)weight_prec << 14) |
        ((u64)kernel_is_const << 21) |
        ((u64)bias_is_const << 22) |
        ((u64)input_short_str << 26) |
        ((u64)sym_saturate << 61) |
        ((u64)pad_ins_is_const << 62) |
        ((u64)requant_is_const << 63);
    low[1] = ((u64)ins_w) |
        ((u64)ins_h << 4) |
        ((u64)(dilation_w - 1) << 8) |
        ((u64)(dilation_h - 1) << 12) |
        ((u64)pad_h_t << 16) |
        ((u64)pad_h_b << 20) |
        ((u64)pad_w_l << 24) |
        ((u64)pad_w_r << 28) |
        ((u64)stride_w << 32) |
        ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n) |
        ((u64)output_c << 16) |
        ((u64)output_h << 32) |
        ((u64)output_w << 48);
    high[2] = ((u64)input_c << 16) |
        ((u64)input_h << 32) |
        ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
        ((u64)kw << 48);
    high[3] = ((u64)str[0] << 32) |
        ((u64)str[1] << 48);
    low[4] = (u64)rm_mode << 32;
    high[4] = (u64)output_addr |
        ((u64)input_addr << 32);
    low[5] = (u64)weight_addr |
        ((u64)bias_addr << 32);
    high[5] = 0ull;
    low[6] = (u64)str[2] |
        ((u64)str[3] << 32);
    high[6] = 0ull;
    low[7] = 0ull;
    high[7] = (u64)ins_const_val |
        ((u64)pad_ins_addr << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
        ((u64)sym_saturate << 1) |
        ((u64)kzp_is_const << 2) |
        ((u64)requant_is_const << 3) |
        ((u64)do_relu << 4) |
        ((u64)do_requant << 5) |
        ((u64)rm_mode << 6) |
        ((u64)weight_prec << 9) |
        (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)CONV << 41) |
        ((u64)CONV_NORMAL << 45) |
        ((u64)input_sign << 50) |
        ((u64)weight_sign << 51) |
        ((u64)bias_sign << 52) |
        ((u64)pad_mode << 53) |
        ((u64)res_sign << 55) |
        ((u64)bd_power_step() << 59) |
        ((u64)result_add << 63);
    high[0] = ((u64)kernel_rotate) |
        ((u64)output_prec << 1) |
        ((u64)input_prec << 4) |
        ((u64)kernel_is_const << 7) |
        ((u64)bias_is_const << 8) |
        ((u64)pad_h_t << 9) |
        ((u64)pad_h_b << 13) |
        ((u64)pad_w_l << 17) |
        ((u64)pad_w_r << 21) |
        ((u64)stride_w << 25) |
        ((u64)stride_h << 29) |
        ((u64)pad_ins_is_const << 33) |
        ((u64)input_short_str << 34) |
        ((u64)output_addr << 37);
    low[1] = ((u64)ins_w) |
        ((u64)ins_h << 4) |
        ((u64)(dilation_w - 1) << 8) |
        ((u64)(dilation_h - 1) << 12) |
        ((u64)str[0] << 16) |
        ((u64)input_n << 32) |
        ((u64)output_c << 48);
    high[1] = ((u64)output_h) |
        ((u64)output_w << 16) |
        ((u64)input_c << 32) |
        ((u64)input_h << 48);
    low[2] = ((u64)input_w) |
        ((u64)kh << 16) |
        ((u64)kw << 32) |
        ((u64)str[1] << 48);
    high[2] = (u64)input_addr |
        ((u64)weight_addr << 32);
    low[3] = ((u64)bias_addr) |
        ((u64)ins_const_val << 32);
    high[3] = ((u64)pad_ins_addr) |
        ((u64)str[2] << 32) |
        ((u64)str[3] << 48);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, CONV,
      CONV_NORMAL, output_prec, pid_node, high, low, elt);
}

void atomic_conv_gen_cmd(
    u32 input_addr,
    u32 weight_addr,
    u32 bias_addr,
    u32 pad_ins_addr,
    u32 rescale_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int *input_stride,
    int do_relu,
    PREC input_prec,
    PREC output_prec,
    PREC bias_prec,
    int input_sign,
    int weight_sign,
    int res_sign,
    int bias_sign,
    int do_rescale,
    int rescale_is_const,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE * pid_node) {

    FW_DBG("%s: "
           "input_addr = 0x%08x, weight_addr = 0x%08x, bias_addr = 0x%08x, "
           "pad_ins_addr = 0x%08x, output_addr = 0x%08x, "
           "iN = %d, iC = %d, iH = %d, iW = %d, oc = %d, "
           "kh = %d, kw = %d, stride_h = %d, stride_w = %d, ins_h = %d, ins_w = %d, "
           "dilation_h = %d, dilation_w = %d, pad = (%d %d %d %d), "
           "kernel_is_const = %d, bias_is_const = %d, pad_ins_is_const = %d, kernel_rotate = %d, "
           "in_prec = %d, out_prec = %d, pad_mode = %d, thread_id = %d\n",
           __func__,
           input_addr, weight_addr, bias_addr,
           pad_ins_addr, output_addr,
           input_n, input_c, input_h, input_w, output_c,
           kh, kw, stride_h, stride_w, ins_h, ins_w,
           dilation_h, dilation_w, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
           kernel_is_const, bias_is_const, pad_ins_is_const, kernel_rotate,
           input_prec, output_prec, pad_mode, thread_id);

    int kh_ext = dilation_h * (kh - 1) + 1;
    int kw_ext = dilation_w * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_r + pad_w_l + 1;
    int output_h = (ih_ext - kh_ext) / stride_h + 1;
    int output_w = (iw_ext - kw_ext) / stride_w + 1;
    int input_short_str = input_stride == NULL ? 0 : 3;
    int str[4] = {0};
    int tsk_eu_typ = CONV_NORMAL;
    if (input_stride != NULL) {
        memcpy(str, input_stride, 4 * sizeof(int));
        str[0] = input_n == 1 ? 0 : str[0];
        str[1] = (input_c + input_addr / LOCAL_MEM_SIZE) <= NPU_NUM ? 0 : str[1];
    }

#ifdef USING_CMODEL
    ASSERT(input_addr < LOCAL_MEM_SIZE * (input_prec == FP32 ? NPU_NUM : 1));
    ASSERT(pad_ins_is_const ||get_npu_index(pad_ins_addr) == get_npu_index(input_addr));
    ASSERT(output_addr < LOCAL_MEM_SIZE * NPU_NUM);
    ASSERT(kernel_is_const || get_npu_index(weight_addr) == get_npu_index(output_addr));
    ASSERT(bias_is_const || get_npu_index(bias_addr) == get_npu_index(output_addr));
    ASSERT((input_stride && (input_addr % get_bytesize(input_prec) == 0)) || (input_addr % ALIGN_BYTES == 0));
    ASSERT((input_prec == FP16 && (output_prec == FP32 || output_prec == FP16)) ||
           (input_prec == BFP16 && (output_prec == FP32 || output_prec == BFP16)) ||
           (input_prec == FP8 && (output_prec == FP32 || output_prec == FP16 || output_prec == FP8) ) ||
           (input_prec == FP32 && output_prec == FP32) ||
           (input_prec == TF32 && output_prec == FP32));
    ASSERT((input_prec == FP16 && (bias_prec == FP16 || bias_prec == FP32)) ||
           (input_prec == BFP16 && (bias_prec == BFP16 || bias_prec == FP32)) ||
           (input_prec == FP8 && (bias_prec == FP16 || bias_prec == FP32)) ||
           (input_prec == FP32 && bias_prec == FP32) ||
           (input_prec == TF32 && bias_prec == FP32));
    ASSERT(kernel_is_const || (weight_addr % (input_prec == FP32 ? sizeof(float) : ALIGN_BYTES) == 0));
    ASSERT(bias_is_const || (bias_addr % sizeof(float) == 0));
    ASSERT(pad_ins_is_const || pad_ins_addr % (get_bytesize(input_prec) * 2) == 0);
    ASSERT(output_addr % ALIGN_BYTES == 0);
    ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
    ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
    ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
    ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
    ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
    ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
    ASSERT(output_c < (((int)1) << 16) && (output_c > 0));
    ASSERT(output_h > 0 && output_h < (((int)1) << 16));
    ASSERT(output_w > 0 && output_w < (((int)1) << 16));
    ASSERT(stride_h > 0 && stride_h < 16);
    ASSERT(stride_w > 0 && stride_w < 16);
    ASSERT(pad_h_t >= 0 && pad_h_t < 16);
    ASSERT(pad_h_b >= 0 && pad_h_b < 16);
    ASSERT(pad_w_r >= 0 && pad_w_r < 16);
    ASSERT(pad_w_l >= 0 && pad_w_l < 16);
    ASSERT(dilation_h > 0 && dilation_h < 16);
    ASSERT(dilation_w > 0 && dilation_w < 16);
    ASSERT(ins_h >= 0 && ins_h < 15);
    ASSERT(ins_w >= 0 && ins_w < 15);
    ASSERT(kh > 0 && kh < 65536 && kw > 0 && kw < 65536);
    ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
    ASSERT(bias_is_const >= 0 && bias_is_const < 2);
    ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
    ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
    ASSERT(do_relu >= 0 && do_relu < 2);
    ASSERT(input_sign >= 0 && input_sign < 2);
    ASSERT(weight_sign >= 0 && weight_sign < 2);
    ASSERT(bias_sign >= 0 && bias_sign < 2);
    ASSERT(res_sign >= 0 && res_sign < 2);
    ASSERT(do_rescale == 0 || do_rescale == 1);
#endif

    if(input_prec == FP8 && do_rescale) {
        u32 value[1] = {rescale_addr};
        int indice[1] = {6};
        atomic_bd_trwr_gen_cmd(
            value,
            indice,
            1,
            thread_id,
            pid_node);
    }
    CONV_GET_PROFILE(input_n, output_c, output_h, output_w, input_c, kh, kw, input_prec, pid_node);

    if(input_prec == TF32) {
        input_prec = FP32;
        tsk_eu_typ = CONV_TF32;
    }

    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)CONV << 41) |
        ((u64)tsk_eu_typ << 45) |
        ((u64)do_rescale << 50) |
        ((u64)pad_mode << 53) |
        ((u64)res_sign << 55) |
        ((u64)bd_power_step() << 59);
    high[0] = (u64)result_add |
        ((u64)do_relu << 1) |
        ((u64)kernel_rotate << 4) |
        ((u64)input_sign << 5) |
        ((u64)weight_sign << 6) |
        ((u64)bias_sign << 7) |
        ((u64)output_prec << 8) |
        ((u64)input_prec << 11) |
        ((u64)input_prec << 14) |
        ((u64)bias_prec << 17) |
        ((u64)kernel_is_const << 21) |
        ((u64)bias_is_const << 22) |
        ((u64)input_short_str << 26) |
        ((u64)pad_ins_is_const << 62) |
        ((u64)rescale_is_const << 63);
    low[1] = ((u64)ins_w) |
        ((u64)ins_h << 4) |
        ((u64)(dilation_w - 1) << 8) |
        ((u64)(dilation_h - 1) << 12) |
        ((u64)pad_h_t << 16) |
        ((u64)pad_h_b << 20) |
        ((u64)pad_w_l << 24) |
        ((u64)pad_w_r << 28) |
        ((u64)stride_w << 32) |
        ((u64)stride_h << 36);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)input_n) |
        ((u64)output_c << 16) |
        ((u64)output_h << 32) |
        ((u64)output_w << 48);
    high[2] = ((u64)input_c << 16) |
        ((u64)input_h << 32) |
        ((u64)input_w << 48);
    low[3] = ((u64)kh << 32) |
        ((u64)kw << 48);
    high[3] = ((u64)str[0] << 32) |
        ((u64)str[1] << 48);
    high[4] = (u64)output_addr |
        ((u64)input_addr << 32);
    low[5] = (u64)weight_addr |
        ((u64)bias_addr << 32);
    low[6] = (u64)str[2] |
        ((u64)str[3] << 32);
    high[7] = (u64)ins_const_val |
        ((u64)pad_ins_addr << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
        ((u64)rescale_is_const << 3) |
        ((u64)do_relu << 4) |
        ((u64)do_rescale << 5) |
        ((u64)input_prec << 9) | //kernel_prec = input_prec
        (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
        ((u64)1ull << 37) |
        ((u64)CONV << 41) |
        ((u64)tsk_eu_typ << 45) |
        ((u64)input_sign << 50) |
        ((u64)weight_sign << 51) |
        ((u64)bias_sign << 52) |
        ((u64)pad_mode << 53) |
        ((u64)res_sign << 55) |
        ((u64)bias_prec << 56) |
        ((u64)bd_power_step() << 59) |
        ((u64)result_add << 63);
    high[0] = ((u64)kernel_rotate) |
        ((u64)output_prec << 1) |
        ((u64)input_prec << 4) |
        ((u64)kernel_is_const << 7) |
        ((u64)bias_is_const << 8) |
        ((u64)pad_h_t << 9) |
        ((u64)pad_h_b << 13) |
        ((u64)pad_w_l << 17) |
        ((u64)pad_w_r << 21) |
        ((u64)stride_w << 25) |
        ((u64)stride_h << 29) |
        ((u64)pad_ins_is_const << 33) |
        ((u64)input_short_str << 34) |
        ((u64)output_addr << 37);
    low[1] = ((u64)ins_w) |
        ((u64)ins_h << 4) |
        ((u64)(dilation_w - 1) << 8) |
        ((u64)(dilation_h - 1) << 12) |
        ((u64)str[0] << 16) |
        ((u64)input_n << 32) |
        ((u64)output_c << 48);
    high[1] = ((u64)output_h) |
        ((u64)output_w << 16) |
        ((u64)input_c << 32) |
        ((u64)input_h << 48);
    low[2] = ((u64)input_w) |
        ((u64)kh << 16) |
        ((u64)kw << 32) |
        ((u64)str[1] << 48);
    high[2] = (u64)input_addr |
        ((u64)weight_addr << 32);
    low[3] = (u64)bias_addr |
        ((u64)ins_const_val << 32);
    high[3] = ((u64)pad_ins_addr) |
        ((u64)str[2] << 32) |
        ((u64)str[3] << 48);

    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, CONV,
      tsk_eu_typ, output_prec, pid_node, high, low, elt);
}
