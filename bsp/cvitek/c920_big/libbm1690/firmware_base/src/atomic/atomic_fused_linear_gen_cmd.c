#include "firmware_common.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

void atomic_fused_linear_gen_cmd(
    u32   A_addr,
    u32   B_addr,
    u32   C_addr,
    u32   R_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    int   B_is_const,
    int   C_is_const,
    PREC  input_prec,
    PREC  output_prec,
    FP8_TYPE fp8_type,
    LIN_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, C_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "B_is_const=%d, C_is_const=%d, input_prec=%d, output_prec=%d, op=%d, fp8_type=%d\n",
           __func__,
           A_addr, B_addr, C_addr, R_addr,
           input_n, input_c, input_h, input_w,
           B_is_const, C_is_const, input_prec, output_prec, op, fp8_type);
#ifdef USING_CMODEL
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
    if (!B_is_const) {
        ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
        ASSERT(B_addr % get_bytesize(input_prec) == 0);
    }
    if (op == LIN_MAC && !C_is_const) {
        ASSERT(get_npu_index(R_addr) == get_npu_index(C_addr));
        ASSERT(C_addr % get_bytesize(input_prec) == 0);
    }
    ASSERT(op == LIN_MAC || op == LIN_ADD_SQR || op == LIN_SUB_SQR);
    ASSERT((input_n) < (((int)1) << 16) && ((input_n) > 0));
    ASSERT((input_c) < (((int)1) << 16) && ((input_c) > 0));
    ASSERT((input_h) < (((int)1) << 16) && ((input_h) > 0));
    ASSERT((input_w) < (((int)1) << 16) && ((input_w) > 0));
    ASSERT(input_prec == FP32 || input_prec == FP16 || input_prec == BFP16  || input_prec == FP8);
    if (input_prec == FP32)
        ASSERT(output_prec == FP32);
    if (input_prec == FP16)
        ASSERT(output_prec == FP32 || output_prec == FP16);
    if (input_prec == BFP16)
        ASSERT(output_prec == FP32 || output_prec == BFP16);
    if (input_prec == FP8) {
        ASSERT(fp8_type == FP8E5M2 || fp8_type == FP8E4M3);
        ASSERT(output_prec == FP32 || output_prec == FP16);
    }

#endif
    FUSED_LINEAR_GET_PROFILE(input_n, input_c, input_h, input_w, input_prec, output_prec, R_addr, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)LIN << 41) |
              ((u64)op << 45) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)fp8_type << 5) |
                ((u64)output_prec << 8) |
                ((u64)input_prec << 11) |
               ((u64)B_is_const << 21) |
               ((u64)C_is_const << 22);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)input_n) |
              ((u64)input_c << 16) |
              ((u64)input_h << 32) |
              ((u64)input_w << 48);
        high[4] = ((u64)R_addr) |
               ((u64)A_addr << 32);
        low[5] = ((u64)B_addr) |
              ((u64)C_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 3;
    u64 low[3] = {0}, high[3] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = 1ull |
              (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)LIN << 41) |
              ((u64)op << 45) |
              ((u64)B_is_const << 50) |
              ((u64)C_is_const << 51) |
              ((u64)output_prec << 52) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)input_prec << 20) |
               ((u64)fp8_type << 23) |
               ((u64)input_n << 32) |
               ((u64)input_c << 48);
        low[1] = ((u64)input_h) |
              ((u64)input_w << 16)|
              ((u64)R_addr << 32);
        high[1] = ((u64)A_addr) |
               ((u64)B_addr << 32);
        low[2] = ((u64)C_addr);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, LIN,
      op, output_prec, pid_node, high, low, elt);
}

