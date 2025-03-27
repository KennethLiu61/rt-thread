#include "firmware_common.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

void atomic_cw_transpose_gen_cmd(
    u32   A_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    PREC dtype,
    TRAN_OP op,
    int  thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG("%s: "
           "A_addr = 0x%08x, Y_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "op=%d\n",
           __func__,
           A_addr, Y_addr,
           input_n, input_c, input_h, input_w,
           op);
#ifdef USING_CMODEL
    int A_npu_idx = get_npu_index(A_addr);
    int Y_npu_idx = get_npu_index(Y_addr);
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(Y_addr % ALIGN_BYTES == 0);
    ASSERT((op == TRAN_C_W_TRANSPOSE  && A_npu_idx == 0) || (op == TRAN_W_C_TRANSPOSE && Y_npu_idx == 0));
    ASSERT((input_n) < (((int)1) << 16) && ((input_n) > 0));
    ASSERT((input_c) < (((int)1) << 16) && ((input_c) > 0));
    ASSERT((input_h) < (((int)1) << 16) && ((input_h) > 0));
    ASSERT((input_w) < (((int)1) << 16) && ((input_w) > 0));
    ASSERT(dtype == FP32 || dtype == FP16 || dtype == BFP16 || dtype == INT8 || dtype == FP8 || dtype == INT16 || dtype == INT32);
    ASSERT(op == TRAN_C_W_TRANSPOSE || op == TRAN_W_C_TRANSPOSE);
    ASSERT(A_addr != Y_addr);
#endif
    CW_TRANSPOSE_GET_PROFILE(input_n, input_c, input_h, input_w, dtype, op, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 high[8] = {0};
    u64 low[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)TRANS_BC << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)dtype << 8);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)input_n) |
            ((u64)input_w << 16) |
            ((u64)input_h << 32) |
            ((u64)input_c << 48);
        high[2] = ((u64)input_c << 16) |
            ((u64)input_w << 48);
        high[4] = (u64)Y_addr |
            ((u64)A_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = 1ull |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)TRANS_BC << 41) |
            ((u64)op << 45) |
            ((u64)dtype << 50) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)input_n) |
            ((u64)input_w << 16) |
            ((u64)input_h << 32) |
            ((u64)input_c << 48);
        low[1] = ((u64)input_c) |
            ((u64)input_w << 16);
        high[1] = (u64)Y_addr |
            ((u64)A_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, TRANS_BC,
      op, dtype, pid_node, high, low, elt);
}

