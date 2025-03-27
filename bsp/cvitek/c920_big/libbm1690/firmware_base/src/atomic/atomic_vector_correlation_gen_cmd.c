#include "firmware_common.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"


void atomic_vector_correlation_gen_cmd(
        u32 A_addr,
        u32 B_addr,
        u32 R_addr,
        int A_len,
        int B_len,
        int A_w,
        int B_w,
        AR_OP op,
        PREC A_prec,
        PREC B_prec,
        PREC R_prec,
        ROUND_MODE round_mode,
        u32 select_val,
        int A_sign,
        int B_sign,
        int R_sign,
        int thread_id,
        CMD_ID_NODE* pid_node) {
    FW_DBG("%s: "
           "A_addr = 0x%x, B_addr = 0x%x, R_addr = 0x%x, "
           "A_len = %d, B_len = %d, A_w = %d, B_w = %d, "
           "op = %d, A_prec = %d, B_prec = %d, R_prec = %d, "
           "round_mode = %d, A_sign = %d, B_sign = %d, R_sign = %d \n",
           __func__,
           A_addr, B_addr, R_addr,
           A_len, B_len, A_w, B_w,
           op, A_prec, B_prec, R_prec,
           round_mode, A_sign, B_sign, R_sign);
    int A_c = (A_len + A_w - 1) / A_w;
    int B_c = (B_len + B_w - 1) / B_w;
    int A_w_last = A_len - (A_c - 1) * A_w;
    int opd2_n_str = 2;     // iter 3 for DIV OP
#ifdef USING_CMODEL
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(B_addr % ALIGN_BYTES == 0);
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(B_addr / LOCAL_MEM_SIZE == R_addr / LOCAL_MEM_SIZE);
    ASSERT(A_w > 0 && A_w < 65536);
    ASSERT(B_w > 0 && B_w < 65536);
    ASSERT(A_c > 0 && A_c < 65536);
    ASSERT(B_c > 0 && B_c < 65536);
    ASSERT(op == AR_MUL || op == AR_ADD || op == AR_SUB ||
           op == AR_MAX || op == AR_MIN || op == AR_AND ||
           op == AR_OR  || op == AR_XOR || op == AR_SG  ||
           op == AR_SE  || op == AR_DIV || op == AR_SL  ||
           op == AR_ADD_SATU || op == AR_SUB_SATU ||
           op == AR_MUL_SATU);
    ASSERT(A_sign == 0 || A_sign == 1);
    ASSERT(B_sign == 0 || B_sign == 1);
    ASSERT(R_sign == 0 || R_sign == 1);
#endif
    VEC_CORR_GET_PROFILE( (A_w*(A_c-1)+A_w_last), B_c, 1, B_w, R_addr, A_prec, B_prec, R_prec, op, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)VC << 41) |
            ((u64)op << 45) |
            ((u64)R_sign << 55) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)A_sign << 5) |
            ((u64)B_sign << 6) |
            ((u64)R_prec << 8) |
            ((u64)A_prec << 11) |
            ((u64)B_prec << 14);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)B_c << 16) |
            ((u64)B_w << 48);
        high[2] = ((u64)A_c << 16) |
            ((u64)A_w << 48);
        low[3] = ((u64)A_w_last << 48);
        low[4] = ((u64)opd2_n_str << 32);
        high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
        low[5] = (u64)B_addr |
            ((u64)select_val << 32);
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
            ((u64)VC << 41) |
            ((u64)op << 45) |
            ((u64)A_sign << 50) |
            ((u64)B_sign << 51) |
            ((u64)opd2_n_str << 52) |
            ((u64)R_sign << 55) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)R_prec) |
            ((u64)A_prec << 3) |
            ((u64)B_prec << 6) |
            ((u64)R_addr << 9);
        low[1] = (u64)A_addr |
            ((u64)B_c << 48);
        high[1] = (u64)B_w |
            ((u64)A_c << 16) |
            ((u64)A_w << 32) |
            ((u64)A_w_last << 48);
        low[2] = (u64)B_addr |
            ((u64)select_val << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, VC,
      op, R_prec, pid_node, high, low, elt);
}
