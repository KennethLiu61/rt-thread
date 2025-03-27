#include "firmware_common.h"
#include "atomic_tensor_arithmetic_gen_cmd.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

#define CHECK_AR_STRIDE(p_stride) \
      ASSERT((p_stride[0] < (((int)1) << 18)) && (p_stride[0] >= 0)); \
      ASSERT((p_stride[1] < (((int)1) << 18)) && (p_stride[1] >= 0)); \
      ASSERT((p_stride[2] < (((int)1) << 18)) && (p_stride[2] >= 0)); \
      ASSERT((p_stride[3] < (((int)1) << 18)) && (p_stride[3] >= 0)); \

#define CHECK_AR_ZERO_DST_STRIDE(dst_stride, dst_shape) \
       if (dst_stride[0] == 0) ASSERT(dst_shape[0] == 1); \
       if (dst_stride[1] == 0) ASSERT(dst_shape[1] == 1); \
       if (dst_stride[2] == 0) ASSERT(dst_shape[2] == 1); \
       if (dst_stride[3] == 0) ASSERT(dst_shape[3] == 1); \

// NOTICE:
// /*****************************************/
// Short_str 0/1 can not support broadcast b
// if b need to broadcast, make Short_str = 3
// and set correspond dimension stride 0
// /*****************************************/

// use for two_opd tensor_arithmetic
// include two_opd FP32/FP16/BFP16 AR
// can alse use for some two_opd fixed_point AR
void atomic_tensor_arithmetic_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str,
    int * Sign,
    int sym_saturate,
    PREC * Prec,
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "B_nstride=%d, B_cstride=%d, B_hstride=%d, B_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, B_is_const=%d, "
           "Sign_0=%d, Sign_1=%d, Sign_2=%d, A_short_str=%d, B_short_str=%d, R_short_str=%d, "
           "A_prec=%d, B_prec=%d, R_prec=%d, OP=%d, bd_cmd_id=%u, gdma_cmd_id=%u\n",
           __func__,
           A_addr, B_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[0],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[1],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[2],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, B_is_const,
           Sign[0], Sign[1], Sign[2], Short_str[0], Short_str[1], Short_str[2],
           Prec[0], Prec[1], Prec[2], op, pid_node->bd_cmd_id, pid_node->gdma_cmd_id);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 OPD1_N_STR = B_need_stride ? tensor_B_stride[0]&0x0000FFFF : 0;
    u32 OPD1_C_STR = B_need_stride ? tensor_B_stride[1]&0x0000FFFF : 0;
    u32 OPD1_H_STR = B_need_stride ? (((tensor_B_stride[0]&0x30000) << 2) | (tensor_B_stride[2]&0x0003FFFF)) : 0;
    u32 OPD1_W_STR = B_need_stride ? (((tensor_B_stride[1]&0x30000) << 2)| (tensor_B_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    if (!A_is_const) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[2]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
    ASSERT((op != AR_NOT) && (op != AR_COPY) && (op != AR_DATA_CONVERT));
    ASSERT((op != AR_SG) && (op != AR_SE) && (op != AR_SL) && (op != AR_ABS));
    ASSERT((op != AR_GET_FIRST_ZERO) && (op != AR_GET_FIRST_ONE));
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
    if ((op == AR_MIN || op == AR_MAX) && Prec[0] != FP8) {
       ASSERT(Sign[0] == Sign[1] && Prec[0] == Prec[1] && Prec[0] == Prec[2]);
    } else if ((op == AR_MIN || op == AR_MAX) && Prec[0] == FP8){
       ASSERT(Prec[0] == Prec[1] && Prec[0] == Prec[2]);
    }
    if (op == AR_ADD || op == AR_SUB || op == AR_MUL) {
       if (is_float_prec(Prec[0]) && Prec[0] != FP8 && Prec[1] != FP8) {
           ASSERT(Prec[0] == Prec[1] && Prec[0] == Prec[2]);
       } else if (Prec[0] == FP8 && Prec[1] == FP8){
           ASSERT(Prec[2] == FP8 || Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP8 && Prec[1] == FP16 && op != AR_SUB){
           ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP8 && Prec[1] == FP32 && op != AR_SUB){
           ASSERT(Prec[2] == FP32);
       } else if (Prec[0] == FP16 && Prec[1] == FP8 && op == AR_SUB){
           ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP32 && Prec[1] == FP8 && op == AR_SUB){
           ASSERT(Prec[2] == FP32);
       } else if (is_fixed_prec(Prec[0])){
           ASSERT(is_fixed_prec(Prec[1]) && is_fixed_prec(Prec[2]));
       }
    }
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
       CHECK_AR_STRIDE(tensor_R_stride);
       // int dst_shape[4] = {N, C, H, W};
       // CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif
    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[2],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   OPD1_H_STR, OPD1_W_STR, B_is_const ? 0 : Short_str[1],
                   A_is_const ? 0xffffffff : A_addr, B_is_const ? 0xffffffff : B_addr, R_addr,
                   Prec[0], Prec[1], Prec[2], op, 0, A_is_const, B_is_const, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)2 << 51)  |
          ((u64)Sign[2] << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Sign[0] << 5) |
            ((u64)Sign[1] << 6) |
            ((u64)Prec[2] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)Prec[1] << 14) |
            ((u64)A_is_const << 20) |
            ((u64)B_is_const << 21) |
            ((u64)Short_str[2] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)Short_str[1] << 29) |
            ((u64)sym_saturate << 61);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)OPD1_N_STR) |
          ((u64)OPD1_C_STR << 16);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    low[5] = ((u64)B_addr);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    high[6] = ((u64)OPD1_H_STR) |
            ((u64)OPD1_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          ((u64)B_is_const << 51) |
          ((u64)2 << 53) |
          ((u64)sym_saturate << 55) |
          ((u64)Sign[2] << 56) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[2]) |
            ((u64)Prec[0] << 3) |
            ((u64)Prec[1] << 6) |
            ((u64)Sign[0] << 12) |
            ((u64)Sign[1] << 13) |
            ((u64)Short_str[2] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)Short_str[1] << 20) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr) |
            ((u64)B_addr << 32);
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16) |
            ((u64)OPD1_N_STR << 32) |
            ((u64)OPD1_C_STR << 48);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = ((u64)OPD0_W_STR) |
            ((u64)OPD1_H_STR << 20) |
            ((u64)OPD1_W_STR << 40);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec[2], pid_node, high, low, elt);
}

void atomic_tensor_arithmetic_div_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str,
    PREC prec,
    int iter,
    int thread_id,
    CMD_ID_NODE * pid_node
) {
    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "B_nstride=%d, B_cstride=%d, B_hstride=%d, B_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, B_is_const=%d, "
           "A_short_str=%d, B_short_str=%d, R_short_str=%d, "
           "iter=%d\n",
           __func__,
           A_addr, B_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[0],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[1],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[2],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, B_is_const,
           Short_str[0], Short_str[1], Short_str[2],
           iter);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 OPD1_N_STR = B_need_stride ? tensor_B_stride[0]&0x0000FFFF : 0;
    u32 OPD1_C_STR = B_need_stride ? tensor_B_stride[1]&0x0000FFFF : 0;
    u32 OPD1_H_STR = B_need_stride ? (((tensor_B_stride[0]&0x30000) << 2) | (tensor_B_stride[2]&0x0003FFFF)) : 0;
    u32 OPD1_W_STR = B_need_stride ? (((tensor_B_stride[1]&0x30000) << 2)| (tensor_B_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    ASSERT(is_float_prec(prec));
    if (!A_is_const) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(prec) == 0);
    }
    if (!B_is_const) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(prec) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(prec) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT(prec != INT4);
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(iter >= 0 && iter <= 4);
#endif
    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[2],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   OPD1_H_STR, OPD1_W_STR, B_is_const ? 0 : Short_str[1],
                   A_is_const ? 0xffffffff : A_addr, B_is_const ? 0xffffffff : B_addr, R_addr,
                   prec, prec, prec, AR_DIV, iter, A_is_const, B_is_const, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) | ((u64)AR << 41) |
          ((u64)AR_DIV << 45) | ((u64)2 << 51) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec << 8) |
            ((u64)prec << 11) |
            ((u64)prec << 14) |
            ((u64)A_is_const << 20) |
            ((u64)B_is_const << 21) |
            ((u64)Short_str[2] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)Short_str[1] << 29);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)OPD1_N_STR) |
          ((u64)OPD1_C_STR << 16) |
          ((u64)(iter - 1) << 32);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    low[5] = ((u64)B_addr);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    high[6] = ((u64)OPD1_H_STR) |
            ((u64)OPD1_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)AR_DIV << 45) |
          ((u64)A_is_const << 50) |
          ((u64)B_is_const << 51) |
          ((u64)2 << 53) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)prec) |
            ((u64)prec << 3) |
            ((u64)prec << 6) |
            ((u64)Short_str[2] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)Short_str[1] << 20) |
            ((u64)(iter - 1) << 23) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr) |
            ((u64)B_addr << 32);
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16) |
            ((u64)OPD1_N_STR << 32) |
            ((u64)OPD1_C_STR << 48);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = ((u64)OPD0_W_STR) |
            ((u64)OPD1_H_STR << 20) |
            ((u64)OPD1_W_STR << 40);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      AR_DIV, prec, pid_node, high, low, elt);
}

// this function used for ternary tensor_arithmetic
void atomic_tensor_arithmetic_ternary_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int C_is_const,
    int * Short_str, // len = 3, opd0, opd1, res
    int * Sign, // len = 2, opd0, opd1
    int sym_saturate,
    PREC * Prec, // len = 4, opd0, opd1, opd2, res
    AR_OP op,
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node) {

    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, C_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "B_nstride=%d, B_cstride=%d, B_hstride=%d, B_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, B_is_const=%d, C_is_const=%d, "
           "Sign_0=%d, Sign_1=%d, Sign_2=%d, A_short_str=%d, B_short_str=%d, R_short_str=%d, "
           "A_prec=%d, B_prec=%d, C_prec=%d, R_prec=%d, "
           "OP=%d, Round_mode=%d\n",
           __func__,
           A_addr, B_addr, C_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[0],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[1],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[2],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, B_is_const, C_is_const,
           Sign[0], Sign[1], Sign[2], Short_str[0], Short_str[1], Short_str[2],
           Prec[0], Prec[1], Prec[2], Prec[3],
           op, round);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 OPD1_N_STR = B_need_stride ? tensor_B_stride[0]&0x0000FFFF : 0;
    u32 OPD1_C_STR = B_need_stride ? tensor_B_stride[1]&0x0000FFFF : 0;
    u32 OPD1_H_STR = B_need_stride ? (((tensor_B_stride[0]&0x30000) << 2) | (tensor_B_stride[2]&0x0003FFFF)) : 0;
    u32 OPD1_W_STR = B_need_stride ? (((tensor_B_stride[1]&0x30000) << 2)| (tensor_B_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (!C_is_const) {
      ASSERT(get_npu_index(C_addr) == get_npu_index(R_addr));
      ASSERT(C_addr % get_bytesize(Prec[2]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[3]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((C_is_const == 0) || (C_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[3] != INT4);
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(op == AR_MAC || op == AR_ADD_SATU ||
           op == AR_SUB_SATU || op == AR_MUL_SATU ||
           op == AR_ADD || op == AR_SUB ||
           op == AR_MUL);
    if (op == AR_MAC) {
       ASSERT((Prec[0] == INT8 && Prec[1] == INT8 && Prec[3] == INT16) ||
              (Prec[0] == INT4 && Prec[1] == INT4 && Prec[3] == INT8));
       ASSERT(Prec[2] == INT16 && C_is_const);
    } else if (Prec[0] == INT4 || Prec[1] == INT4) {
       ASSERT(Prec[2] == INT8);
    }
#endif
    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[2],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   OPD1_H_STR, OPD1_W_STR, B_is_const ? 0 : Short_str[1],
                   A_is_const ? 0xffffffff : A_addr, B_is_const ? 0xffffffff : B_addr, R_addr,
                   Prec[0], Prec[1], Prec[3], op, 0, A_is_const, B_is_const, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)3 << 51)  |
          ((u64)Sign[2] << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Sign[0] << 5) |
            ((u64)Sign[1] << 6) |
            ((u64)Prec[3] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)Prec[1] << 14) |
            ((u64)Prec[2] << 17) |
            ((u64)A_is_const << 20) |
            ((u64)B_is_const << 21) |
            ((u64)C_is_const << 22) |
            ((u64)Short_str[2] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)Short_str[1] << 29) |
            ((u64)sym_saturate << 61);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)OPD1_N_STR) |
          ((u64)OPD1_C_STR << 16) |
          ((u64)round << 32);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    low[5] = ((u64)B_addr) |
          ((u64)C_addr << 32);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32) ;
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    high[6] = ((u64)OPD1_H_STR) |
            ((u64)OPD1_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          ((u64)B_is_const << 51) |
          ((u64)C_is_const << 52) |
          ((u64)3 << 53) |
          ((u64)sym_saturate << 55) |
          ((u64)Sign[2] << 56) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[3]) |
            ((u64)Prec[0] << 3) |
            ((u64)Prec[1] << 6) |
            ((u64)Prec[2] << 9) |
            ((u64)Sign[0] << 12) |
            ((u64)Sign[1] << 13) |
            ((u64)Short_str[2] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)Short_str[1] << 20) |
            ((u64)round << 23) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr) |
            ((u64)B_addr << 32);
    low[2] = ((u64)C_addr) |
          ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16) |
            ((u64)OPD1_N_STR << 32) |
            ((u64)OPD1_C_STR << 48);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = ((u64)OPD0_W_STR) |
            ((u64)OPD1_H_STR << 20) |
            ((u64)OPD1_W_STR << 40);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec[3], pid_node, high, low, elt);
}

// use for SE/SG/SL
void atomic_tensor_arithmetic_select_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, // len = 2, opd0, opd1
    PREC * Prec, //len = 2, opd0/opd1, opd2/res
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node){

    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, C_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "B_nstride=%d, B_cstride=%d, B_hstride=%d, B_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, B_is_const=%d, "
           "Sign_0=%d, Sign_1=%d, Sign_2=%d, A_short_str=%d, B_short_str=%d, R_short_str=%d, "
           "A_prec=%d, R_prec=%d, "
           "OP=%d\n",
           __func__,
           A_addr, B_addr, C_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[0],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[1],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[2],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, B_is_const,
           Sign[0], Sign[1], Sign[2], Short_str[0], Short_str[1], Short_str[2],
           Prec[0], Prec[1],
           op);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 OPD1_N_STR = B_need_stride ? tensor_B_stride[0]&0x0000FFFF : 0;
    u32 OPD1_C_STR = B_need_stride ? tensor_B_stride[1]&0x0000FFFF : 0;
    u32 OPD1_H_STR = B_need_stride ? (((tensor_B_stride[0]&0x30000) << 2) | (tensor_B_stride[2]&0x0003FFFF)) : 0;
    u32 OPD1_W_STR = B_need_stride ? (((tensor_B_stride[1]&0x30000) << 2)| (tensor_B_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

    //TENSOR_ARITHMETIC_GET_CYCLE(N, C, H, W, tensor_R_addr, op, pid_node);
#ifdef USING_CMODEL
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    ASSERT((op == AR_SG) || (op == AR_SE) || (op == AR_SL));
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
//     ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0]));
#endif

    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[2],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   OPD1_H_STR, OPD1_W_STR, B_is_const ? 0 : Short_str[1],
                   A_is_const ? 0xffffffff : A_addr, B_is_const ? 0xffffffff : B_addr, R_addr,
                   Prec[0], Prec[0], Prec[1], op, 0, A_is_const, B_is_const, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)3 << 51)  |
          ((u64)Sign[2] << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Sign[0] << 5) |
            ((u64)Sign[1] << 6) |
            ((u64)Prec[1] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)Prec[0] << 14) |
            ((u64)Prec[1] << 17) |
            ((u64)A_is_const << 20) |
            ((u64)B_is_const << 21) |
            (1ull << 22) |
            ((u64)Short_str[2] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)Short_str[1] << 29);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)OPD1_N_STR) |
          ((u64)OPD1_C_STR << 16);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    low[5] = ((u64)B_addr) |
          ((u64)C_addr << 32);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    high[6] = ((u64)OPD1_H_STR) |
            ((u64)OPD1_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          ((u64)B_is_const << 51) |
          (1ull << 52) |
          ((u64)3 << 53) |
          ((u64)Sign[2] << 56) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[1]) |
            ((u64)Prec[0] << 3) |
            ((u64)Prec[0] << 6) |
            ((u64)Prec[1] << 9) |
            ((u64)Sign[0] << 12) |
            ((u64)Sign[1] << 13) |
            ((u64)Short_str[2] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)Short_str[1] << 20) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr) |
            ((u64)B_addr << 32);
    low[2] = ((u64)C_addr) |
          ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16) |
            ((u64)OPD1_N_STR << 32) |
            ((u64)OPD1_C_STR << 48);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = ((u64)OPD0_W_STR) |
            ((u64)OPD1_H_STR << 20) |
            ((u64)OPD1_W_STR << 40);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec[1], pid_node, high, low, elt);
}

// use for two_opds with round ops
void atomic_tensor_arithmetic_with_round_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign,
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op,
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node){

    FW_DBG("%s: "
           "A_addr = 0x%08x, B_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "B_nstride=%d, B_cstride=%d, B_hstride=%d, B_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, B_is_const=%d, "
           "Sign_0=%d, Sign_1=%d, A_short_str=%d, B_short_str=%d, R_short_str=%d, "
           "A_prec=%d, B_prec=%d, R_prec=%d, "
           "OP=%d, Round_mode=%d\n",
           __func__,
           A_addr, B_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[0],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[1],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[2],
           tensor_B_stride == NULL ? 0 : tensor_B_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, B_is_const,
           Sign[0], Sign[1], Short_str[0], Short_str[1], Short_str[2],
           Prec[0], Prec[1], Prec[2],
           op, round);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 OPD1_N_STR = B_need_stride ? tensor_B_stride[0]&0x0000FFFF : 0;
    u32 OPD1_C_STR = B_need_stride ? tensor_B_stride[1]&0x0000FFFF : 0;
    u32 OPD1_H_STR = B_need_stride ? (((tensor_B_stride[0]&0x30000) << 2) | (tensor_B_stride[2]&0x0003FFFF)) : 0;
    u32 OPD1_W_STR = B_need_stride ? (((tensor_B_stride[1]&0x30000) << 2) | (tensor_B_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[2]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(op == AR_LOGIC_SHIFT || op == AR_ARITH_SHIFT || op == AR_ROTATE_SHIFT);
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    ASSERT(Sign[1] == 1);
    ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0])
           && is_fixed_prec(Prec[0]) && is_fixed_prec(Prec[1]));
    if (op == AR_ROTATE_SHIFT) {
       ASSERT(Prec[0] == Prec[2]);
    }
#endif
    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[2],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   OPD1_H_STR, OPD1_W_STR, B_is_const ? 0 : Short_str[1],
                   A_is_const ? 0xffffffff : A_addr, B_is_const ? 0xffffffff : B_addr, R_addr,
                   Prec[0], Prec[1], Prec[2], op, 0, A_is_const, B_is_const, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)2 << 51) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Sign[0] << 5) |
            ((u64)Sign[1] << 6) |
            ((u64)Prec[2] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)Prec[1] << 14) |
            ((u64)A_is_const << 20) |
            ((u64)B_is_const << 21) |
            ((u64)Short_str[2] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)Short_str[1] << 29);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)OPD1_N_STR) |
          ((u64)OPD1_C_STR << 16) |
          ((u64)round << 32);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    low[5] = ((u64)B_addr);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    high[6] = ((u64)OPD1_H_STR) |
            ((u64)OPD1_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          ((u64)B_is_const << 51) |
          ((u64)2 << 53) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[2]) |
            ((u64)Prec[0] << 3) |
            ((u64)Prec[1] << 6) |
            ((u64)Sign[0] << 12) |
            ((u64)Sign[1] << 13) |
            ((u64)Short_str[2] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)Short_str[1] << 20) |
            ((u64)round << 23) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr) |
            ((u64)B_addr << 32);
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16) |
            ((u64)OPD1_N_STR << 32) |
            ((u64)OPD1_C_STR << 48);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = ((u64)OPD0_W_STR) |
            ((u64)OPD1_H_STR << 20) |
            ((u64)OPD1_W_STR << 40);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec[2], pid_node, high, low, elt);
}

// use for dtype convert
void atomic_tensor_arithmetic_dtype_convert_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str, // num = 2, opd0, res0
    int * Sign, // num = 2, opd0, res0
    int sym_saturate,
    PREC * Prec, // num = 2, opd0, res0
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node){

    FW_DBG("%s: "
           "A_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, "
           "Sign_A=%d, Sign_R=%d, A_short_str=%d, R_short_str=%d, "
           "A_prec=%d, R_prec=%d, "
           "Round_mode=%d\n",
           __func__,
           A_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const,
           Sign[0], Sign[1], Short_str[0], Short_str[1],
           Prec[0], Prec[1],
           round);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

    //TENSOR_ARITHMETIC_GET_CYCLE(N, C, H, W, tensor_R_addr, op, pid_node);
#ifdef USING_CMODEL
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    if(A_addr == R_addr) ASSERT(get_bytesize(Prec[0]) >= get_bytesize(Prec[1]));
#endif

    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[1],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   0, 0, 0,
                   A_is_const ? 0xffffffff : A_addr, 0, R_addr,
                   Prec[0], Prec[0], Prec[1], AR_DATA_CONVERT, 0, A_is_const, 0, pid_node);

    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)AR_DATA_CONVERT << 45) |
          (1ull << 51) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Sign[0] << 5) |
            ((u64)Sign[1] << 7) |
            ((u64)Prec[1] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)A_is_const << 20) |
            ((u64)Short_str[1] << 23) |
            ((u64)Short_str[0] << 26) |
            ((u64)sym_saturate << 61);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    low[4] = ((u64)round << 32);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)AR_DATA_CONVERT << 45) |
          ((u64)A_is_const << 50) |
          (1ull << 53) |
          ((u64)sym_saturate << 55) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[1]) |
            ((u64)Prec[0] << 3) |
            ((u64)Sign[0] << 12) |
            ((u64)Short_str[1] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)round << 23) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = ((u64)A_addr);
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40) |
          ((u64)Sign[1] << 60);
    high[3] = (u64)OPD0_W_STR;
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      AR_DATA_CONVERT, Prec[1], pid_node, high, low, elt);
}

// copy/copy_mb/abs/not
//void atomic_tensor_arithmetic_copy_like_gen_cmd(
void atomic_tensor_arithmetic_single_opd_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,// num = 2, opd0, res0
    int sign, // for abs and FP8
    PREC Prec,
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node){

    FW_DBG("%s: "
           "A_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, A_short_str=%d, R_short_str=%d, "
           "sign = %d, prec=%d, OP=%d\n",
           __func__,
           A_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const, Short_str[0], Short_str[1],
           sign, Prec, op);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % (get_eu_num(Prec) * get_bytesize(Prec)) == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ceiling_func(get_eu_num(Prec) * get_bit_width(Prec), 8) == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((op == AR_NOT) || (op == AR_COPY) || (op == AR_ABS));
    ASSERT(Prec != INT4);
    if (op == AR_NOT) {
      ASSERT(!is_float_prec(Prec) && Prec != INT4);
    }
    ASSERT(Prec != INT4 || op == AR_ABS);
    if (A_need_stride) {
      if (Prec == INT4) {
        ASSERT(op == AR_COPY);
        ASSERT(tensor_A_stride[3] == 1 && (tensor_A_stride[2] & 0x1) == 0 &&
               (tensor_A_stride[1] & 0x1) == 0 && (tensor_A_stride[0] & 0x1) == 0);
      }

      CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
      if (Prec == INT4) {
        ASSERT(op == AR_COPY);
        ASSERT(tensor_R_stride[3] == 1 && (tensor_R_stride[2] & 0x1) == 0 &&
               (tensor_R_stride[1] & 0x1) == 0 && (tensor_R_stride[0] & 0x1) == 0);
      }
      CHECK_AR_STRIDE(tensor_R_stride);
//       int dst_shape[4] = {N, C, H, W};
//       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif

    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[1],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   0, 0, 0,
                   A_is_const ? 0xffffffff : A_addr, 0, R_addr,
                   Prec, Prec, Prec, op, 0, A_is_const, 0, pid_node);

    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          (1ull << 51) |
          ((u64)bd_power_step() << 59);
    high[0] = (u64)sign << 5 |
            ((u64)Prec << 8) |
            ((u64)Prec << 11) |
            ((u64)A_is_const << 20) |
            ((u64)Short_str[1] << 23) |
            ((u64)Short_str[0] << 26);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8;  ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          (1ull << 53) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec) |
            ((u64)Prec << 3) |
            ((u64)sign << 12) |
            ((u64)Short_str[1] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = (u64)A_addr;
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = (u64)OPD0_W_STR;
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec, pid_node, high, low, elt);
}

// use for get_first_zero/get_first_one
void atomic_tensor_arithmetic_get_first_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str, // num = 2, opd0, res0
    PREC * Prec, // A && R can have different dtype
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node){

    FW_DBG("%s: "
           "A_addr = 0x%08x, R_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "A_nstride=%d, A_cstride=%d, A_hstride=%d, A_wstride=%d, "
           "R_nstride=%d, R_cstride=%d, R_hstride=%d, R_wstride=%d, "
           "A_is_const=%d, "
           "A_short_str=%d, R_short_str=%d, "
           "A_prec=%d, R_prec=%d, "
           "OP=%d\n",
           __func__,
           A_addr, R_addr,
           N, C, H, W,
           tensor_A_stride == NULL ? 0 : tensor_A_stride[0],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[1],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[2],
           tensor_A_stride == NULL ? 0 : tensor_A_stride[3],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[0],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[1],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[2],
           tensor_R_stride == NULL ? 0 : tensor_R_stride[3],
           A_is_const,
           Short_str[0], Short_str[1],
           Prec[0], Prec[1],
           op);

    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    u32 OPD0_N_STR = A_need_stride ? tensor_A_stride[0]&0x0000FFFF : 0;
    u32 OPD0_C_STR = A_need_stride ? tensor_A_stride[1]&0x0000FFFF : 0;
    u32 OPD0_H_STR = A_need_stride ? (((tensor_A_stride[0]&0x30000) << 2) | (tensor_A_stride[2]&0x0003FFFF)) : 0;
    u32 OPD0_W_STR = A_need_stride ? (((tensor_A_stride[1]&0x30000) << 2) | (tensor_A_stride[3]&0x0003FFFF)) : 0;
    u32 RES0_N_STR = R_need_stride ? tensor_R_stride[0]&0x0000FFFF : 0;
    u32 RES0_C_STR = R_need_stride ? tensor_R_stride[1]&0x0000FFFF : 0;
    u32 RES0_H_STR = R_need_stride ? (((tensor_R_stride[0]&0x30000) << 2) | (tensor_R_stride[2]&0x0003FFFF)) : 0;
    u32 RES0_W_STR = R_need_stride ? (((tensor_R_stride[1]&0x30000) << 2) | (tensor_R_stride[3]&0x0003FFFF)) : 0;

#ifdef USING_CMODEL
    if (!A_is_const) {
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((op == AR_GET_FIRST_ZERO) || (op == AR_GET_FIRST_ONE));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
    ASSERT(!is_float_prec(Prec[0]) && !is_float_prec(Prec[1]) && Prec[0] != INT4 && Prec[1] != INT4);
    ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0]));
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
       CHECK_AR_STRIDE(tensor_R_stride);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif
    AR_GET_PROFILE(N, C, H, W, RES0_H_STR, RES0_W_STR, Short_str[1],
                   OPD0_H_STR, OPD0_W_STR, A_is_const ? 0 : Short_str[0],
                   0, 0, 0,
                   A_is_const ? 0xffffffff : A_addr, 0, R_addr,
                   Prec[0], Prec[0], Prec[1], op, 0, A_is_const, 0, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          (1ull << 51) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[1] << 8) |
            ((u64)Prec[0] << 11) |
            ((u64)A_is_const << 20) |
            ((u64)Short_str[1] << 23) |
            ((u64)Short_str[0] << 26);
    high[1] = bd_get_lane_mask();
    low[2] = ((u64)N) |
          ((u64)C << 16) |
          ((u64)H << 32) |
          ((u64)W << 48);
    high[3] = ((u64)RES0_N_STR) |
            ((u64)RES0_C_STR << 16) |
            ((u64)OPD0_N_STR << 32) |
            ((u64)OPD0_C_STR << 48);
    high[4] = ((u64)R_addr) |
            ((u64)A_addr << 32);
    high[5] = ((u64)RES0_H_STR) |
            ((u64)RES0_W_STR << 32);
    low[6] = ((u64)OPD0_H_STR) |
          ((u64)OPD0_W_STR << 32);
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 8; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 4;
    u64 low[4] = {0}, high[4] = {0};
    low[0] = 1ull |
          (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
          ((u64)1ull << 37) |
          ((u64)AR << 41) |
          ((u64)op << 45) |
          ((u64)A_is_const << 50) |
          (1ull << 53) |
          ((u64)bd_power_step() << 59);
    high[0] = ((u64)Prec[1]) |
            ((u64)Prec[0] << 3) |
            ((u64)Short_str[1] << 14) |
            ((u64)Short_str[0] << 17) |
            ((u64)N << 32) |
            ((u64)C << 48);
    low[1] = ((u64)H) |
          ((u64)W << 16) |
          ((u64)R_addr << 32);
    high[1] = (u64)A_addr;
    low[2] = ((u64)RES0_N_STR << 32) |
          ((u64)RES0_C_STR << 48);
    high[2] = ((u64)OPD0_N_STR) |
            ((u64)OPD0_C_STR << 16);
    low[3] = ((u64)RES0_H_STR) |
          ((u64)RES0_W_STR << 20) |
          ((u64)OPD0_H_STR << 40);
    high[3] = (u64)OPD0_W_STR;
    BEGIN_FAST_GEN_CMD_BD(thread_id)
    for (int i = 0; i < 4; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, AR,
      op, Prec[1], pid_node, high, low, elt);
}

