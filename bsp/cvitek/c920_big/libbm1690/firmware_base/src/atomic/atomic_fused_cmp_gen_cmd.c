#include "firmware_common.h"
#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"


void atomic_fused_cmp_gen_cmd(
    u32 tensorA_addr,
    u32 tensorB_addr,
    u32 tensorC_addr,
    u32 tensorD_addr,
    u32 tensorR0_addr,
    u32 tensorR1_addr,
    int N,
    int C,
    int H,
    int W,
    int A_is_constant,
    int B_is_constant,
    int C_is_constant,
    int D_is_constant,
    int sign,
    int side,
    int bin_w,
    int A_short_str, //normal(0:align, 3:tensor)
    int B_short_str, //normal(0:align, 3:tensor)
    PREC AB_dtype,
    PREC CD_dtype,
    PREC RES0_dtype,  // only used for srch bin
    CMP_OP op,
    int thread_id,
    CMD_ID_NODE* pid_node) {
    FW_DBG("%s: "
           "tensorA_addr = 0x%08x, tensorB_addr = 0x%08x, tensorC_addr = 0x%08x, tensorD_addr = 0x%08x,"
           "tensorR0_addr = 0x%08x, tensorR1_addr = 0x%08x, "
           "N = %d, C = %d, H = %d, W = %d, "
           "A_is_constant = %d, B_is_constant = %d, C_is_constant = %d, D_is_constant = %d, "
           "sign = %d, side = %d, bin_w = %d, A_short_str = %d, B_short_str = %d, "
           "AB_type = %d, CD_type = %d, RES0_dtype = %d, op = %d\n",
           __func__,
           tensorA_addr, tensorB_addr, tensorC_addr, tensorD_addr, tensorR0_addr, tensorR1_addr,
           N, C, H, W,
           A_is_constant, B_is_constant, C_is_constant, D_is_constant,
           sign, side, bin_w, A_short_str, B_short_str,
           AB_dtype, CD_dtype, RES0_dtype, op);
#ifdef USING_CMODEL
    int AB_type_size = get_bytesize(AB_dtype);
    int A_align_num = (A_short_str == 0 ? (int)ALIGN_BYTES : AB_type_size);
    int B_align_num = (B_short_str == 0 ? (int)ALIGN_BYTES : AB_type_size);
    ASSERT((N) < (((int)1) << 16) && ((N) > 0));
    ASSERT((C) < (((int)1) << 16) && ((C) > 0));
    ASSERT((H) < (((int)1) << 16) && ((H) > 0));
    ASSERT((W) < (((int)1) << 16) && ((W) > 0));
    ASSERT((A_is_constant == 0) || (A_is_constant == 1));
    ASSERT((B_is_constant == 0) || (B_is_constant == 1));
    ASSERT((C_is_constant == 0) || (C_is_constant == 1));
    ASSERT((D_is_constant == 0) || (D_is_constant == 1));
    ASSERT((sign) == 0 || (sign) == 1);
    ASSERT((side) == 0 || (side) == 1);
    ASSERT(((op) <= 27) && ((op) >= 22));
    ASSERT((A_short_str) == 0 || (A_short_str) == 3);
    ASSERT((B_short_str) == 0 || (B_short_str) == 3);
    ASSERT((((int)AB_dtype) <= 7) && (((int)AB_dtype) >= 0));
    ASSERT(AB_dtype != INT4 && CD_dtype != INT4);
    ASSERT((((int)CD_dtype) <= 7) && (((int)CD_dtype) >= 0));
    if (op == CMP_SRCH_BIN) {
        ASSERT(RES0_dtype == INT8 || RES0_dtype == INT16 || RES0_dtype == INT32);
        ASSERT(B_is_constant == 0);
    }
    // check addr align according to dtype
    if (A_is_constant == 0) ASSERT(tensorA_addr % A_align_num == 0);
    if (B_is_constant == 0) ASSERT(tensorB_addr % B_align_num == 0);
    if (C_is_constant == 0) ASSERT(tensorC_addr % ALIGN_BYTES == 0);
    if (D_is_constant == 0) ASSERT(tensorD_addr % ALIGN_BYTES == 0);
    ASSERT(tensorR0_addr % ALIGN_BYTES == 0);
    if (op == CMP_GT_AND_SG || op == CMP_LT_AND_SL) {
        ASSERT(tensorR1_addr % ALIGN_BYTES == 0);
    }
#endif
    FUSED_CMP_GET_PROFILE(N, C, H, W, CD_dtype, (tensorR0_addr > tensorR1_addr) ? tensorR0_addr : tensorR1_addr, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)CMP << 41) |
              ((u64)op << 45) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)sign << 5) |
               ((u64)side << 7) |
               ((u64)RES0_dtype << 8) |
               ((u64)AB_dtype << 11) |
               ((u64)CD_dtype << 17) |
               ((u64)A_is_constant << 20) |
               ((u64)B_is_constant << 21) |
               ((u64)C_is_constant << 22) |
               ((u64)A_short_str << 26) |
               ((u64)B_short_str << 29) |
               ((u64)D_is_constant << 62);
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)N << 0) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        low[3] = ((u64)bin_w << 48);
        high[4] = ((u64)tensorR0_addr << 0) |
               ((u64)tensorA_addr << 32);
        low[5] = ((u64)tensorB_addr << 0) |
              ((u64)tensorC_addr << 32);
        high[7] = ((u64)tensorR1_addr << 0) |
               ((u64)tensorD_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 3;
    u64 low[3] = {0}, high[3] = {0};

    u64 opd2_addr, opd2_prec;
    if (op == CMP_SRCH_BIN) {
        opd2_addr = bin_w;
        opd2_prec = RES0_dtype;
    } else {
        opd2_addr = tensorC_addr;
        opd2_prec = CD_dtype;
    }
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = ((u64)1ull << 0) |
              (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
              ((u64)1ull << 37) |
              ((u64)CMP << 41) |
              ((u64)op << 45) |
              ((u64)A_is_constant << 50) |
              ((u64)B_is_constant << 51) |
              ((u64)C_is_constant << 52) |
              ((u64)D_is_constant << 53) |
              ((u64)bd_power_step() << 59);
        high[0] = ((u64)AB_dtype << 0) |
               ((u64)opd2_prec << 3) |
               ((u64)A_short_str << 6) |
               ((u64)B_short_str << 9) |
               ((u64)side << 62) |
               ((u64)sign << 63);
        low[1] = ((u64)N << 0) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        high[1] = ((u64)tensorR0_addr << 0) |
               ((u64)tensorR1_addr << 32);
        low[2] = ((u64)tensorA_addr << 0) |
              ((u64)tensorB_addr << 32);
        high[2] = ((u64)opd2_addr << 0) |
               ((u64)tensorD_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
    profile_time_set_node(ENGINE_BD, CMP,
      op, RES0_dtype, pid_node, high, low, elt);
#endif
}


