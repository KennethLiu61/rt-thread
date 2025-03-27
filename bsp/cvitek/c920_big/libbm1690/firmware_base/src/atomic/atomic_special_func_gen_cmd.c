#include "firmware_common.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"
#include "atomic_gen_cmd.h"

void atomic_sfu_gen_cmd(
    u32     A_addr,
    u32     Y_addr,
    int     input_n,
    int     input_c,
    int     input_h,
    int     input_w,
    int     n,
    SFU_OP  sfu_op,
    u32     table_start_addr,
    PREC    res0_prec,
    PREC    opd0_prec,
    int    thread_id,
    CMD_ID_NODE * pid_node
    ) {
    FW_DBG("%s: "
           "A_addr = 0x%08x, Y_addr = 0x%08x, "
           "N=%d, C=%d, H=%d, W=%d, "
           "taylor_n=%d, sfu_op=%d, "
           "table_start_addr = 0x%08x,"
           "res0_prec=%d,opd0_prec=%d\n",
           __func__, A_addr, Y_addr,
           input_n, input_c, input_h, input_w,
           n, sfu_op, table_start_addr,
           res0_prec,opd0_prec);
#ifdef USING_CMODEL
    ASSERT(input_n != 0 && input_c != 0 && input_h != 0 && input_w != 0);
    u32 A_npu_idx = get_npu_index(A_addr);
    u32 Y_npu_idx = get_npu_index(Y_addr);
    u32 table_npu_idx = get_npu_index(table_start_addr);
    ASSERT(A_npu_idx == Y_npu_idx);
    switch (sfu_op) {
        case SFU_TAYLOR_4X:
        case SFU_TAYLOR:
            ASSERT(((res0_prec == FP16) && (opd0_prec == FP16)) ||
                      ((res0_prec == BFP16) && (opd0_prec == BFP16)) ||
                      ((res0_prec == FP32) && (opd0_prec == FP32)));
            ASSERT(table_npu_idx == Y_npu_idx);
            ASSERT(table_start_addr % ALIGN_BYTES == 0);
            ASSERT(n >= 2);
            break;
        case SFU_NORM:
            if(opd0_prec == FP16)
                ASSERT(res0_prec == INT16 || res0_prec == FP16);
            else if(opd0_prec == BFP16)
                ASSERT(res0_prec == INT16 || res0_prec == BFP16);
            else if(opd0_prec == FP32)
                ASSERT(res0_prec == INT32 || res0_prec == FP32);
            else  ASSERT(0);
            break;
        case SFU_RSQ:
            ASSERT(1 <= n && n <= 4);
            ASSERT(is_float_prec(opd0_prec) && opd0_prec == res0_prec);
            break;
        default : ASSERT(0);
    }

#endif
    SFU_GET_PROFILE(input_n, input_c, input_h, input_w, sfu_op, res0_prec, n, Y_addr, pid_node);
    const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
    int elt = 8;
    u64 low[8] = {0}, high[8] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SFU << 41) |
            ((u64)sfu_op << 45) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)res0_prec << 8) |
            ((u64)opd0_prec << 11);
        low[1] = 0x0000001100000000ull;
        high[1] = bd_get_lane_mask();
        low[2] = ((u64)input_n) |
              ((u64)input_c << 16) |
              ((u64)input_h << 32) |
              ((u64)input_w << 48);
        high[2] = 0x0001000100010001ull;
        low[3] = ((u64)n)|
               0x0001000100010000ull;
        high[3] = 0x0001000100010001ull;
        low[4] = 0x0001000000010001ull |
             ((u64)((n-1)&0xffff) << 32);
        high[4] = ((u64)Y_addr)|
               ((u64)A_addr << 32);
        low[5] = (u64)table_start_addr;
        high[5] = 0x0000000100000001ull;
        low[6] = 0x0000000100000001ull;
        high[6] = 0x0000000100000001ull;
        low[7] = 0x0000000100000001ull;
        high[7] = 0x0ull;
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#else
    int elt = 2;
    u64 low[2] = {0}, high[2] = {0};
    BEGIN_FAST_GEN_CMD_BD(thread_id)
        low[0] = 0x1ull |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SFU << 41) |
            ((u64)sfu_op << 45) |
            ((u64)bd_power_step() << 59);
        high[0] = ((u64)res0_prec) |
            ((u64)opd0_prec << 3) |
            ((u64)((n-1)&0x3) << 6) |
            ((u64)input_n << 16) |
            ((u64)input_c << 32) |
            ((u64)input_h << 48);
        low[1] = ((u64)input_w) |
            ((u64)n << 16) |
            ((u64)Y_addr << 32);
        high[1] = ((u64)A_addr) |
            ((u64)table_start_addr << 32);
        for (int i = 0; i < elt; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_BD(pid_node)
#endif
    profile_time_set_node(ENGINE_BD, SFU,
      sfu_op, res0_prec, pid_node, high, low, elt);
}
