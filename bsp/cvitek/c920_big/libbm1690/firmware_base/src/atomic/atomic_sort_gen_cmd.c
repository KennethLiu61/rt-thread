#include "firmware_common.h"
#include "sort_reg_def.h"
#include "gen_cmd.h"


void atomic_sort_gen_cmd(
        u64 src_data_addr,
        u64 src_idx_addr,
        u64 dst_data_addr,
        u64 dst_idx_addr,
        int data_type,   // 0:fp32 1:int32 2:uint32
        int row_num,     // support 2d sort, {row_num, len}, and sort at the last axis
        int len,
        int is_descend,
        int idx_enable,
        int idx_auto,
        int topk,
        CMD_ID_NODE *pid_node) {
    FW_DBG("%s: "
           "src_data_addr = 0x%llx, src_idx_addr = 0x%llx, "
           "dst_data_addr = 0x%llx, dst_idx_addr = 0x%llx, "
           "data_type = %d, row_num:%d, len = %d, is_descend = %d "
           "idx_enable = %d, idx_auto = %d, topk = %d\n",
           __func__,
           src_data_addr, src_idx_addr,
           dst_data_addr, dst_idx_addr,
           data_type, row_num, len, is_descend,
           idx_enable, idx_auto, topk);
#ifdef USING_CMODEL

    ASSERT(src_data_addr % sizeof(float) == 0);
    ASSERT(src_idx_addr % sizeof(float) == 0);
    ASSERT(dst_data_addr % sizeof(float) == 0);
    ASSERT(dst_idx_addr % sizeof(float) == 0);
    ASSERT(data_type < 3 && data_type >= 0);
    ASSERT(row_num > 0 && len > 0);
    ASSERT(is_descend < 2 && is_descend >= 0);
    ASSERT(idx_enable < 2 && idx_enable >= 0);
    ASSERT(idx_auto < 2 && idx_auto >= 0);
    ASSERT(topk <= len && topk > 0);
#endif

    const volatile u64 reg_addr = HAU_CMD_BASE_ADDR;
    u64 high, low;
    BEGIN_FAST_GEN_CMD(HAU)
        low = (1ull << 16) |
              (((src_data_addr >> MAX_GMEM_BIT) & TAG_MASK) << 32) |
              (((dst_data_addr >> MAX_GMEM_BIT) & TAG_MASK) << 40) |
              (((src_idx_addr >> MAX_GMEM_BIT) & TAG_MASK) << 48) |
              (((dst_idx_addr >> MAX_GMEM_BIT) & TAG_MASK) << 56);
        high = (u64)row_num << 32;
        WRITE_CMD_EX(reg_addr, 0, high, low);
        WRITE_CMD_EX(reg_addr, 1, 0ull, 0ull);
        WRITE_CMD_EX(reg_addr, 2, 0ull, 0ull);
        low = (u64)data_type |
            ((row_num == 1 ? 0ull : 1ull) << 3) |
            (1ull << 5) |
            ((u64)is_descend << 7) |
            ((u64)idx_enable << 8) |
            ((u64)idx_auto << 9) |
            (1ull << 23) |
            ((src_data_addr & 0xffffffff) << 32);
        high = ((src_data_addr >> 32) & 0xff) |
            ((dst_data_addr >> 24) & 0xff00) |
            ((src_idx_addr >> 16) & 0xff0000) |
            ((dst_idx_addr >> 8) & 0xff000000) |
            ((dst_data_addr & 0xffffffff) << 32);
        WRITE_CMD_EX(reg_addr, 3, high, low);
        low = (u64)topk |
            ((u64)len << 32);
        high = (src_idx_addr & 0xffffffff) |
            ((dst_idx_addr & 0xffffffff) << 32);
        WRITE_CMD_EX(reg_addr, 4, high, low);
    END_FAST_GEN_CMD(HAU, pid_node)
}
