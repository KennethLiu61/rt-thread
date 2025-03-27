#include "atomic_dma_utils.h"
#include "atomic_sdma_gen_cmd.h"
#include "sdma_reg_def.h"
#include "sdma_reg_value.h"
#include "gen_cmd.h"

#define ASSERT_SDMA_TENSOR_NSIZE(n) \
    ASSERT_FS_INFO(n>0 && n<=SDMA_MAX_N, #n "=%d", n)

#define ASSERT_SDMA_TENSOR_CSIZE(c) \
    ASSERT_FS_INFO(c>0 && c<=SDMA_MAX_C, #c "=%d", c)

#define ASSERT_SDMA_TENSOR_HSIZE(h) \
    ASSERT_FS_INFO(h>0 && h<=SDMA_MAX_H, #h "=%d", h)

#define ASSERT_SDMA_TENSOR_WSIZE(w) \
    ASSERT_FS_INFO(w>0 && w<=SDMA_MAX_W, #w "=%d", w)

#define ASSERT_SDMA_TENSOR_SIZE(n,c,h,w) \
    ASSERT_SDMA_TENSOR_NSIZE(n); \
    ASSERT_SDMA_TENSOR_CSIZE(c); \
    ASSERT_SDMA_TENSOR_HSIZE(h); \
    ASSERT_SDMA_TENSOR_WSIZE(w)

#define ASSERT_SDMA_WSTRIDE(wstr, byte_len) \
    ASSERT_FS_INFO((wstr * byte_len <= SDMA_MAX_WSTRIDE_BYTE_LEN) && (wstr != 0), "W stride byte len = %d", wstr * byte_len)

#define ASSERT_SDMA_WSTRIDE_FP20(wstr) \
    ASSERT_FS_INFO(wstr == 1, "When data type is fp20, W stride should be 1")

#define ASSERT_SDMA_COMPACT_FP20(n,c,h,w,nstr,cstr,hstr,wstr) \
    ASSERT_SDMA_WSTRIDE_FP20(wstr); \
    ASSERT_FS_INFO(hstr == (w), "When data type is fp20, 51 elements constitute fp20 block"); \
    ASSERT_FS_INFO(cstr == (h * hstr), "When data type is fp20, c stride should be compacted"); \
    ASSERT_FS_INFO(nstr == (c * cstr), "When data type is fp20, n stride should be compacted");

inline static u64 sdma_get_lane_mask() {
    u64 lane_mask = 0xffffffffffffffff;
#if defined(USING_CMODEL) && defined(SG_TV_GEN)
    char *en_lane_mask = getenv("TV_GEN_EN_LANE_MASK");
    char *p = getenv("TV_GEN_LOG_PATH");
    char path_int[1024 * 2] = {'\0'};
    if (p) {
        strcpy(path_int, p);
    } else {
        strcpy(path_int, "");
    }
    strcat(path_int, "lane_mask_param");
    if (access(path_int, F_OK) == 0) {
        FILE *file = fopen(path_int, "r");
        fscanf(file, "%llx\n", &lane_mask);
        fclose(file);
    } else if (en_lane_mask && atoi(en_lane_mask) == 1) {
        lane_mask = 0;
        for (int i = 0; i < 64; i++) {
            lane_mask |= (rand() % 3 ? 1ull : 0ull) << i;
        }
        if (lane_mask == 0)
            lane_mask = 1ull << (rand() % NPU_NUM);

        FILE *file = fopen(path_int, "w");
        fprintf(file, "%llx\n", lane_mask);
        fclose(file);
    }
#endif
    return lane_mask;
}

void sdma_tensor_general_move_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,

    u64 dst_addr,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose,  // N/C transpose
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_N=%d, src_C=%d, src_H=%d, src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx, dst_N=%d, dst_C=%d, dst_H=%d, dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d\n",
        __func__, src_addr, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    int type_len = get_sdma_format_type_len(src_format);
    ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
    ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    if (src_format == SDMA_FP20) {
        ASSERT(transpose == 0);
        ASSERT(src_addr % ALIGN_BYTES == 0);
        ASSERT(dst_addr % ALIGN_BYTES == 0);
        ASSERT_SDMA_WSTRIDE_FP20(src_W_stride);
        ASSERT_SDMA_WSTRIDE_FP20(dst_W_stride);
    } else {
        // int type_len = get_sdma_format_type_len(src_format);
        // ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
        // ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    int special_func = transpose ? SDMA_FUNC_TRANS : SDMA_FUNC_NONE;
    if(transpose) {
        src_W_stride = 1;
        dst_W_stride = 1;
    }

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    SDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, special_func, 3,
                            src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride,
                            dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            ((u64)SDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
    high[1] = ((u64)src_H_stride) | ((u64)src_W_stride << 32);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
    low[3] = ((u64)src_N) |
            ((u64)src_C << 16) | ((u64)src_H << 32) |
            ((u64)src_W << 48);
    high[3] = ((u64)dst_N) |
            ((u64)dst_C << 16) |
            ((u64)dst_H << 32) |
            ((u64)dst_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = sdma_get_lane_mask();

    BEGIN_FAST_GEN_CMD_SDMA(port_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_TENSOR,
        special_func, src_format | SDMA_S2S << 4, pid_node, high, low, 6);
}

static inline int get_constant_value(const void * p_val, int format) {
    int constant = 0;
    int type_len = get_sdma_format_type_len(format);
    if (format == SDMA_FP20) {
        type_len = 4;
    }
    memcpy(&constant, p_val, type_len);
    return constant;
}

void sdma_fill_constant_gen_global_cmd_stride(
    u64 sys_mem_start_addr,
    const void* const_val,
    int data_format,
    int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int stride_enable,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: global_offset = 0x%llx, format=%d, n=%d, c=%d, h=%d, w=%d, "
           "nstride=%d, cstride=%d, hstride=%d, wstride=%d, stride_en=%d\n",
           __func__, sys_mem_start_addr, data_format, dst_N, dst_C, dst_H,
           dst_W, dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
           stride_enable);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    if (data_format == SDMA_FP20) {
        ASSERT(sys_mem_start_addr % ALIGN_BYTES == 0);
        ASSERT_SDMA_WSTRIDE_FP20(dst_W_stride);
    } else {
        // if (stride_enable) {
        //     ASSERT_SDMA_WSTRIDE(dst_W_stride, get_sdma_format_type_len(data_format));
        // }
    }
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT_FS_INFO(!is_lmem(sys_mem_start_addr),
                   "can't be local memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif

    u64 dst_addr = sys_mem_start_addr;
    int constant = get_constant_value(const_val, data_format);
    stride_enable = stride_enable ? __TRUE__ : __FALSE__;

    // dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, src_data_format, special_func,
    // src_wstride, dst_wstride, src_hstride, dst_hstride, src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    SDMA_CONSTANT_GET_PROFILE(dst_N, dst_C, dst_H, dst_W,  0, dst_addr, data_format,  0,
                      dst_W_stride, dst_H_stride, dst_C_stride, dst_N_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = ((u64)stride_enable << 1) |
              ((u64)SDMA_TENSOR << 32) |
              ((u64)SDMA_FUNC_NONE << 37) |
              (1ull << 40) |
              ((u64)data_format << 41);
        high[0] = ((u64)constant << 32);
        low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
        high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
        high[3] = ((u64)dst_N) |
               ((u64)dst_C << 16) |
               ((u64)dst_H << 32) |
               ((u64)dst_W << 48);
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = sdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_TENSOR,
        SDMA_FUNC_NONE, data_format | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_general_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int src_format,
    stride_type src_count,
    int src_is_const,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, dst_addr = 0x%llx, data_format=%d, src_count=%d\n",
        __func__, src_addr, dst_addr, src_format, src_count);

#ifdef USING_CMODEL
    ASSERT(src_format != SDMA_FP20);
    ASSERT_FS_INFO(src_count > 0, "src_count=%d", src_count);
    ASSERT(src_is_const == 0 || src_is_const == 1);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    //  src_count, src_addr, dst_addr, src_data_format, src_is_const, pid_node
    SDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, src_format,  0, pid_node);
    u32 const_value = (u32)get_constant_value(&src_addr, src_format);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = ((u64)SDMA_GENERAL << 32) |
              ((u64)SDMA_FUNC_NONE << 37) |
              ((u64)src_is_const << 40) |
              ((u64)src_format << 41);
        high[0] = (u64)const_value << 32;
        low[1] = ((u64)src_count << 32);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = sdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_GENERAL,
        SDMA_FUNC_NONE, src_format | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_general_cwtrans_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int src_N,  int src_C,
    int src_H,  int src_W,
    int src_format,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int stride_enable,
    int port_id,
    CMD_ID_NODE * pid_node) {
    ASSERT(0);
//     FW_DBG("%s: src_addr = 0x%llx, dst_addr = 0x%llx, src_N=%d, src_C=%d, "
//            "src_H=%d, src_W=%d, src_format=%d, src_N_stride=%d, src_C_stride=%d, "
//            "src_H_stride=%d, dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
//            "stride_en = %d\n",
//            __func__, src_addr, dst_addr,
//            src_N, src_C, src_H, src_W, src_format, src_N_stride, src_C_stride,
//            src_H_stride, dst_N_stride, dst_C_stride, dst_H_stride,
//            stride_enable);
// #ifdef USING_CMODEL
//     ASSERT_SDMA_TENSOR_SIZE(src_N, src_C, src_H, src_W);
//     ASSERT(src_format != SDMA_FP20);
//     ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
//                    "can't be static memory src_addr:0x%llx, dst_addr:0x%llx",
//                    src_addr, dst_addr);
//     ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
//                    "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
//                    src_addr, dst_addr);
// #endif

//     // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format,
//     //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
//     SDMA_CW_TRANS_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format,
//                               1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, stride_enable, pid_node);

//     const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
//     BEGIN_FAST_GEN_CMD_SDMA(port_id)
//         u64 low = 0, high = 0;
//         low = ((u64)stride_enable << 1) |
//               (1ull << 2) |
//               ((u64)SDMA_CW_TRANS << 32) |
//               ((u64)src_format << 41);
//         high = 0;
//         WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
//         low = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
//         high = ((u64)src_H_stride);
//         WRITE_CMD_EX_32BIT(reg_addr, 1, high, low);
//         low = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
//         high = ((u64)dst_H_stride);
//         WRITE_CMD_EX_32BIT(reg_addr, 2, high, low);
//         low = ((u64)src_N) | ((u64)src_C << 16) | ((u64)src_H << 32) | ((u64)src_W << 48);
//         WRITE_CMD_EX_32BIT(reg_addr, 3, 0ull, low);
//         low = src_addr & 0x1ffffffffffful;
//         high = dst_addr & 0x1ffffffffffful;
//         WRITE_CMD_EX_32BIT(reg_addr, 4, high, low);
//         high = sdma_get_lane_mask();
//         WRITE_CMD_EX_32BIT(reg_addr, 5, high, 0ull);
//     END_FAST_GEN_CMD_SDMA(pid_node)
}

void sdma_tensor_general_move_with_mask_gen_cmd(
    u64 src_addr, // global addr or local addr
    u64 mask_addr, // global addr or local addr
    u64 dst_addr, // global addr or local addr
    int src_format,
    int mask_format,
    int N,
    int C,
    int H,
    int W,
    int port_id,
    CMD_ID_NODE * pid_node) {
    ASSERT(0);
    // FW_DBG("%s: "
    //        "src_addr = 0x%llx, mask_addr = 0x%llx, dst_addr = 0x%llx, src_format=%d, mask_format=%d, "
    //        "N=%d, C=%d, H=%d, W=%d, ",
    //        __func__,
    //        src_addr, mask_addr, dst_addr,
    //        src_format, mask_format, N, C, H, W);

    // ASSERT(src_format != SDMA_FP20);
    // ASSERT(dst_addr != SDMA_FP20);

    // int32_t special_func = 0;
    // /// mode0: {N:16bit, C:16bit, H:16bit, W:16bit}
    // /// mode1: {N:16bit, C:16bit, H:1,     W:32bit}
    // if (W >= (1 << 16)) {
    //     special_func = 1;
    //     ASSERT_SDMA_TENSOR_NSIZE(N);
    //     ASSERT_SDMA_TENSOR_NSIZE(C);
    //     ASSERT(H == 1);
    // } else {
    //     special_func = 0;
    //     ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    // }

    // // src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format,  pid_node
    // SDMA_FILTER_PROFILE(N, C, H, W, src_addr, dst_addr, src_format, pid_node);

    // const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    // BEGIN_FAST_GEN_CMD_SDMA(port_id)
    //     u64 low = 0, high = 0;
    //     low = ((u64)SDMA_FILTER << 32) |
    //           ((u64)special_func << 37) |
    //           ((u64)src_format << 41) |
    //           ((u64)mask_format << 45);
    //     high = 0;
    //     WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
    //     WRITE_CMD_EX_32BIT(reg_addr, 1, 0ull, 0ull);
    //     WRITE_CMD_EX_32BIT(reg_addr, 2, 0ull, 0ull);
    //     low = ((u64)N) | ((u64)C << 16) |
    //           ((u64)(W >= (1 << 16) ? (W >> 16) : H) << 32) |
    //           ((u64)(W & 0xffff) << 48);
    //     WRITE_CMD_EX_32BIT(reg_addr, 3, 0ull, low);
    //     low = src_addr & 0x1ffffffffffful;
    //     high = dst_addr & 0x1ffffffffffful;
    //     WRITE_CMD_EX_32BIT(reg_addr, 4, high, low);
    //     low = mask_addr & 0x1ffffffffffful;
    //     high = 0xffffffffffffffff;
    //     WRITE_CMD_EX_32BIT(reg_addr, 5, high, low);
    // END_FAST_GEN_CMD_SDMA(pid_node)
}

void sdma_tensor_move_nonzero_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int src_format, // Only INT8/INT16/INT32
    int dst_format, // Only INT8/INT16/INT32
    int N,
    int C,
    int H,
    int W,
    u32 base_idx,
    int port_id,
    CMD_ID_NODE * pid_node) {
    ASSERT(0);
    // FW_DBG("%s: "
    //        "src_addr = 0x%llx, dst_addr = 0x%llx, src_format=%d, dst_format=%d, "
    //        "N=%d, C=%d, H=%d, W=%d, base_idx=%d, ",
    //        __func__,
    //        src_addr, dst_addr, src_format, dst_format,
    //        N, C, H, W, base_idx);

    // ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);

    // ASSERT(src_format == SDMA_INT8 || src_format == SDMA_INT16 || src_format == SDMA_INT32);
    // ASSERT(dst_format == SDMA_INT8 || dst_format == SDMA_INT16 || dst_format == SDMA_INT32);
    // // src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, pid_node
    // SDMA_NONZERO_PROFILE(N, C, H, W, src_addr, dst_addr, src_format, dst_format, pid_node);

    // const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    // BEGIN_FAST_GEN_CMD_SDMA(port_id)
    //     u64 low = 0, high = 0;
    //     low = ((u64)SDMA_NONZERO << 32) |
    //           ((u64)src_format << 41) |
    //           ((u64)dst_format << 45);
    //     high = 0;
    //     WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
    //     WRITE_CMD_EX_32BIT(reg_addr, 1, 0ull, 0ull);
    //     low = ((u64)base_idx);
    //     WRITE_CMD_EX_32BIT(reg_addr, 2, 0ull, low);
    //     low = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
    //     WRITE_CMD_EX_32BIT(reg_addr, 3, 0ull, low);
    //     low = src_addr & 0x1ffffffffffful;
    //     high = dst_addr & 0x1ffffffffffful;
    //     WRITE_CMD_EX_32BIT(reg_addr, 4, high, low);
    //     high = sdma_get_lane_mask();
    //     WRITE_CMD_EX_32BIT(reg_addr, 5, high, 0ull);
    // END_FAST_GEN_CMD_SDMA(pid_node)
}

unsigned int get_sdma_filter_res_num_gen_cmd(int port_id, CMD_ID_NODE * pid_node) {
    unsigned int tmp = 0;
    poll_all_engine_done(pid_node);
    READ_SDMA_FILTER_RES_NUM(tmp, port_id, pid_node);
    return tmp;
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_gather_gen_cmd(
    u64 src_addr,
    u64 index_addr,
    u64 dst_addr,
    u32 const_val,
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 index_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr=0x%llx, index_addr=0x%llx, dst_addr=0x%llx, "
        "C=%d, src_H=%d, src_W=%d, index_H=%d, start_pos = %d, "
        "src_C_stride=%d, src_H_stride=%d, index_C_stride=%d, index_H_stride=%d, "
        "dst_C_stride=%d, dst_H_stride=%d, "
        "src_format=%d, src_C_is1=%d, index_C_is1=%d, stride_en=%d, \n",
        __func__, src_addr, index_addr, dst_addr,
        C, src_H, src_W, index_H, start_pos,
        src_C_stride, src_H_stride,
        index_C_stride, index_H_stride,
        dst_C_stride, dst_H_stride,
        src_format, src_C_is1, index_C_is1,
        stride_enable);

    ASSERT_SDMA_TENSOR_CSIZE(C);
    ASSERT_SDMA_TENSOR_WSIZE(src_W);
    ASSERT(src_format != SDMA_FP20);

    if (stride_enable) ASSERT_FS(index_H_stride == 1);

   // src_c, src_h, src_w, index_H, src_addr, dst_addr, src_data_format,
   // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    SDMA_GATHER_GET_PROFILE( C, src_H, src_W, index_H, src_addr, dst_addr, src_format,
                             1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride,
                             index_H_stride, index_C_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = ((u64)stride_enable << 1) |
              ((u64)SDMA_GATHER << 32) |
              ((u64)src_format << 41);
        high[0] = (u64)const_val << 32;
        low[1] = ((u64)src_C_stride) | ((u64)src_H_stride << 32);
        high[1] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[2] = ((u64)index_C_stride) | ((u64)start_pos << 32);
        high[2] = ((u64)(src_C_is1 ? (1<<16) : (C<<16))) | ((u64)src_H << 32);
        low[3] = ((u64)src_W) | ((u64)index_H << 32);
        high[3] = ((u64)(index_C_is1 ? 1 : C) << 16) | ((u64)src_H << 32);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        low[5] = index_addr & 0x1ffffffffffful;
        high[5] = sdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_GATHER,
        0, src_format | SDMA_S2S << 4, pid_node, high, low, 6);
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_scatter_gen_cmd(
    u64 src_addr,
    u64 index_addr,
    u64 dst_addr,
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 dst_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int port_id,
    int inplace_add,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr=0x%llx, index_addr=0x%llx, dst_addr=0x%llx, "
        "C=%d, src_H=%d, src_W=%d, dst_H=%d, start_pos=%d,"
        "src_C_stride=%d, src_H_stride=%d, index_C_stride=%d, index_H_stride=%d, "
        "dst_C_stride=%d, dst_H_stride=%d, "
        "src_format=%d, src_C_is1=%d, index_C_is1=%d, stride_en=%d, inplace_add=%d\n",
            __func__, src_addr, index_addr, dst_addr,
            C, src_H, src_W, dst_H, start_pos,
            src_C_stride, src_H_stride,
            index_C_stride, index_H_stride,
            dst_C_stride, dst_H_stride,
            src_format, src_C_is1, index_C_is1,
            stride_enable, inplace_add);

    ASSERT_FS(C <= SDMA_MAX_C);
    ASSERT_FS(src_W <= SDMA_MAX_W);
    if (stride_enable) ASSERT_FS(index_H_stride == 1);
    if(src_format == SDMA_FP8_E4M3 || src_format == SDMA_FP8_E5M2) {
        inplace_add = 0;
    }

    //  src_c, src_h, src_w, dst_h, src_addr, dst_addr, src_data_format, src_C_is1, index_C_is1,
    //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    SDMA_SCATTER_GET_PROFILE( C, src_H, src_W, dst_H, src_addr, dst_addr, src_format, src_C_is1, index_C_is1,
                1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, index_H_stride, index_C_stride,
                stride_enable, inplace_add, pid_node);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = ((u64)stride_enable << 1) |
              ((u64)SDMA_SCATTER << 32) |
              ((u64)inplace_add << 37) |
              ((u64)src_format << 41);
        low[1] = ((u64)src_C_stride) | ((u64)src_H_stride << 32);
        high[1] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[2] = ((u64)index_C_stride) | ((u64)start_pos << 32);
        high[2] = ((u64)(src_C_is1 ? (1<<16) : (C<<16))) | ((u64)src_H << 32);
        low[3] = ((u64)src_W) | ((u64)dst_H << 32);
        high[3] = ((u64)(index_C_is1 ? 1 : C) << 16) | ((u64)src_H << 32);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        low[5] = index_addr & 0x1ffffffffffful;
        high[5] = sdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_SCATTER,
        inplace_add, src_format | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_tensor_reverse_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t src_n_stride,
    uint32_t src_c_stride,
    uint32_t src_h_stride,
    uint32_t dst_n_stride,
    uint32_t dst_c_stride,
    uint32_t dst_h_stride,
    int32_t reverse_axis,
    int32_t data_format,
    int port_id,
    CMD_ID_NODE *pid_node)
{
    ASSERT(0);
//     FW_DBG("%s, "
//            "src_addr:0x%llx, dst_addr:0x%llx, N:%d, C:%d, "
//            "H:%d, W:%d, src_n_stride:%d, src_c_stride:%d, "
//            "src_h_stride:%d, dst_n_stride:%d, dst_c_stride:%d, "
//            "dst_h_stride:%d, data_format::%d, reversed_axis:%d\n",
//            __func__, src_addr, dst_addr, N, C, H, W, src_n_stride,
//            src_c_stride, src_h_stride, dst_n_stride,
//            dst_c_stride, dst_h_stride, data_format,
//            reverse_axis);
// #ifdef USING_CMODEL
//     ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
//     ASSERT(reverse_axis >= 0 && reverse_axis <= 3);
//     ASSERT(data_format != SDMA_FP20);
//     ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
//                    "can't be static memory src_addr:0x%llx, dst_addr:0x%llx",
//                    src_addr, dst_addr);
//     ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
//                    "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
//                    src_addr, dst_addr);
// #endif

//     const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
//     int32_t special_func = 3 - reverse_axis;
//     BEGIN_FAST_GEN_CMD_SDMA(port_id)
//         u64 low = 0, high = 0;
//         low = (1ull << 1) |
//               (1ull << 2) |
//               ((u64)SDMA_REVERSE << 32) |
//               ((u64)special_func << 37) |
//               ((u64)data_format << 41);
//         high = 0;
//         WRITE_CMD_EX_32BIT(reg_addr, 0, high, low);
//         low = ((u64)src_n_stride) | ((u64)src_c_stride << 32);
//         high = ((u64)src_h_stride) | ((u64)1 << 32);
//         WRITE_CMD_EX_32BIT(reg_addr, 1, high, low);
//         low = ((u64)dst_n_stride) | ((u64)dst_c_stride << 32);
//         high = ((u64)dst_h_stride) | ((u64)1 << 32);
//         WRITE_CMD_EX_32BIT(reg_addr, 2, high, low);
//         low = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
//         high = low;
//         WRITE_CMD_EX_32BIT(reg_addr, 3, high, low);
//         low = src_addr & 0x1ffffffffffful;
//         high =dst_addr & 0x1ffffffffffful;
//         WRITE_CMD_EX_32BIT(reg_addr, 4, high, low);
//         high = sdma_get_lane_mask();
//         WRITE_CMD_EX_32BIT(reg_addr, 5, high, 0ull);
//     END_FAST_GEN_CMD_SDMA(pid_node)
}

void sdma_lossy_compress_gen_cmd(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d \n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              ((u64)SDMA_LOSSY_COMPRESS << 32) |
              ((u64)SDMA_FUNC_NONE << 37);
        low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
        high[1] = ((u64)src_H_stride) | ((u64)dst_N_stride << 32);
        low[2] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[3] = ((u64)N) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        high[3] = ((u64)N) |
               ((u64)C << 16) |
               ((u64)H << 32) |
               ((u64)W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_LOSSY_COMPRESS,
        SDMA_FUNC_NONE, 0 | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_lossy_decompress_gen_cmd(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d \n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              ((u64)SDMA_LOSSY_DECOMPRESS << 32) |
              ((u64)SDMA_FUNC_NONE << 37);
        low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
        high[1] = ((u64)src_H_stride) | ((u64)dst_N_stride << 32);
        low[2] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[3] = ((u64)N) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        high[3] = ((u64)N) |
               ((u64)C << 16) |
               ((u64)H << 32) |
               ((u64)W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_LOSSY_DECOMPRESS,
        SDMA_FUNC_NONE, 0 | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_lossy_compress_reduce_gen_cmd(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "reduce_psum_op=%d, reduce_opcode=%d\n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride,
        reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_lmem(src_addr),
                   "can't be static or local memory, src_addr:0x%llx",
                   src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
    u32 are_type = get_sdma_are_dtype(SDMA_FP20);
    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              ((u64)SDMA_LOSSY_COMPRESS << 32) |
              ((u64)SDMA_FUNC_NONE << 37);
        low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
        high[1] = ((u64)src_H_stride) | ((u64)dst_N_stride << 32);
        low[2] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[3] = ((u64)N) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        high[3] = ((u64)N) |
               ((u64)C << 16) |
               ((u64)H << 32) |
               ((u64)W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        low[5] = ((u64)are_type) |
              ((u64)reduce_opcode << 4) |
              ((u64)reduce_psum_op << 8) |
              ((u64)1 << 15);
        high[5] = high[4];
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_LOSSY_COMPRESS,
        SDMA_FUNC_NONE, SDMA_FP20 | SDMA_S2S << 4, pid_node, high, low, 6);
}

void sdma_lossy_decompress_reduce_gen_cmd(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "reduce_psum_op=%d, reduce_opcode=%d\n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride,
        reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_lmem(src_addr),
                   "can't be static or local memory, src_addr:0x%llx",
                   src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
    u32 are_type = get_sdma_are_dtype(SDMA_FP32);
    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              ((u64)SDMA_LOSSY_DECOMPRESS << 32) |
              ((u64)SDMA_FUNC_NONE << 37);
        low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
        high[1] = ((u64)src_H_stride) | ((u64)dst_N_stride << 32);
        low[2] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
        low[3] = ((u64)N) |
              ((u64)C << 16) |
              ((u64)H << 32) |
              ((u64)W << 48);
        high[3] = ((u64)N) |
               ((u64)C << 16) |
               ((u64)H << 32) |
               ((u64)W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        low[5] = ((u64)are_type) |
              ((u64)reduce_opcode << 4) |
              ((u64)reduce_psum_op << 8) |
              ((u64)1<<15);
        high[5] = high[4];
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_LOSSY_DECOMPRESS,
        SDMA_FUNC_NONE, SDMA_FP32 | SDMA_S2S << 4, pid_node, high, low, 6);
}

// Only support GDMA write
void sdma_tensor_reduce_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose,  // N/C transpose, fp20 not support
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx,  src_N=%d, src_C=%d, src_H=%d, "
        "src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx,  dst_N=%d, dst_C=%d, dst_H=%d, "
        "dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d\n",
        __func__, src_addr, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride);

#ifdef USING_CMODEL
    ASSERT_SDMA_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    if (src_format == SDMA_FP20) {
        // ASSERT(direction == SDMA_VALUE_DIR_S2S);
        ASSERT(transpose == 0);
        ASSERT(src_addr % 128 == 0);
        ASSERT(dst_addr % 128 == 0);
        ASSERT_SDMA_COMPACT_FP20((u32)src_N, (u32)src_C, (u32)src_H, (u32)src_W, src_N_stride, src_C_stride, src_H_stride, src_W_stride)
        ASSERT_SDMA_COMPACT_FP20((u32)dst_N, (u32)dst_C, (u32)dst_H, (u32)dst_W, dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride)
    } else {
        // int type_len = get_sdma_format_type_len(src_format);
        // ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
        // ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    if(src_format == SDMA_INT32) {
        ASSERT(reduce_opcode != SDMA_ARE_MUL);
    }
#endif
    // just for compatible with old param_list, after tv_gen test, we should assert it
    transpose = 0;
    int special_func = transpose ? SDMA_FUNC_TRANS : SDMA_FUNC_NONE;
    u32 are_type = get_sdma_are_dtype(src_format);

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    SDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, special_func, 3,
                            src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride,
                            dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = port_id == -1 ? SDMA_CMD_BASE_ADDR : VSDMA_CMD_BASE_ADDR(port_id);
    BEGIN_FAST_GEN_CMD_SDMA(port_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              ((u64)SDMA_TENSOR << 32) |
              ((u64)special_func << 37) |
              ((u64)src_format << 41);
        low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
        high[1] = ((u64)src_H_stride) | ((u64)src_W_stride << 32);
        low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
        high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
        low[3] = ((u64)src_N) |
              ((u64)src_C << 16) | ((u64)src_H << 32) |
              ((u64)src_W << 48);
        high[3] = ((u64)dst_N) |
               ((u64)dst_C << 16) |
               ((u64)dst_H << 32) |
               ((u64)dst_W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = (dst_addr & 0x1ffffffffffful);
        low[5] = ((u64)are_type) |
              ((u64)reduce_opcode << 4) |
              ((u64)reduce_psum_op << 8) |
              ((u64)1<<15);
        high[5] = sdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX_32BIT(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_SDMA(pid_node)
    profile_time_set_node(fast_cmd->sdma_type, SDMA_TENSOR,
        special_func, src_format | SDMA_S2S << 4, pid_node, high, low, 6);
}
