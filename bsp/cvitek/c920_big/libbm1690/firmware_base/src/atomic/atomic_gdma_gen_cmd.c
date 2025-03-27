#include "atomic_dma_utils.h"
#include "atomic_gdma_gen_cmd.h"
#include "firmware_common_macro.h"
#include "firmware_debug_macro.h"
#include "gdma_reg_def.h"
#include "gdma_reg_value.h"
#include "gen_cmd.h"

#define ASSERT_TENSOR_NSIZE(n) \
    ASSERT_FS_INFO(n>0 && n<=GDMA_MAX_N, #n "=%d", n)

#define ASSERT_TENSOR_CSIZE(c) \
    ASSERT_FS_INFO(c>0 && c<=GDMA_MAX_C, #c "=%d", c)

#define ASSERT_TENSOR_HSIZE(h) \
    ASSERT_FS_INFO(h>0 && h<=GDMA_MAX_H, #h "=%d", h)

#define ASSERT_TENSOR_WSIZE(w) \
    ASSERT_FS_INFO(w>0 && w<=GDMA_MAX_W, #w "=%d", w)

#define ASSERT_TENSOR_SIZE(n,c,h,w) \
    ASSERT_TENSOR_NSIZE(n); \
    ASSERT_TENSOR_CSIZE(c); \
    ASSERT_TENSOR_HSIZE(h); \
    ASSERT_TENSOR_WSIZE(w)

#define ASSERT_WSTRIDE(wstr, byte_len) \
    ASSERT_FS_INFO((wstr * byte_len <= MAX_WSTRIDE_BYTE_LEN) && (wstr != 0), "W stride byte len = %d", wstr * byte_len)

#define ASSERT_WSTRIDE_FP20(wstr) \
    ASSERT_FS_INFO(wstr == 1, "When data type is fp20, W stride should be 1")

#define ASSERT_COMPACT_FP20(n,c,h,w,nstr,cstr,hstr,wstr) \
    ASSERT_WSTRIDE_FP20(wstr); \
    ASSERT_FS_INFO(hstr == (w), "When data type is fp20, 51 elements constitute fp20 block"); \
    ASSERT_FS_INFO(cstr == (h * hstr), "When data type is fp20, c stride should be compacted"); \
    ASSERT_FS_INFO(nstr == (c * cstr), "When data type is fp20, n stride should be compacted");



inline static u64 gdma_get_lane_mask() {
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
    if (en_lane_mask && access(path_int, F_OK) == 0 && atoi(en_lane_mask) == 1) {
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

void tensor_align_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    int src_format,
    int direction,
    int transpose,  // N/C transpose
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, "
           "sys_mem_start_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
           "data_format=%d, direction=%d, transpose=%d\n",
           __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
           src_N, src_C, src_H, src_W, src_format, direction, transpose);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "directin=%d", direction);
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT(src_format != GDMA_FP20);
#endif

    u64 sys_addr = sys_mem_start_addr;
    u64 local_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int is_local_to_sys = direction == GDMA_L2S;
    u64 src_addr = is_local_to_sys ? local_addr : sys_addr;
    u64 dst_addr = is_local_to_sys ? sys_addr : local_addr;
    int special_func = transpose ? GDMA_FUNC_TRANS : GDMA_FUNC_NONE;

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W, src_addr, dst_addr, src_format, direction, special_func, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[3] = ((u64)src_N) |
            ((u64)src_C << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_stride_move_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int src_N,
        int src_C,
        int src_H,
        int src_W,
        stride_type src_N_stride,
        stride_type src_C_stride,
        stride_type src_H_stride,
        stride_type src_W_stride,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int src_format,
        int direction,
        int transpose,  // N/C transpose
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, sys_mem_start_addr = 0x%llx, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d, "
        "src_format=%d, direction=%d, transpose=%d\n",
            __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
            src_N, src_C, src_H, src_W,
            src_N_stride, src_C_stride, src_H_stride, src_W_stride,
            dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
            src_format, direction, transpose);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "directin=%d", direction);
    // int type_len = get_gdma_format_type_len(src_format);
    // ASSERT_WSTRIDE(src_W_stride, type_len);
    // ASSERT_WSTRIDE(dst_W_stride, type_len);
    ASSERT(src_format != GDMA_FP20 && src_format < GDMA_FORMAT_NUM);
#endif

    u64 sys_addr = sys_mem_start_addr;
    u64 local_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int special_func = transpose ? GDMA_FUNC_TRANS: GDMA_FUNC_NONE;
    if(transpose) {
        src_W_stride = 1;
        dst_W_stride = 1;
    }
    int is_local_to_sys = direction == GDMA_L2S;
    u64 src_addr = is_local_to_sys ? local_addr : sys_addr;
    u64 dst_addr = is_local_to_sys ? sys_addr : local_addr;

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable,  pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction, special_func, 3,
                      src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
    high[1] = ((u64)src_H_stride) | ((u64)src_W_stride << 32);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
    low[3] = ((u64)src_N) |
            ((u64)src_C << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_general_move_gen_cmd(
        u64 src_addr, //local_addr or global_addr
        int src_local_idx, //use only from local_mem
        int src_N,
        int src_C,
        int src_H,
        int src_W,
        stride_type src_N_stride,
        stride_type src_C_stride,
        stride_type src_H_stride,
        stride_type src_W_stride,
        int src_format,
        u64 dst_addr, //local_addr or global_addr
        int dst_local_idx, //use only to local_mem
        int dst_N,
        int dst_C,
        int dst_H,
        int dst_W,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int direction,
        int transpose,  // N/C transpose
        CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_local_idx=%d,  src_N=%d, src_C=%d, src_H=%d, "
        "src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx, dst_local_idx=%d,  dst_N=%d, dst_C=%d, dst_H=%d, "
        "dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d\n",
        __func__, src_addr, src_local_idx, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_local_idx, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    if (src_format == GDMA_FP20) {
        ASSERT(direction == GDMA_S2S);
        ASSERT(transpose == 0);
        ASSERT(src_addr % 128 == 0);
        ASSERT(dst_addr % 128 == 0);
        ASSERT_WSTRIDE_FP20(src_W_stride);
        ASSERT_WSTRIDE_FP20(dst_W_stride);
    } else {
        // int type_len = get_gdma_format_type_len(src_format);
        // ASSERT_WSTRIDE(src_W_stride, type_len);
        // ASSERT_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    if (SRC_IS_LOCAL(direction)) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    } else {
        src_addr = src_addr;
    }
    if (DST_IS_LOCAL(direction)) {
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    } else {
        dst_addr = dst_addr;
    }
    int special_func = transpose ? GDMA_FUNC_TRANS : GDMA_FUNC_NONE;
    if(transpose) {
        dst_W_stride = 1;
        src_W_stride = 1;
    }

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction, special_func, 3,
                      src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_compact_move_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int src_N, int src_C, int src_H, int src_W,
        int src_format,
        int direction,
        int transpose,  // N/C transpose
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, "
           "sys_mem_start_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, format=%d, "
           "direction=%d, transpose=%d\n",
           __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
           src_N, src_C, src_H, src_W, src_format, direction, transpose);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_FS(direction == GDMA_L2S ||
              direction == GDMA_S2L);
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT(src_format != GDMA_FP20);
#endif

    u64 sm_addr = sys_mem_start_addr;
    u64 lm_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    stride_type W_stride = 1;
    stride_type H_stride = src_W * W_stride;
    stride_type C_stride = src_H * H_stride;

    int is_local_to_sys = direction == GDMA_L2S;

    u64 src_addr = 0; // is_local_to_sys? local_addr : sys_addr;
    u64 dst_addr = 0; // is_local_to_sys ? sys_addr : local_addr;
    stride_type src_nstride = 0;
    stride_type dst_nstride = 0;
    int dst_C = transpose ? src_N : src_C;
    if (is_local_to_sys) {
        src_addr = lm_addr;
        dst_addr = sm_addr;
        src_nstride = (src_C + local_mem_idx + NPU_NUM - 1)/NPU_NUM * C_stride;
        dst_nstride = dst_C * C_stride;
    } else {
        src_addr = sm_addr;
        dst_addr = lm_addr;
        src_nstride = src_C * C_stride;
        dst_nstride = (dst_C + local_mem_idx + NPU_NUM - 1)/NPU_NUM * C_stride;
    }

    int special_func = transpose ? GDMA_FUNC_TRANS: GDMA_FUNC_NONE;

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction, special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction, special_func, 1,
                      W_stride, W_stride, H_stride, H_stride, C_stride, C_stride, src_nstride, dst_nstride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_nstride) | ((u64)C_stride << 32);
    high[1] = ((u64)H_stride) | ((u64)W_stride << 32);
    low[2] = ((u64)dst_nstride) | ((u64)C_stride << 32);
    high[2] = ((u64)H_stride) | ((u64)W_stride << 32);
    low[3] = ((u64)src_N) |
            ((u64)src_C << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void matrix_move_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int sec_size,
        int row_num, int col_num, //means matrix in sys_mem  is row*col,
        int src_format,
        int direction,
        int transpose,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, "
           "sys_mem_start_addr = 0x%llx, sec_size=%d, row_num=%d, col_num=%d, "
           "src_format=%d, direction=%d, transpose=%d\n",
           __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
           sec_size, row_num, col_num, src_format, direction, transpose);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "direction=%d", direction);
    ASSERT_FS_INFO(sec_size > 0 && row_num > 0 && col_num > 0,
                   "row=%d, col=%d, sec=%d", row_num, col_num, sec_size);
    ASSERT_FS(row_num <= GDMA_MAX_H && col_num <= GDMA_MAX_W);
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif

    u64 sm_addr = sys_mem_start_addr;
    u64 lm_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int is_local_to_sys = direction == GDMA_L2S;
    u64 src_addr = is_local_to_sys ? lm_addr : sm_addr;
    u64 dst_addr = is_local_to_sys ? sm_addr : lm_addr;

    u64 lm_row = transpose ? col_num : row_num;
    int lm_col = transpose ? row_num : col_num;
    int sec_num = (lm_col + sec_size - 1) / sec_size;

    int special_func = transpose ? GDMA_FUNC_TRANS : GDMA_FUNC_NONE;

    // row_num, col_num, sec_size, src_addr, dst_addr, src_data_format, direction, transpose,
    // global_row_stride, local_row_stride, local_sec_stride, stride_enable, pid_node
    GDMA_MATRIX_GET_PROFILE(row_num, col_num, sec_size, src_addr, dst_addr, src_format, direction, transpose, 0, 0, 0, 0, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    if (is_local_to_sys) {
        low[0] = gdma_get_cache_en() << 4 |
                ((u64)GDMA_MATRIX << 32) |
                ((u64)special_func << 37) |
                ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        low[3] = ((u64)lm_row) |
                ((u64)sec_num << 16) |
                ((u64)1<<32) |
                ((u64)sec_size << 48);
        high[3] = ((u64)1) |
                ((u64)1<<16) |
                ((u64)row_num << 32) |
                ((u64)col_num << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();

    } else {
        low[0] = gdma_get_cache_en() << 4 |
            ((u64)GDMA_MATRIX << 32) |
                ((u64)special_func << 37) |
                ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        low[3] =  ((u64)1) |
                ((u64)1<<16) |
                ((u64)row_num << 32) |
                ((u64)col_num << 48);
        high[3] = ((u64)lm_row) |
                ((u64)sec_num << 16) |
                ((u64)1<<32) |
                ((u64)sec_size << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();
    }
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_MATRIX,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void matrix_stride_move_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int sec_size,
        int row_num, int col_num,
        stride_type global_row_stride,
        stride_type local_row_stride,
        stride_type local_sec_stride,
        int src_format,
        int direction,
        int transpose,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, "
           "sys_mem_start_addr = 0x%llx, sec_size=%d, row_num=%d, col_num=%d, "
           "global_row_stride=%d, local_sec_stride=%d, local_row_stride=%d, "
           "src_format=%d, direction=%d, transpose=%d, thread_id=%d, bd_cmd_id=%u, "
           "gdma_cmd_id=%u\n",
           __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
           sec_size, row_num, col_num, global_row_stride, local_sec_stride,
           local_row_stride, src_format, direction, transpose, thread_id,
           pid_node->bd_cmd_id, pid_node->gdma_cmd_id);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "direction=%d", direction);
    ASSERT_FS_INFO(sec_size > 0 && row_num > 0 && col_num > 0,
                   "row=%d, col=%d, sec=%d", row_num, col_num, sec_size);
    ASSERT_FS(row_num <= GDMA_MAX_H && col_num <= GDMA_MAX_W);
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif

    u64 sm_addr = sys_mem_start_addr;
    u64 lm_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int is_local_to_sys = direction == GDMA_L2S;
    u64 src_addr = is_local_to_sys ? lm_addr : sm_addr;
    u64 dst_addr = is_local_to_sys ? sm_addr : lm_addr;

    u64 lm_row = transpose ? col_num : row_num;
    int lm_col = transpose ? row_num : col_num;
    int sec_num = (lm_col + sec_size - 1) / sec_size;

    int special_func = transpose ? GDMA_FUNC_TRANS : GDMA_FUNC_NONE;

    // row_num, col_num, sec_size, src_addr, dst_addr, src_data_format, direction, transpose,
    // global_row_stride, local_row_stride, local_sec_stride, stride_enable, pid_node
    GDMA_MATRIX_GET_PROFILE(row_num, col_num, sec_size, src_addr, dst_addr, src_format, direction, transpose,
                      global_row_stride, local_row_stride, local_sec_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    if (is_local_to_sys) {
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
                ((u64)GDMA_MATRIX << 32) |
                ((u64)special_func << 37) |
                ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        low[1] = ((u64)local_row_stride) | ((u64)local_sec_stride << 32);
        high[2] = ((u64)global_row_stride);
        low[3] = ((u64)lm_row) |
                ((u64)sec_num << 16) |
                ((u64)1 << 32) |
                ((u64)sec_size << 48);
        high[3] = ((u64) 1) |
                ((u64) 1<< 16) |
                ((u64)row_num << 32) |
                ((u64)col_num << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();
    } else {
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
                ((u64)GDMA_MATRIX << 32) |
                ((u64)special_func << 37) |
                ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        high[1] = ((u64)global_row_stride);
        low[2] = ((u64)local_row_stride) | ((u64)local_sec_stride << 32);
        low[3] = ((u64)1) |
                ((u64)1<<16) |
                ((u64)row_num << 32) |
                ((u64)col_num << 48);
        high[3] = ((u64)lm_row) |
                ((u64)sec_num << 16) |
                ((u64)sec_size << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();
    }
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_MATRIX,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void general_matrix_move_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int sec_size,
        int row_num, int col_num,
        stride_type row_stride,
        int src_format,
        int direction,
        int transpose,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    int lm_col = transpose ? row_num: col_num;
    int sec_num = (lm_col + sec_size - 1) / sec_size;
    int align_factor = 4 / get_gdma_format_type_len(src_format);
    stride_type lm_sec_stride = ALIGN(sec_size, EU_NUM * align_factor);
    stride_type lm_row_stride = (local_mem_idx + sec_num + NPU_NUM - 1)/NPU_NUM * lm_sec_stride;
    matrix_stride_move_gen_cmd(
        local_mem_start_addr,
        local_mem_idx,
        sys_mem_start_addr,
        sec_size,
        row_num, col_num,
        row_stride,
        lm_row_stride,
        lm_sec_stride,
        src_format,
        direction,
        transpose,
        thread_id,
        pid_node);
}

static inline int get_constant_value(const void * p_val, int format) {
    int constant = 0;
    int type_len = get_gdma_format_type_len(format);
    if (format == GDMA_FP20) {
        type_len = 4;
    }
    memcpy(&constant, p_val, type_len);
    return constant;
}

void fill_constant_gen_local_cmd_stride(
        int local_mem_start_addr,
        int local_mem_idx,
        const void *const_val,
        int data_format,
        int dst_N, int dst_C, int dst_H, int dst_W,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int stride_enable,
        int use_broadcast,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_offset = 0x%x, local_idx=%d, format=%d, n=%d, c=%d, "
           "h=%d, w=%d, nstride=%d, cstride=%d, hstride=%d, wstride=%d, "
           "stride_en=%d, broadcast=%d\n",
           __func__, local_mem_start_addr, local_mem_idx, data_format, dst_N,
           dst_C, dst_H, dst_W, dst_N_stride, dst_C_stride, dst_H_stride,
           dst_W_stride, stride_enable, use_broadcast);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    if (stride_enable) {
        // ASSERT_WSTRIDE(dst_W_stride, get_gdma_format_type_len(data_format));
    }
    if (use_broadcast) {
        if (stride_enable) {
          ASSERT_FS_INFO(dst_W_stride == 1,
                         "broadcast only support wstride == 1, stride:%d",
                         dst_W_stride);
        }
        ASSERT_FS_INFO(local_mem_idx + dst_C <= NPU_NUM,
                       "broadcast cannot overflow NPU_NUM");
    }
#endif

    u64 dst_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int constant  = get_constant_value(const_val, data_format);
    u64 lane_mask = gdma_get_lane_mask();
    int special_func = (use_broadcast) ? GDMA_FUNC_BROADCAST :
                       GDMA_FUNC_NONE;
    if(use_broadcast) {
        dst_W_stride = 1;
        lane_mask = 0xffffffffffffffff;
    }
    stride_enable = stride_enable ? __TRUE__ : __FALSE__;

    // dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, src_data_format, special_func, is_local_dst,
    //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_CONSTANT_GET_PROFILE(dst_N, dst_C, dst_H, dst_W,  0, dst_addr, data_format,  special_func, 1,
                      dst_W_stride, dst_H_stride, dst_C_stride, dst_N_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            (1ull << 40) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20) | ((u64)constant << 32);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
    high[3] = ((u64)dst_N) |
            ((u64)dst_C << 16) |
            ((u64)dst_H << 32) |
            ((u64)dst_W << 48);
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = lane_mask;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, data_format | GDMA_S2L << 4, pid_node, high, low, 6);
}

void fill_constant_gen_global_cmd_stride(
        u64 sys_mem_start_addr,
        const void* const_val,
        int data_format,
        int dst_N, int dst_C, int dst_H, int dst_W,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int stride_enable,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: global_offset = 0x%llx, format=%d, n=%d, c=%d, h=%d, w=%d, "
           "nstride=%d, cstride=%d, hstride=%d, wstride=%d, stride_en=%d\n",
           __func__, sys_mem_start_addr, data_format, dst_N, dst_C, dst_H,
           dst_W, dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
           stride_enable);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    if (data_format == GDMA_FP20) {
        ASSERT(sys_mem_start_addr % 128 == 0);
        ASSERT_WSTRIDE_FP20(dst_W_stride);
    } else {
        if (stride_enable) {
            // ASSERT_WSTRIDE(dst_W_stride, get_gdma_format_type_len(data_format));
        }
    }
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif

    u64 dst_addr = sys_mem_start_addr;
    int constant = get_constant_value(const_val, data_format);
    stride_enable = stride_enable ? __TRUE__ : __FALSE__;

    // dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, src_data_format, special_func, is_local_dst,
    // src_wstride, dst_wstride, src_hstride, dst_hstride, src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_CONSTANT_GET_PROFILE(dst_N, dst_C, dst_H, dst_W,  0, dst_addr, data_format,  0, 0,
                      dst_W_stride, dst_H_stride, dst_C_stride, dst_N_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)GDMA_FUNC_NONE << 37) |
            (1ull << 40) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20) | ((u64)constant << 32);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
    high[3] = ((u64)dst_N) |
            ((u64)dst_C << 16) |
            ((u64)dst_H << 32) |
            ((u64)dst_W << 48);
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      GDMA_FUNC_NONE, data_format | GDMA_S2S << 4, pid_node, high, low, 6);
}

void general_gdma_gen_cmd(
        u64 src_addr,
        u64 dst_addr,
        int src_format,
        stride_type src_count,
        int src_is_const,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, dst_addr = 0x%llx, data_format=%d, src_count=%d\n",
        __func__, src_addr, dst_addr, src_format, src_count);

#ifdef USING_CMODEL
    ASSERT(src_format != GDMA_FP20);
    ASSERT_FS_INFO(src_count > 0, "src_count=%d", src_count);
    ASSERT(src_is_const == 0 || src_is_const == 1);
#endif
    bool src_is_local = __FALSE__, dst_is_local = __FALSE__;
    if (!src_is_const) {
        if (is_lmem(src_addr)) {
          src_addr = CALC_LOCAL_ADDR(0, src_addr);
          src_is_local = __TRUE__;
        } else if (is_smem(src_addr)) {
          src_addr = CALC_STATIC_ADDR(src_addr);
        }
    }
    if (is_lmem(dst_addr)) {
        dst_addr = CALC_LOCAL_ADDR(0, dst_addr);
        dst_is_local = __TRUE__;
    } else if (is_smem(dst_addr)) {
        dst_addr = CALC_STATIC_ADDR(dst_addr);
    }
    //  src_count, src_addr, dst_addr, src_data_format, src_is_const, pid_node
    GDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, src_format,  0, pid_node);
    u32 const_value = (u32)get_constant_value(&src_addr, src_format);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = gdma_get_cache_en() << 4 |
        ((u64)GDMA_GENERAL << 32) |
            ((u64)GDMA_FUNC_NONE << 37) |
            ((u64)src_is_const << 40) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20) | ((u64)const_value << 32);
    low[1] = ((u64)src_count << 32);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    u32 direction = dst_is_local? (src_is_local ? GDMA_L2L : GDMA_S2L) : (src_is_local ? GDMA_L2S : GDMA_S2S);
    profile_time_set_node(ENGINE_GDMA, GDMA_GENERAL,
      GDMA_FUNC_NONE, src_format | direction << 4, pid_node, high, low, 6);
}

void general_broadcast_gen_cmd(
        u64 src_addr, // src_addr or constant
        int local_mem_start_addr,
        int local_mem_idx,
        int src_format,
        stride_type src_count,
        int dst_c, // Broadcast src_count data to dst_c lanes, local_idx + dst_c <= NPU_NUM
        int src_is_const, // 0: not const, 1: is const
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, dst_loffset = 0x%x, dst_lm_idx = %d, "
           "data_format=%d, src_count=%d, dst_c=%d, src_is_const=%d\n",
           __func__, src_addr, local_mem_start_addr, local_mem_idx, src_format,
           src_count, dst_c, src_is_const);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(src_count > 0, "src_count=%d", src_count);
    ASSERT_FS_INFO(local_mem_idx + dst_c <= NPU_NUM,
                   "broadcast cannot over NPU_NUM, idx=%d, dst_c=%d",
                   local_mem_idx, dst_c);
    ASSERT_FS_INFO(local_mem_start_addr < LOCAL_MEM_SIZE,
                   "broadcast need offset for per npu, local_mem_start_addr:%d",
                   local_mem_start_addr);
#endif
    bool src_is_local = __FALSE__;
    u64 dst_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    int constant = 0;
    int fill_const = __FALSE__;
    if (src_is_const) {
        fill_const = __TRUE__;
        constant = get_constant_value(&src_addr, src_format);
    } else if (is_lmem(src_addr)) {
        src_addr = CALC_LOCAL_ADDR(0, src_addr);
        src_is_local = __TRUE__;
    } else if (is_smem(src_addr)) {
        src_addr = CALC_STATIC_ADDR(src_addr);
    }

    //  src_count, src_addr, dst_addr, src_data_format, src_is_const, pid_node
    GDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, src_format,  src_is_const, pid_node);

    u64 low[6] = {0}, high[6] = {0};
    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    low[0] = gdma_get_cache_en() << 4 |
        ((u64)GDMA_GENERAL << 32) |
            ((u64)1 << 37) |
            ((u64)fill_const << 40) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20) | ((u64)constant << 32);
    low[1] = ((u64)src_count << 32);
    high[3] = ((u64)dst_c << 16);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = 0xffffffffffffffff;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_GENERAL,
      1, src_format | (src_is_local ? GDMA_L2L : GDMA_S2L) << 4, pid_node, high, low, 6);
}

void general_cwtrans_gen_cmd(
        u64 src_addr, // local_addr or global_addr
        int src_local_idx, // use only from local_mem
        u64 dst_addr, // local_addr or global_addr
        int dst_local_idx, // use only from local_mem
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
        int direction,  // Support combination of Global、Local and L2 for src and dst
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, src_lm_idx = %d, "
           "dst_addr = 0x%llx, dst_lm_idx = %d, src_N=%d, src_C=%d, "
           "src_H=%d, src_W=%d, src_format=%d, src_N_stride=%d, src_C_stride=%d, "
           "src_H_stride=%d, dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
           "stride_en = %d, direction = %d\n",
           __func__, src_addr, src_local_idx, dst_addr, dst_local_idx,
           src_N, src_C, src_H, src_W, src_format, src_N_stride, src_C_stride,
           src_H_stride, dst_N_stride, dst_C_stride, dst_H_stride,
           stride_enable, direction);

    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);

    int from_lmem = direction == GDMA_L2S ||
                    direction == GDMA_L2L;
    if (from_lmem) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    }
    int to_lmem = direction == GDMA_L2L ||
                  direction == GDMA_S2L;
    if (to_lmem) {
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    }

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,
    //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_CW_TRANS_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction,
                       1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
            (1ull << 2) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_CW_TRANS << 32) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
    high[1] = ((u64)src_H_stride);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride);
    low[3] = ((u64)src_N) | ((u64)src_C << 16) | ((u64)src_H << 32) | ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_CW_TRANS,
      src_format, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_general_move_with_mask_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 mask_addr, // local_addr or global_addr
    int mask_local_idx, // use only from local_mem
    int mask_in_lmem, // 1: mask is in local mem, 0: mask is in global mem
    u64 dst_addr, // global addr only
    int src_format,
    int mask_format,
    int N,
    int C,
    int H,
    int W,
    int direction, // src to dst direction, support L2S, S2S, L22S, L2L2, S2L2, L22L2
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: "
           "src_addr = 0x%llx, src_local_idx=%d, mask_addr = 0x%llx, mask_local_idx = %d, dst_addr = 0x%llx, src_format=%d, mask_format=%d, "
           "N=%d, C=%d, H=%d, W=%d, direction=%d\n",
           __func__,
           src_addr, src_local_idx, mask_addr, mask_local_idx, dst_addr,
           src_format, mask_format, N, C, H, W, direction);

    ASSERT(src_format != GDMA_FP20);
    ASSERT(mask_format != GDMA_FP20);

    int32_t special_func = 0;
    /// mode0: {N:16bit, C:16bit, H:16bit, W:16bit}
    /// mode1: {N:16bit, C:16bit, H:1,     W:32bit}
    if (W >= (1 << 16)) {
        special_func = 1;
        ASSERT_TENSOR_NSIZE(N);
        ASSERT_TENSOR_NSIZE(C);
        ASSERT(H == 1);
    } else {
        special_func = 0;
        ASSERT_TENSOR_SIZE(N, C, H, W);
    }
    ASSERT_FS(direction == GDMA_L2S ||
              direction == GDMA_S2S);

    int src_from_lmem = direction == GDMA_L2S;
    if (src_from_lmem) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
        ASSERT_FS(src_addr % ALIGN_BYTES == 0);
    }
    if (mask_in_lmem) {
        mask_addr = CALC_LOCAL_ADDR(mask_local_idx, mask_addr);
        ASSERT_FS(mask_addr % ALIGN_BYTES == 0);
    }

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format,  direction, pid_node
    GDMA_MASKED_SEL_PROFILE(N, C, H, W, src_addr, dst_addr, src_format, direction,  pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] =  gdma_get_cache_en() << 4 |
            ((u64)GDMA_FILTER << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41) |
            ((u64)mask_format << 45);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[3] = ((u64)N) | ((u64)C << 16) |
            ((u64)(W >= (1 << 16) ? (W >> 16) : H) << 32) |
            ((u64)(W & 0xffff) << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    low[5] = mask_addr & 0x1ffffffffffful;
    high[5] = 0xffffffffffffffff;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_FILTER,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_move_nonzero_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 dst_addr, // global addr only
    int src_format, // INT8/INT16/INT32/FP32/FP16/BF16
    int dst_format, //INT8/INT16/INT32
    int N,
    int C,
    int H,
    int W,
    u32 base_idx,
    int direction, // only support L2S, S2S
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: "
           "src_addr = 0x%llx, src_local_idx=%d, dst_addr = 0x%llx, src_format=%d, dst_format=%d, "
           "N=%d, C=%d, H=%d, W=%d, base_idx=%d, direction=%d, ",
           __func__,
           src_addr, src_local_idx, dst_addr,
           src_format, dst_format, N, C, H, W, base_idx, direction);

    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT_FS(direction == GDMA_L2S ||
              direction == GDMA_S2S);

    int src_from_lmem = direction == GDMA_L2S;
    if (src_from_lmem) {
        ASSERT(src_addr < LOCAL_MEM_SIZE);
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
        ASSERT_FS(src_addr % ALIGN_BYTES == 0);
    }

    ASSERT(src_format != GDMA_FP20);
    ASSERT(dst_format == GDMA_INT8 || dst_format == GDMA_INT16 || dst_format == GDMA_INT32);
    // src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, direction, pid_node
    GDMA_NONZERO_PROFILE(N, C, H, W, src_addr, dst_addr, src_format, dst_format, direction,  pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = gdma_get_cache_en() << 4 |
            ((u64)GDMA_NONZERO << 32) |
            ((u64)src_format << 41) |
            ((u64)dst_format << 45);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[2] = ((u64)base_idx);
    low[3] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_NONZERO,
      0, src_format | direction << 4, pid_node, high, low, 6);
}

unsigned int get_gdma_filter_res_num_gen_cmd(CMD_ID_NODE * pid_node) {
    unsigned int tmp = 0;
    poll_all_engine_done(pid_node);
    READ_GDMA_FILTER_RES_NUM(tmp, pid_node);

    return tmp;
}

void tensor_broadcast_move_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    int dst_lmem_start_addr, //local_addr
    int dst_local_idx,
    int src_N,
    int src_H,
    int src_W,
    int dst_C, // Restriction: dst_local_idx + dst_C <= NPU_NUM
    stride_type src_N_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_H_stride,
    int data_format,
    int stride_enable,
    int direction, // Only support, S2L, L2L, L22L
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr=0x%llx, src_local_idx=%d, dst_mem_start_addr = 0x%x, dst_local_idx=%d, "
        "src_N=%d, src_H=%d, src_W=%d, dst_C=%d, "
        "src_N_stride=%d, src_H_stride=%d, "
        "dst_N_stride=%d, dst_H_stride=%d, "
        "data_format=%d, stride_en=%d, direction=%d\n",
            __func__, src_addr, src_local_idx, dst_lmem_start_addr, dst_local_idx,
            src_N, src_H, src_W, dst_C,
            src_N_stride, src_H_stride,
            dst_N_stride, dst_H_stride,
            data_format, stride_enable, direction);


    ASSERT_TENSOR_SIZE(src_N, dst_C, src_H, src_W);
    ASSERT_FS_INFO(direction == GDMA_S2L || direction == GDMA_L2L, "directin=%d", direction);
    ASSERT(data_format != GDMA_FP20);
    ASSERT_FS_INFO(dst_C + dst_local_idx <= NPU_NUM,
                   "tensor broadcast dst_c + npu_idx <= NPU_NUM, dst_C + npu_idx:%d",
                   dst_C + dst_local_idx);

    int src_from_lmem = direction == GDMA_L2L;
    if (src_from_lmem) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    }
    u64 dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_lmem_start_addr);

    // src_n, src_h, src_w, dst_c, src_addr, dst_addr, data_format, direction, src_wstride, dst_wstride,
    // src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_BROADCAST_GET_PROFILE(src_N, src_H, src_W,  dst_C,  src_addr, dst_addr, data_format, direction,
                     1, 1, src_H_stride, dst_H_stride, 1, 1, src_N_stride, dst_N_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)GDMA_FUNC_BROADCAST << 37) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_N_stride);
    high[1] = ((u64)src_H_stride | (u64)1 << 32);
    low[2] = ((u64)dst_N_stride);
    high[2] = ((u64)dst_H_stride | (u64)1 << 32);
    low[3] = ((u64)src_N) |
            ((u64)1ul << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    high[3] = ((u64)src_N) |
            ((u64)dst_C << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = 0xffffffffffffffff;
    BEGIN_FAST_GEN_CMD_GDMA(MASTER_THREAD)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      GDMA_FUNC_BROADCAST, data_format | direction << 4, pid_node, high, low, 6);
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_gather_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 index_addr, // local_addr or global_addr
    int index_local_idx, // use only from local_mem
    int index_in_lmem, // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    u64 const_val,
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
    int direction, // Support S2S, S2L, L2S, L2L
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr=0x%llx, src_local_idx=%d, index_addr=0x%llx, index_local_idx=%d, index_in_lmem=%d, dst_addr=0x%llx, dst_local_idx=%d, "
        "C=%d, src_H=%d, src_W=%d, index_H=%d, start_pos=%d, "
        "src_C_stride=%d, src_H_stride=%d, index_C_stride=%d, index_H_stride=%d, "
        "dst_C_stride=%d, dst_H_stride=%d, "
        "src_format=%d, src_C_is1=%d, index_C_is1=%d, stride_en=%d, direction=%d\n",
            __func__, src_addr, src_local_idx, index_addr, index_local_idx,
            index_in_lmem, dst_addr, dst_local_idx,
            C, src_H, src_W, index_H, start_pos,
            src_C_stride, src_H_stride,
            index_C_stride, index_H_stride,
            dst_C_stride, dst_H_stride,
            src_format, src_C_is1, index_C_is1,
            stride_enable, direction);

    ASSERT_TENSOR_CSIZE(C);
    ASSERT_TENSOR_WSIZE(src_W);
    ASSERT(src_format != GDMA_FP20);

    if (stride_enable) ASSERT_FS(index_H_stride == 1);

    if (index_in_lmem) {
        index_addr = CALC_LOCAL_ADDR(index_local_idx, index_addr);
    }

    if (direction == GDMA_S2L) {
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    } else if (direction == GDMA_L2S) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    } else if (direction == GDMA_L2L) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    }

   // src_c, src_h, src_w, index_H, src_addr, dst_addr, src_data_format, direction,
   // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_GATHER_GET_PROFILE( C, src_H, src_W, index_H, src_addr, dst_addr, src_format, direction,
                       1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, index_H_stride, index_C_stride, stride_enable, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_GATHER << 32) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20) | ((u64)const_val << 32);
    low[1] = ((u64)src_C_stride) | ((u64)src_H_stride << 32);
    high[1] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
    low[2] = ((u64)index_C_stride) | ((u64)start_pos << 32);
    high[2] = ((u64)(src_C_is1 ? (1<<16) : (C<<16))) | ((u64)src_H << 32);
    low[3] = ((u64)src_W) | ((u64)index_H << 32);
    high[3] = ((u64)(index_C_is1 ? 1 : C) << 16) | ((u64)index_H << 32);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    low[5] = index_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_GATHER,
      0, src_format | direction << 4, pid_node, high, low, 6);
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_scatter_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 index_addr, // local_addr or global_addr
    int index_local_idx, // use only from local_mem
    int index_in_lmem, // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
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
    int direction, // Support S2S, S2L, L2S, L2L, S2L2, L22S, L2L2, L22L, L22L2
    int inplace_add,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr=0x%llx, src_local_idx=%d, index_addr=0x%llx, index_local_idx=%d, index_in_lmem=%d, dst_addr=0x%llx, dst_local_idx=%d, "
        "C=%d, src_H=%d, src_W=%d, dst_H=%d, start_pos=%d,"
        "src_C_stride=%d, src_H_stride=%d, index_C_stride=%d, index_H_stride=%d, "
        "dst_C_stride=%d, dst_H_stride=%d, "
        "src_format=%d, src_C_is1=%d, index_C_is1=%d, stride_en=%d, direction=%d, inplace_add=%d\n",
            __func__, src_addr, src_local_idx, index_addr, index_local_idx,
            index_in_lmem, dst_addr, dst_local_idx,
            C, src_H, src_W, dst_H, start_pos,
            src_C_stride, src_H_stride,
            index_C_stride, index_H_stride,
            dst_C_stride, dst_H_stride,
            src_format, src_C_is1, index_C_is1,
            stride_enable, direction, inplace_add);

    ASSERT_FS(C <= GDMA_MAX_C);
    ASSERT_FS(src_W <= GDMA_MAX_W);
    ASSERT(src_format != GDMA_FP20);
    if (stride_enable) ASSERT_FS(index_H_stride == 1);

    if (index_in_lmem) {
        index_addr = CALC_LOCAL_ADDR(index_local_idx, index_addr);
    }

    if (direction == GDMA_S2L) {
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    } else if (direction == GDMA_L2S) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    } else if (direction == GDMA_L2L) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    }
    if(src_format == GDMA_FP8_E4M3 || src_format == GDMA_FP8_E5M2) {
        inplace_add = 0;
    }

    //  src_c, src_h, src_w, dst_h, src_addr, dst_addr, src_data_format, direction, src_C_is1, index_C_is1, inplace_add,
    //src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_SCATTER_GET_PROFILE( C, src_H, src_W, dst_H, src_addr, dst_addr, src_format, direction, src_C_is1, index_C_is1,
                       1, 1, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, index_H_stride, index_C_stride, stride_enable, inplace_add, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = ((u64)stride_enable << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_SCATTER << 32) |
            ((u64)inplace_add << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_C_stride) | ((u64)src_H_stride << 32);
    high[1] = ((u64)dst_C_stride) | ((u64)dst_H_stride << 32);
    low[2] = ((u64)index_C_stride) | ((u64)start_pos << 32);
    high[2] = ((u64)(src_C_is1 ? (1<<16) : (C<<16))) | ((u64)src_H << 32);
    low[3] = ((u64)src_W) | ((u64)dst_H << 32);
    high[3] = ((u64)(index_C_is1 ? 1 : C) << 16) | ((u64)src_H << 32);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    low[5] = index_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_SCATTER,
      inplace_add, src_format | direction << 4, pid_node, high, low, 6);
}

// only support l2s
void tensor_normal_compress_gen_cmd(
    uint32_t local_mem_addr,
    u64      sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard, // only valid for fp16
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node)
{
    FW_DBG("%s: ,"
           "local_mem_addr:0x%x, sys_mem_addr:0x%llx, "
           "N:%d, C:%d, H:%d, W:%d, "
           "local_n_stride:%d, local_c_stride:%d, "
           "local_h_stride:%d, bias0:%d, bias1:%d, "
           "is_signed:%d, zero_guard:%d, data_format:%d\n",
           __func__, local_mem_addr, sys_mem_addr, N, C, H, W, local_n_stride,
           local_c_stride, local_h_stride, bias0, bias1, is_signed, zero_guard,
           data_format);

    u64 src_addr = CALC_LOCAL_ADDR(0, local_mem_addr);
    u64 dst_addr = sys_mem_addr;
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                   data_format == GDMA_INT16 ||
                   data_format == GDMA_FP16 ||
                   data_format == GDMA_BF16,
                   "format:%d", data_format);
    ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 &&
           zero_guard < 2);
    ASSERT(data_format == GDMA_FP16 ||
           data_format == GDMA_BF16 ||
           !is_signed || (bias0 < 128 && bias1 < 128));
    ASSERT(get_npu_index(local_mem_addr) < NPU_NUM);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_COMPRESS << 32) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)local_n_stride) |
            ((u64)local_c_stride << 32);
    high[1] = ((u64)local_h_stride);
    low[3] = ((u64)N) |
            ((u64)C << 16) |
            ((u64)H << 32) |
            ((u64)W << 48);
    high[3] = ((u64)bias0 << 32) |
            ((u64)bias1 << 40) |
            ((u64)is_signed << 48) |
            ((u64)(data_format == GDMA_FP16 ? zero_guard :
                    (data_format == GDMA_BF16 ? 1 : 0)) << 49);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = 0xffffffffffffffff;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_COMPRESS,
      0, data_format | GDMA_L2S << 4, pid_node, high, low, 6);
}

// only support s2l
void tensor_normal_decompress_gen_cmd(
    uint32_t local_mem_addr,
    u64      sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node)
{
    FW_DBG("%s: ,"
           "local_mem_addr:0x%x, sys_mem_addr:0x%llx, "
           "N:%d, C:%d, H:%d, W:%d, "
           "local_n_stride:%d, local_c_stride:%d, "
           "local_h_stride:%d, bias0:%d, bias:%d,"
           "is_signd:%d, zero_graud:%d, data_format:%d\n",
           __func__, local_mem_addr, sys_mem_addr, N, C, H, W, local_n_stride,
           local_c_stride, local_h_stride, bias0, bias1, is_signed, zero_guard,
           data_format);

    u64 dst_addr = CALC_LOCAL_ADDR(0, local_mem_addr);
    u64 src_addr = sys_mem_addr;
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                   data_format == GDMA_INT16 ||
                   data_format == GDMA_FP16 ||
                   data_format == GDMA_BF16,
                   "format:%d", data_format);
    ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 && zero_guard < 2);
    ASSERT(get_npu_index(local_mem_addr) < NPU_NUM);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_DECOMPRESS << 32) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    high[1] = ((u64)local_n_stride << 32);
    low[2] = ((u64)local_c_stride) |
            ((u64)local_h_stride << 32);
    low[3] = ((u64)N) |
            ((u64)C << 16) |
            ((u64)H << 32) |
            ((u64)W << 48);
    high[3] = ((u64)bias0 << 32) |
            ((u64)bias1 << 40) |
            ((u64)is_signed << 48) |
            ((u64)(data_format == GDMA_FP16 ? zero_guard :
                    (data_format == GDMA_BF16 ? 1 : 0)) << 49);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_DECOMPRESS,
      0, data_format | GDMA_S2L << 4, pid_node, high, low, 6);
}

// only support l2s
void tensor_racu_compress_gen_cmd(
    uint32_t local_mem_addr,
    u64      racu_sys_mem_addr,
    u64      meta_sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint32_t racu_n_stride,
    uint32_t racu_c_stride,
    uint32_t racu_h_stride,
    uint32_t meta_n_stride,
    uint32_t meta_c_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node)
{
    FW_DBG("%s: ,"
           "local_mem_addr:0x%x, racu_sys_mem_addr:0x%llx, "
           "meta_sys_mem_addr:0x%llx, "
           "N:%d, C:%d, H:%d, W:%d, "
           "local_n_stride:%d, local_c_stride:%d, "
           "local_h_stride:%d, racu_n_stride:%d, "
           "racu_c_stride:%d, racu_h_stride:%d, "
           "meta_n_stride:%d, meta_c_stride:%d, "
           "bias0:%d, bias1:%d, is_signed:%d, zero_guard:%d, data_format:%d\n",
           __func__, local_mem_addr, racu_sys_mem_addr, meta_sys_mem_addr, N, C,
           H, W, local_n_stride, local_c_stride, local_h_stride, racu_n_stride,
           racu_c_stride, racu_h_stride, meta_n_stride, meta_c_stride, bias0,
           bias1, is_signed, zero_guard, data_format);

    int type_len = get_gdma_format_type_len(data_format);
    ASSERT(get_npu_index(local_mem_addr) == 0);
    u64 src_laddr = CALC_LOCAL_ADDR(0, local_mem_addr);
    u64 racu_gaddr = racu_sys_mem_addr;
    u64 meta_gaddr = meta_sys_mem_addr;
    ASSERT_TENSOR_SIZE(N, C, H, W);
    // because max enc_size is 19bit
    ASSERT((W * sg_min(C, NPU_NUM) * type_len) <= (1 << (12 + NNVLC_ALIGN_SHIFT)));
    ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                   data_format == GDMA_INT16 ||
                   data_format == GDMA_FP16 ||
                   data_format == GDMA_BF16,
                   "format:%d", data_format);
    ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 && zero_guard < 2);
    ASSERT(data_format == GDMA_FP16 ||
           data_format == GDMA_BF16 ||
           !is_signed || (bias0 < 128 && bias1 < 128));
    ASSERT((racu_h_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
           (racu_c_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
           (racu_n_stride & (NNVLC_ALIGN_BYTES - 1)) == 0);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_COMPRESS << 32) |
            ((u64)1 << 37) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)local_n_stride) | ((u64)local_c_stride << 32);
    high[1] = ((u64)local_h_stride) | ((u64)(racu_n_stride / type_len) << 32);
    // hardware calculate racu offset useing racu_stride * type_len, but actual
    // racu is in byte unit, so here using stride / type_len as racu_stride
    low[2] = ((u64)(racu_c_stride / type_len)) | ((u64)(racu_h_stride / type_len) << 32);
    high[2] = ((u64)meta_n_stride) | ((u64)meta_c_stride << 32);
    low[3] = ((u64)N) |
            ((u64)C << 16) |
            ((u64)H << 32) |
            ((u64)W << 48);
    high[3] = ((u64)1) |
            ((u64)bias0 << 32) |
            ((u64)bias1 << 40) |
            ((u64)is_signed << 48) |
            ((u64)(data_format == GDMA_FP16 ? zero_guard :
                    (data_format == GDMA_BF16 ? 1 : 0)) << 49);
    low[4] = src_laddr & 0x1ffffffffffful;
    high[4] = racu_gaddr & 0x1ffffffffffful;
    low[5] = meta_gaddr & 0x1ffffffffffful;
    high[5] = 0xffffffffffffffff;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_COMPRESS,
      1, data_format | GDMA_L2S << 4, pid_node, high, low, 6);
}

// only support s2l
void tensor_racu_decompress_gen_cmd(
    uint32_t local_mem_addr,
    u64      racu_sys_mem_addr,
    u64      meta_sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint32_t racu_n_stride,
    uint32_t racu_c_stride,
    uint32_t racu_h_stride,
    uint32_t meta_n_stride,
    uint32_t meta_c_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node)
{
    FW_DBG("%s: ,"
           "local_mem_addr:0x%x, racu_sys_mem_addr:0x%llx, "
           "meta_sys_mem_addr:0x%llx, "
           "N:%d, C:%d, H:%d, W:%d, "
           "local_n_stride:%d, local_c_stride:%d, "
           "local_h_stride:%d, racu_n_stride:%d, "
           "racu_c_stride:%d, racu_h_stride:%d, "
           "meta_n_stride:%d, meta_c_stride:%d, "
           "bias0:%d, bias1:%d, is_signed:%d, zero_guard:%d, data_format:%d\n",
           __func__, local_mem_addr, racu_sys_mem_addr, meta_sys_mem_addr, N, C,
           H, W, local_n_stride, local_c_stride, local_h_stride, racu_n_stride,
           racu_c_stride, racu_h_stride, meta_n_stride, meta_c_stride, bias0,
           bias1, is_signed, zero_guard, data_format);

    int type_len = get_gdma_format_type_len(data_format);
    ASSERT(get_npu_index(local_mem_addr) == 0);
    u64 dst_laddr = CALC_LOCAL_ADDR(0, local_mem_addr);
    u64 racu_gaddr = racu_sys_mem_addr;
    u64 meta_gaddr = meta_sys_mem_addr;
    ASSERT_TENSOR_SIZE(N, C, H, W);
    // because max enc_size is 19bit
    ASSERT((W * sg_min(C, NPU_NUM) * type_len) <= (1 << (12 + NNVLC_ALIGN_SHIFT)));
    ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                   data_format == GDMA_INT16 ||
                   data_format == GDMA_FP16 ||
                   data_format == GDMA_BF16,
                   "format:%d", data_format);
    ASSERT(is_signed >= 0 && is_signed < 2);
    ASSERT(data_format == GDMA_FP16 ||
           data_format == GDMA_BF16 ||
           !is_signed || (bias0 < 128 && bias1 < 128));
    ASSERT((racu_h_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
           (racu_c_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
           (racu_n_stride & (NNVLC_ALIGN_BYTES - 1)) == 0);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_DECOMPRESS << 32) |
            ((u64)1 << 37) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    // hardware calculate racu offset useing racu_stride * type_len, but actual
    // racu is in byte unit, so here using stride / type_len as racu_stride
    low[1] = ((u64)(racu_n_stride / type_len)) | ((u64)(racu_c_stride / type_len) << 32);
    high[1] = ((u64)(racu_h_stride / type_len)) | ((u64)local_n_stride << 32);
    low[2] = ((u64)local_c_stride) | ((u64)local_h_stride << 32);
    high[2] = ((u64)meta_n_stride) | ((u64)meta_c_stride << 32);
    low[3] = ((u64)N) |
            ((u64)C << 16) |
            ((u64)H << 32) |
            ((u64)W << 48);
    high[3] = ((u64)1) |
            ((u64)bias0 << 32) |
            ((u64)bias1 << 40) |
            ((u64)is_signed << 48) |
            ((u64)(data_format == GDMA_FP16 ? zero_guard :
                    (data_format == GDMA_BF16 ? 1 : 0)) << 49);
    low[4] = racu_gaddr & 0x1ffffffffffful;
    high[4] = dst_laddr & 0x1ffffffffffful;
    low[5] = meta_gaddr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_DECOMPRESS,
      1, data_format | GDMA_S2L << 4, pid_node, high, low, 6);
}

void tensor_gdma_reverse_gen_cmd(
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
    int32_t direction,
    int thread_id,
    CMD_ID_NODE *pid_node)
{
    FW_DBG("%s, "
           "src_addr:0x%llx, dst_addr:0x%llx, N:%d, C:%d, "
           "H:%d, W:%d, src_n_stride:%d, src_c_stride:%d, "
           "src_h_stride:%d, dst_n_stride:%d, dst_c_stride:%d, "
           "dst_h_stride:%d, data_format::%d, reversed_axis:%d, direction:%d\n",
           __func__, src_addr, dst_addr, N, C, H, W, src_n_stride,
           src_c_stride, src_h_stride, dst_n_stride,
           dst_c_stride, dst_h_stride, data_format,
           reverse_axis, direction);

    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT(reverse_axis >= 0 && reverse_axis < 4);
    ASSERT(data_format != GDMA_FP20);

    if (SRC_IS_LOCAL(direction)) {
        src_addr = CALC_LOCAL_ADDR(0, src_addr);
        ASSERT(reverse_axis == 1);
    }
    if (DST_IS_LOCAL(direction)) {
        dst_addr = CALC_LOCAL_ADDR(0, dst_addr);
    }

    int32_t special_func = 3 - reverse_axis;
    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_REVERSE << 32) |
            ((u64)special_func << 37) |
            ((u64)data_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_n_stride) | ((u64)src_c_stride << 32);
    high[1] = ((u64)src_h_stride) | ((u64)1 << 32);
    low[2] = ((u64)dst_n_stride) | ((u64)dst_c_stride << 32);
    high[2] = ((u64)dst_h_stride) | ((u64)1 << 32);
    low[3] = ((u64)N) | ((u64)C << 16) | ((u64)H << 32) | ((u64)W << 48);
    high[3] = low[3];
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] =dst_addr & 0x1ffffffffffful;
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_REVERSE,
      special_func, data_format | direction << 4, pid_node, high, low, 6);
}

void gdma_lossy_compress_gen_cmd(
    u64 src_addr, // local_addr or sys
    int src_local_idx, // use only from local_mem
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // Support S2S, L2S
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_local_idx=%d, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "direction=%d\n",
        __func__, src_addr, src_local_idx, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr,  N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride, direction);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_ddr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(is_gmem(dst_addr) || is_l2mem(dst_addr),
                   "must be sys memory, dst_ddr:0x%llx", dst_addr);
#endif

    if (SRC_IS_LOCAL(direction)) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    }

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_LOSSY_COMPRESS << 32) |
              ((u64)GDMA_FUNC_NONE << 37);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_LOSSY_COMPRESS,
      GDMA_FUNC_NONE, 0 | direction << 4, pid_node, high, low, 6);
}

void gdma_lossy_decompress_gen_cmd(
    u64 src_addr, // sys
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // local_addr or sys
    int dst_local_idx, // use only from local_mem
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // S2S or S2L
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, dst_local_idx=%d, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "direction=%d\n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, dst_local_idx, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride, direction);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(is_gmem(src_addr) || is_l2mem(src_addr),
                   "must be sys memory, src_ddr:0x%llx", src_addr);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_ddr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif

    if (DST_IS_LOCAL(direction)) {
        dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    }

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_LOSSY_DECOMPRESS << 32) |
              ((u64)GDMA_FUNC_NONE << 37);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_LOSSY_DECOMPRESS,
      GDMA_FUNC_NONE, 0 | direction << 4, pid_node, high, low, 6);
}

void gdma_lossy_compress_reduce_gen_cmd(
    u64 src_addr, // local_addr, global_addr or l2_addr
    int src_local_idx, // use only from local_mem
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, //l2_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // Support S2S, L2S
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_local_idx=%d, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "direction=%d, reduce_psum_op=%d, reduce_opcode=%d\n",
        __func__, src_addr, src_local_idx, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride,
        direction, reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr),
                   "can't be static memory, src_ddr:0x%llx", src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif

    if (direction == GDMA_L2S) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    }
    u32 are_dtype = get_gdma_are_dtype(GDMA_FP20);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_LOSSY_COMPRESS << 32) |
              ((u64)GDMA_FUNC_NONE << 37);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
        low[5] = ((u64)are_dtype) |
              ((u64)reduce_opcode << 4) |
              ((u64)reduce_psum_op << 8) |
              ((u64)1 << 15);
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_LOSSY_COMPRESS,
      GDMA_FUNC_NONE, GDMA_FP20 | direction << 4, pid_node, high, low, 6);
}

void gdma_lossy_decompress_reduce_gen_cmd(
    u64 src_addr, // sys
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // local_addr or sys
    int dst_local_idx,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // only S2S
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, N=%d, C=%d, H=%d, W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, "
        "dst_addr = 0x%llx, dst_local_idx=%d, N=%d, C=%d, H=%d, W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, "
        "direction=%d, reduce_psum_op=%d, reduce_opcode=%d\n",
        __func__, src_addr, N, C, H, W,
        src_N_stride, src_C_stride, src_H_stride,
        dst_addr, dst_local_idx, N, C, H, W,
        dst_N_stride, dst_C_stride, dst_H_stride,
        direction, reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(is_gmem(src_addr) || is_l2mem(src_addr),
                   "must be sys memory, src_ddr:0x%llx", src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif

    u32 are_dtype = get_gdma_are_dtype(GDMA_FP32);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_LOSSY_DECOMPRESS << 32) |
              ((u64)GDMA_FUNC_NONE << 37);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
        low[5] = ((u64)are_dtype) |
              ((u64)reduce_opcode << 4) |
              ((u64)reduce_psum_op << 8) |
              ((u64)1 << 15);
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_LOSSY_DECOMPRESS,
      GDMA_FUNC_NONE, GDMA_FP32 | direction << 4, pid_node, high, low, 6);
}

// Only support GDMA write
// derived from tensor_stride_move_gen_cmd
void tensor_stride_move_reduce_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int src_N,
        int src_C,
        int src_H,
        int src_W,
        stride_type src_N_stride,
        stride_type src_C_stride,
        stride_type src_H_stride,
        stride_type src_W_stride,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int src_format,
        int direction,
        int transpose,  // N/C transpose
        int reduce_psum_op,
        int reduce_opcode,
        int thread_id,
        CMD_ID_NODE * pid_node) {
    FW_DBG("%s: local_mem_start_addr = 0x%x, local_mem_idx=%d, sys_mem_start_addr = 0x%llx, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d, "
        "src_format=%d, direction=%d, transpose=%d, "
        "reduce_psum_op=%d, reduce_opcode=%d\n",
            __func__, local_mem_start_addr, local_mem_idx, sys_mem_start_addr,
            src_N, src_C, src_H, src_W,
            src_N_stride, src_C_stride, src_H_stride, src_W_stride,
            dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
            src_format, direction, transpose,
            reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "directin=%d", direction);
    // int type_len = get_gdma_format_type_len(src_format);
    // ASSERT_WSTRIDE(src_W_stride, type_len);
    // ASSERT_WSTRIDE(dst_W_stride, type_len);
    ASSERT(src_format != GDMA_FP20);
    if(src_format == GDMA_INT32) {
        ASSERT(reduce_opcode != GDMA_ARE_MUL);
    }
#endif

    u64 sys_addr = sys_mem_start_addr;
    u64 local_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
    transpose = 0;
    int special_func = transpose ? GDMA_FUNC_TRANS: GDMA_FUNC_NONE;
    int is_local_to_sys = direction == GDMA_L2S;
    u64 src_addr = is_local_to_sys ? local_addr : sys_addr;
    u64 dst_addr = is_local_to_sys ? sys_addr : local_addr;
    u32 are_dtype = get_gdma_are_dtype(src_format);
    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable,  pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction, special_func, 3,
                      src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {0}, high[6] = {0};
    low[0] = (1ull << 1) |
            (1ull << 2) |
            gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
    low[1] = ((u64)src_N_stride) | ((u64)src_C_stride << 32);
    high[1] = ((u64)src_H_stride) | ((u64)src_W_stride << 32);
    low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
    high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
    low[3] = ((u64)src_N) |
            ((u64)src_C << 16) |
            ((u64)src_H << 32) |
            ((u64)src_W << 48);
    low[4] = src_addr & 0x1ffffffffffful;
    high[4] = (dst_addr & 0x1ffffffffffful);
    low[5] = ((u64)are_dtype) |
            ((u64)reduce_opcode << 4) |
            ((u64)reduce_psum_op << 8) |
            ((u64)1 << 15);
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void tensor_general_move_local_cross_core_gen_cmd(
    u64 src_addr, //local_addr or smem_addr
    int src_local_idx, //use only from local_mem
    int src_core_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr, //local_addr or smem_addr
    int dst_local_idx, //use only to local_mem
    int dst_core_idx,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_local_idx=%d, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx, dst_local_idx=%d, dst_core_idx=%d, "
        "dst_N=%d, dst_C=%d, dst_H=%d, dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d\n",
        __func__, src_addr, src_local_idx, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_local_idx, dst_core_idx, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    ASSERT(src_format != GDMA_FP20);
    ASSERT(0 <= src_local_idx && src_local_idx < NPU_NUM);
    ASSERT(0 <= dst_local_idx && dst_local_idx < NPU_NUM);
    ASSERT(0 <= dst_core_idx && dst_core_idx < MAX_TPU_CORE_NUM);
    ASSERT(0 <= src_core_idx && src_core_idx < MAX_TPU_CORE_NUM);
    ASSERT(src_core_idx != dst_core_idx);
    ASSERT(src_W_stride == 1);
    ASSERT(dst_W_stride == 1);
    int type_len = get_gdma_format_type_len(src_format);
    ASSERT_WSTRIDE(src_W_stride, type_len);
    ASSERT_WSTRIDE(dst_W_stride, type_len);
#endif
    // sg2260 deleted this instr
    ASSERT(0);
    src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr) + src_core_idx * CORE_OFFSET;
    dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr) + dst_core_idx * CORE_OFFSET;

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, GDMA_L2L, GDMA_FUNC_NONE, 3,
                            src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_TRANSFER << 32) |
              ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
        high[5] = gdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TRANSFER,
      0, src_format | GDMA_L2L << 4, pid_node, high, low, 6);
}

void atomic_transfer_general_gen_cmd(
    u64 src_addr, //local_addr or smem_addr
    int src_core_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr, //local_addr or smem_addr
    int dst_core_idx,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx, dst_core_idx=%d, "
        "dst_N=%d, dst_C=%d, dst_H=%d, dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d\n",
        __func__, src_addr, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_core_idx, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    ASSERT(src_format != GDMA_FP20);
    ASSERT(0 <= dst_core_idx && dst_core_idx < MAX_TPU_CORE_NUM);
    ASSERT(0 <= src_core_idx && src_core_idx < MAX_TPU_CORE_NUM);
    ASSERT(src_core_idx != dst_core_idx);
    ASSERT(src_W_stride == 1);
    ASSERT(dst_W_stride == 1);
    int type_len = get_gdma_format_type_len(src_format);
    ASSERT_WSTRIDE(src_W_stride, type_len);
    ASSERT_WSTRIDE(dst_W_stride, type_len);
    ASSERT(is_smem(src_addr) || is_lmem(src_addr));
    ASSERT(is_smem(dst_addr) || is_lmem(dst_addr));
#endif
    // sg2260 deleted this instr
    ASSERT(0);
    if(is_smem(src_addr)) {
        src_addr = CALC_STATIC_ADDR(src_addr) + src_core_idx * CORE_OFFSET;
    } else if (is_lmem(src_addr)) {
        src_addr = CALC_LOCAL_ADDR(0, src_addr) + src_core_idx * CORE_OFFSET;
    }

    if(is_smem(dst_addr)) {
        dst_addr = CALC_STATIC_ADDR(dst_addr) + dst_core_idx * CORE_OFFSET;
    } else if (is_lmem(dst_addr)) {
        dst_addr = CALC_LOCAL_ADDR(0, dst_addr) + dst_core_idx * CORE_OFFSET;
    }

    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, GDMA_L2L, GDMA_FUNC_NONE, 3,
                            src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
            gdma_get_cache_en() << 4 |
              ((u64)GDMA_TRANSFER << 32) |
              ((u64)src_format << 41);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
        high[5] = gdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TRANSFER,
      0, src_format | GDMA_L2L << 4, pid_node, high, low, 6);
}


// Only support GDMA write
// derived from tensor_general_move_gen_cmd
void tensor_general_move_reduce_gen_cmd(
    u64 src_addr, //local_addr or global_addr
    int src_local_idx, //use only from local_mem
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr, //local_addr or global_addr
    int dst_local_idx, //use only to local_mem
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int direction,
    int transpose,  // N/C transpose
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG(
        "%s: "
        "src_addr = 0x%llx, src_local_idx=%d,  src_N=%d, src_C=%d, src_H=%d, "
        "src_W=%d, "
        "src_N_stride=%d, src_C_stride=%d, src_H_stride=%d, src_W_stride=%d, "
        "src_format=%d, "
        "dst_addr = 0x%llx, dst_local_idx=%d,  dst_N=%d, dst_C=%d, dst_H=%d, "
        "dst_W=%d, "
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d, "
        "reduce_psum_op=%d, reduce_opcode=%d\n",
        __func__, src_addr, src_local_idx, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        dst_addr, dst_local_idx, dst_N, dst_C, dst_H, dst_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride, reduce_psum_op, reduce_opcode);

#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    if (src_format == GDMA_FP20) {
        ASSERT(direction == GDMA_S2S);
        ASSERT(transpose == 0);
        ASSERT(src_addr % 128 == 0);
        ASSERT(dst_addr % 128 == 0);
        ASSERT_WSTRIDE_FP20(src_W_stride);
        ASSERT_WSTRIDE_FP20(dst_W_stride);
        ASSERT_COMPACT_FP20((u32)src_N, (u32)src_C, (u32)src_H, (u32)src_W, src_N_stride, src_C_stride, src_H_stride, src_W_stride)
        ASSERT_COMPACT_FP20((u32)dst_N, (u32)dst_C, (u32)dst_H, (u32)dst_W, dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride)
    } else {
        // int type_len = get_gdma_format_type_len(src_format);
        // ASSERT_WSTRIDE(src_W_stride, type_len);
        // ASSERT_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    if(src_format == GDMA_INT32) {
        ASSERT(reduce_opcode != GDMA_ARE_MUL);
    }
    ASSERT(!DST_IS_LOCAL(direction));
#endif

    if (SRC_IS_LOCAL(direction)) {
        src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
    }
    // just for compatible with old param_list, after tv_gen test, we should assert it
    transpose = 0;
    int special_func = transpose ? GDMA_FUNC_TRANS : GDMA_FUNC_NONE;
    u32 are_dtype = get_gdma_are_dtype(src_format);
    // src_n, src_c, src_h, src_w,  src_addr, dst_addr, src_data_format, direction,  special_func, store_type,
    // src_wstride, dst_wstride, src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node
    GDMA_TENSOR_GET_PROFILE(src_N, src_C, src_H, src_W,  src_addr, dst_addr, src_format, direction, special_func, 3,
                      src_W_stride, dst_W_stride, src_H_stride, dst_H_stride, src_C_stride, dst_C_stride, src_N_stride, dst_N_stride, 1, pid_node);

    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    u64 low[6] = {6}, high[6] = {0};
    low[0] = (1ull << 1) |
        gdma_get_cache_en() << 4 |
            ((u64)GDMA_TENSOR << 32) |
            ((u64)special_func << 37) |
            ((u64)src_format << 41);
    high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
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
    low[5] = ((u64)are_dtype) |
            ((u64)reduce_opcode << 4) |
            ((u64)reduce_psum_op << 8) |
            ((u64)1<<15);
    high[5] = gdma_get_lane_mask();
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
    for (int i = 0; i < 6; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_TENSOR,
      special_func, src_format | direction << 4, pid_node, high, low, 6);
}

void random_mask_init_seed_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int dst_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    uint64_t size,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int src_format,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, dst_addr = 0x%llx, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, size=%lu"
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d, "
        "src_format=%d\n",
            __func__, src_addr, dst_addr,
            src_N, src_C, src_H, src_W, size,
            dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
            src_format);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(!is_smem(src_addr),
                   "can't be static memory sys_addr:0x%llx",
                   src_addr);
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT(src_format != GDMA_FP20);
    ASSERT(src_N == 1);
    ASSERT(dst_W_stride == 1);
#endif
    // sg2260 deleted this instr
    ASSERT(0);
    dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              (1ull << 2) |
              gdma_get_cache_en() << 4 |
              ((u64)GDMA_RANDOM_MASK << 32) |
              ((u64)src_format << 41) |
              ((u64)1 << 46); // init seed
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        low[1] = ((u64)size);
        high[1] = high[0];
        low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
        high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
        low[3] = ((u64)src_N) |
              ((u64)src_C << 16) |
              ((u64)src_H << 32) |
              ((u64)src_W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_RANDOM_MASK,
      0, src_format | GDMA_S2L << 4, pid_node, high, low, 6);
}

void random_mask_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int dst_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    uint64_t size,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int src_format,
    int inter_state,
    int thread_id,
    CMD_ID_NODE * pid_node) {
    FW_DBG("%s: src_addr = 0x%llx, dst_addr = 0x%llx, "
        "src_N=%d, src_C=%d, src_H=%d, src_W=%d, size=%lu"
        "dst_N_stride=%d, dst_C_stride=%d, dst_H_stride=%d, dst_W_stride=%d, "
        "src_format=%d, inter_state=%d\n",
            __func__, src_addr, dst_addr,
            src_N, src_C, src_H, src_W, size,
            dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride,
            src_format, inter_state);

#ifdef USING_CMODEL
    ASSERT_FS_INFO(!is_smem(src_addr),
                   "can't be static memory sys_addr:0x%llx",
                   src_addr);
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT(src_format != GDMA_FP20);
    ASSERT(src_N == 1);
    ASSERT(dst_W_stride == 1);
#endif
    // sg2260 deleted this instr
    ASSERT(0);
    dst_addr = CALC_LOCAL_ADDR(dst_local_idx, dst_addr);
    const volatile u64 reg_addr = GDMA_CMD_BASE_ADDR;
    BEGIN_FAST_GEN_CMD_GDMA(thread_id)
        u64 low[6] = {0}, high[6] = {0};
        low[0] = (1ull << 1) |
              (1ull << 2) |
              gdma_get_cache_en() << 4 |
              ((u64)GDMA_RANDOM_MASK << 32) |
              ((u64)src_format << 41) |
              ((u64)inter_state << 45);
        high[0] = ((u64)pid_node->bd_cmd_id & 0xfffff) | (1ull << 20);
        low[1] = ((u64)size);
        high[1] = high[0];
        low[2] = ((u64)dst_N_stride) | ((u64)dst_C_stride << 32);
        high[2] = ((u64)dst_H_stride) | ((u64)dst_W_stride << 32);
        low[3] = ((u64)src_N) |
              ((u64)src_C << 16) |
              ((u64)src_H << 32) |
              ((u64)src_W << 48);
        low[4] = src_addr & 0x1ffffffffffful;
        high[4] = dst_addr & 0x1ffffffffffful;
        high[5] = gdma_get_lane_mask();
        for (int i = 0; i < 6; ++i) {
            WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
        }
    END_FAST_GEN_CMD_GDMA(pid_node)
    profile_time_set_node(ENGINE_GDMA, GDMA_RANDOM_MASK,
      0, src_format | GDMA_S2L << 4, pid_node, high, low, 6);
}
