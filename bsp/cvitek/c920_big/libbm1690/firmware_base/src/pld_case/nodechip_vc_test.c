#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"
#define SIGN(dtype) ((dtype) & 0x1)
#define PRECISION(dtype) ((dtype) >> 1)
#define WIDTH(dtype) tpu_data_type_size(dtype)
#define MAX_ITER (10)
typedef void(*vc_binary_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    int,
    int,
    int,
    int,
    data_type_t);
typedef vc_binary_func_t fp_vc_binary_func_t;
typedef void(*vc_cmp_binary_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    scalar_t,
    int,
    int,
    int,
    int,
    data_type_t,
    data_type_t);
typedef void(*int_vc_binary_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    int,
    int,
    int,
    int,
    data_type_t,
    data_type_t,
    data_type_t,
    bool);
static const vc_binary_func_t vc_binary_funcs[] = {
    tpu_bdc_vc_and,
    tpu_bdc_vc_or,
    tpu_bdc_vc_xor,
    tpu_bdc_vc_min,
    tpu_bdc_vc_max
};
static const char * const vc_binary_func_str[] = {
    "tpu_bdc_vc_and",
    "tpu_bdc_vc_or",
    "tpu_bdc_vc_xor",
    "tpu_bdc_vc_min",
    "tpu_bdc_vc_max"
};
static const fp_vc_binary_func_t fp_vc_binary_funcs[] = {
    tpu_bdc_fp_vc_add,
    tpu_bdc_fp_vc_sub,
    tpu_bdc_fp_vc_mul
};
static const char * const fp_vc_binary_func_str[] = {
    "tpu_bdc_fp_vc_add",
    "tpu_bdc_fp_vc_sub",
    "tpu_bdc_fp_vc_mul"
};
static const vc_cmp_binary_func_t vc_cmp_binary_funcs[] = {
    tpu_bdc_vc_greater,
    tpu_bdc_vc_less,
    tpu_bdc_vc_equal
};
static const char * const vc_cmp_binary_func_str[] = {
    "tpu_bdc_vc_greater",
    "tpu_bdc_vc_less",
    "tpu_bdc_vc_equal"
};
static const int_vc_binary_func_t int_vc_binary_funcs[] = {
    tpu_bdc_int_vc_add,
    tpu_bdc_int_vc_sub,
    tpu_bdc_int_vc_mul
};
static const char * const int_vc_binary_func_str[] = {
    "tpu_bdc_int_vc_add",
    "tpu_bdc_int_vc_sub",
    "tpu_bdc_int_vc_mul"
};
static const int dtypes[] = {
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_FP16,
    DT_BFP16,
    DT_INT32,
    DT_UINT32,
    DT_FP32
};
static const char * const dtypes_str[] = {
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_FP16",
    "DT_BFP16",
    "DT_INT32",
    "DT_UINT32",
    "DT_FP32"
};
static const int fp_dtypes[] = {
    DT_FP16,
    DT_BFP16,
    DT_FP32
};
static const char * const fp_dtypes_str[] = {
    "DT_FP16",
    "DT_BFP16",
    "DT_FP32"
};
static const int int_dtypes[] = {
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_INT32,
    DT_UINT32
};
static const char * const int_dtypes_str[] = {
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_INT32",
    "DT_UINT32"
};
void nodechip_vc_test() {
    tpu_initialize();
    int src0_len = 64;
    int src1_len = 64 * 64;
    int src0_len_per_channel = 64;
    int src1_len_per_channel = 64;
    dim4 shape_src0 = {.n = 1, .c = 1, .h = 1, .w = 64};
    dim4 shape_src1 = {.n = 1, .c = 64, .h = 1, .w = 64};
    dim4 shape_dst = {.n = 64, .c = 64, .h = 1, .w = 64};
    dim4 stride_dst, stride_src0, stride_src1;
    tpu_aligned_stride(&stride_dst, 0, &shape_dst, DT_FP32);
    tpu_aligned_stride(&stride_src0, 0, &shape_src0, DT_FP32);
    tpu_aligned_stride(&stride_src1, 0, &shape_src1, DT_FP32);
    local_addr_t dst_addr = 0;
    local_addr_t src0_addr = ALIGN(dst_addr + stride_dst.n * shape_dst.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    local_addr_t src1_addr = ALIGN(src0_addr + stride_src0.n * shape_src0.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    TPUKERNEL_ASSERT(src1_addr + stride_src1.n * shape_src1.n * 4 <= (unsigned int)LOCAL_MEM_SIZE);
    u64 st = 0, et = 0;
    for (unsigned int i = 0; i < sizeof(vc_binary_funcs) / sizeof(vc_binary_funcs[0]); ++i) {
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                vc_binary_funcs[i](
                    dst_addr,
                    src0_addr,
                    src1_addr,
                    src0_len,
                    src1_len,
                    src0_len_per_channel,
                    src1_len_per_channel,
                    (data_type_t)dtypes[d]);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "src0_len=%d "
                   "src1_len=%d "
                   "src0_len_per_channel=%d "
                   "src1_len_per_channel=%d "
                   "dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   vc_binary_func_str[i],
                   src0_len,
                   src1_len,
                   src0_len_per_channel,
                   src1_len_per_channel,
                   dtypes_str[d],
                   MAX_ITER,
                   et - st);
        }
    }
    for (unsigned int i = 0; i < sizeof(vc_cmp_binary_funcs) / sizeof(vc_cmp_binary_funcs[0]); ++i) {
        scalar_t true_val = {.u32 = 0};
        for (unsigned int d0 = 0; d0 < sizeof(dtypes) / sizeof(dtypes[0]); ++d0) {
            for (unsigned int d1 = 0; d1 < sizeof(dtypes) / sizeof(dtypes[0]); ++d1) {
                if (WIDTH(dtypes[d0]) < WIDTH(dtypes[d1]))
                    continue;
                st = firmware_timer_get_time_us();
                for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                    vc_cmp_binary_funcs[i](
                        dst_addr,
                        src0_addr,
                        src1_addr,
                        true_val,
                        src0_len,
                        src1_len,
                        src0_len_per_channel,
                        src1_len_per_channel,
                        (data_type_t)dtypes[d1],
                        (data_type_t)dtypes[d0]);
                tpu_poll();
                et = firmware_timer_get_time_us();
                printf("%s: "
                       "src0_len=%d "
                       "src1_len=%d "
                       "src0_len_per_channel=%d "
                       "src1_len_per_channel=%d "
                       "dst_dtype=%s "
                       "src_dtype=%s "
                       "loops=%d "
                       "elapsed time=%lldus"
                       "\n",
                       vc_cmp_binary_func_str[i],
                       src0_len,
                       src1_len,
                       src0_len_per_channel,
                       src1_len_per_channel,
                       dtypes_str[d1],
                       dtypes_str[d0],
                       MAX_ITER,
                       et - st);
            }
        }
    }
    for (unsigned int i = 0; i < sizeof(fp_vc_binary_funcs) / sizeof(fp_vc_binary_funcs[0]); ++i) {
        for (unsigned int d = 0; d < sizeof(fp_dtypes) / sizeof(fp_dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                fp_vc_binary_funcs[i](
                    dst_addr,
                    src0_addr,
                    src1_addr,
                    src0_len,
                    src1_len,
                    src0_len_per_channel,
                    src1_len_per_channel,
                    (data_type_t)fp_dtypes[d]);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "src0_len=%d "
                   "src1_len=%d "
                   "src0_len_per_channel=%d "
                   "src1_len_per_channel=%d "
                   "dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   fp_vc_binary_func_str[i],
                   src0_len,
                   src1_len,
                   src0_len_per_channel,
                   src1_len_per_channel,
                   fp_dtypes_str[d],
                   MAX_ITER,
                   et - st);
        }
    }
    for (unsigned int i = 0; i < sizeof(int_vc_binary_funcs) / sizeof(int_vc_binary_funcs[0]); ++i) {
        for (unsigned int d0 = 0; d0 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d0) {
            for (unsigned int d1 = 0; d1 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d1) {
                for (unsigned int d2 = 0; d2 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d2) {
                    if (int_vc_binary_funcs[i] == tpu_bdc_int_vc_sub) {
                        if (!SIGN(int_dtypes[d2]))
                            continue;
                    } else {
                        if ((SIGN(int_dtypes[d0]) | SIGN(int_dtypes[d1])) != SIGN(int_dtypes[d2]))
                            continue;
                    }
                    st = firmware_timer_get_time_us();
                    for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                        int_vc_binary_funcs[i](
                            dst_addr,
                            src0_addr,
                            src1_addr,
                            src0_len,
                            src1_len,
                            src0_len_per_channel,
                            src1_len_per_channel,
                            (data_type_t)int_dtypes[d2],
                            (data_type_t)int_dtypes[d0],
                            (data_type_t)int_dtypes[d1],
                            false);
                    tpu_poll();
                    et = firmware_timer_get_time_us();
                    printf("%s: "
                           "src0_len=%d "
                           "src1_len=%d "
                           "src0_len_per_channel=%d "
                           "src1_len_per_channel=%d "
                           "dst_dtype=%s "
                           "src0_dtype=%s "
                           "src1_dtype=%s "
                           "saturation=%d "
                           "loops=%d "
                           "elapsed time=%lldus"
                           "\n",
                           int_vc_binary_func_str[i],
                           src0_len,
                           src1_len,
                           src0_len_per_channel,
                           src1_len_per_channel,
                           int_dtypes_str[d2],
                           int_dtypes_str[d0],
                           int_dtypes_str[d1],
                           0,
                           MAX_ITER,
                           et - st);
                    st = firmware_timer_get_time_us();
                    for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                        int_vc_binary_funcs[i](
                            dst_addr,
                            src0_addr,
                            src1_addr,
                            src0_len,
                            src1_len,
                            src0_len_per_channel,
                            src1_len_per_channel,
                            (data_type_t)int_dtypes[d2],
                            (data_type_t)int_dtypes[d0],
                            (data_type_t)int_dtypes[d1],
                            true);
                    tpu_poll();
                    et = firmware_timer_get_time_us();
                    printf("%s: "
                           "src0_len=%d "
                           "src1_len=%d "
                           "src0_len_per_channel=%d "
                           "src1_len_per_channel=%d "
                           "dst_dtype=%s "
                           "src0_dtype=%s "
                           "src1_dtype=%s "
                           "saturation=%d "
                           "loops=%d "
                           "elapsed time=%lldus"
                           "\n",
                           int_vc_binary_func_str[i],
                           src0_len,
                           src1_len,
                           src0_len_per_channel,
                           src1_len_per_channel,
                           int_dtypes_str[d2],
                           int_dtypes_str[d0],
                           int_dtypes_str[d1],
                           1,
                           MAX_ITER,
                           et - st);
                }
            }
        }
    }
    st = firmware_timer_get_time_us();
    for (int maxit = 0; maxit < MAX_ITER; ++maxit)
        tpu_bdc_fp32_vc_div(
            dst_addr,
            src0_addr,
            src1_addr,
            src0_len,
            src1_len,
            src0_len_per_channel,
            src1_len_per_channel);
    tpu_poll();
    et = firmware_timer_get_time_us();
    printf("%s: "
           "src0_len=%d "
           "src1_len=%d "
           "src0_len_per_channel=%d "
           "src1_len_per_channel=%d "
           "loops=%d "
           "elapsed time=%lldus"
           "\n",
           "tpu_bdc_fp32_vc_div",
           src0_len,
           src1_len,
           src0_len_per_channel,
           src1_len_per_channel,
           MAX_ITER,
           et - st);
}

