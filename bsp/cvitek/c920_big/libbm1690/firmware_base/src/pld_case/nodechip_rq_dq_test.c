#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"
#define MAX_ITER (10)
static const int high_dtypes[] = {
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_INT32,
};
static const char * const high_dtypes_str[] = {
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_INT32",
};
static const int low_dtypes[] = {
    DT_INT4,
    DT_UINT4,
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16
};
static const char * const low_dtypes_str[] = {
    "DT_INT4",
    "DT_UINT4",
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16"
};
void nodechip_rq_dq_test() {
    tpu_initialize();
    dim4 shape = {.n = 4, .c = 32, .h = 32, .w = 64};
    dim4 quant_shape = {.n = 1, .c = shape.c, .h = 1, .w = 2};
    dim4 quant_single_shape = {.n = 1, .c = shape.c, .h = 1, .w = 1};
    dim4 stride, quant_stride;
    tpu_aligned_stride(&stride, 0, &shape, DT_FP32);
    tpu_aligned_stride(&quant_stride, 0, &quant_shape, DT_FP32);
    local_addr_t dst_addr = 0;
    local_addr_t src_addr = ALIGN(dst_addr + stride.n * shape.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    local_addr_t quant_addr = ALIGN(src_addr + stride.n * shape.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    TPUKERNEL_ASSERT(quant_addr + quant_stride.n * quant_shape.n * 4 <= (unsigned int)LOCAL_MEM_SIZE);
    scalar_t multiplier = {.s32 = 1};
    scalar_t shift = {.s8 = -1};
    scalar_t offset = {.u32 = 0};
    scalar_t scale = {.f32 = 1.f};
    u64 st = 0, et = 0;
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_int_requant(
                    dst_addr,
                    src_addr,
                    &shape,
                    multiplier.s32,
                    shift.s8,
                    offset,
                    (data_type_t)low_dtypes[d1],
                    (data_type_t)high_dtypes[d0],
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_int_requant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   low_dtypes_str[d1],
                   high_dtypes_str[d0],
                   MAX_ITER,
                   et - st);
        }
    }
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_int_dequant(
                    dst_addr,
                    src_addr,
                    &shape,
                    offset,
                    multiplier.s32,
                    shift.s8,
                    (data_type_t)high_dtypes[d0],
                    (data_type_t)low_dtypes[d1],
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_int_dequant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   high_dtypes_str[d0],
                   low_dtypes_str[d1],
                   MAX_ITER,
                   et - st);
        }
    }
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_fp32_requant(
                    dst_addr,
                    src_addr,
                    &shape,
                    scale.f32,
                    offset.f32,
                    (data_type_t)low_dtypes[d1],
                    (data_type_t)high_dtypes[d0],
                    RM_HALF_AWAY_FROM_ZERO,
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_fp32_requant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   low_dtypes_str[d1],
                   high_dtypes_str[d0],
                   MAX_ITER,
                   et - st);
        }
    }
    for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_fp32_dequant(
                dst_addr,
                src_addr,
                &shape,
                offset,
                scale.f32,
                (data_type_t)low_dtypes[d1],
                RM_HALF_AWAY_FROM_ZERO);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "N=%d "
               "C=%d "
               "H=%d "
               "W=%d "
               "src_dtype=%s "
               "loops=%d "
               "elapsed time=%lldus"
               "\n",
               "tpu_bdc_fp32_dequant",
               shape.n,
               shape.c,
               shape.h,
               shape.w,
               low_dtypes_str[d1],
               MAX_ITER,
               et - st);
    }
    tpu_bdc_set_C(
        quant_addr,
        multiplier,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_bdc_set_C(
        quant_addr + 4,
        shift,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_bdc_set_C(
        quant_addr + 8,
        offset,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_poll();
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_int_pc_requant(
                    dst_addr,
                    src_addr,
                    quant_addr,
                    &shape,
                    (data_type_t)low_dtypes[d1],
                    (data_type_t)high_dtypes[d0],
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_int_pc_requant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   low_dtypes_str[d1],
                   high_dtypes_str[d0],
                   MAX_ITER,
                   et - st);
        }
    }
    tpu_bdc_set_C(
        quant_addr,
        offset,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_bdc_set_C(
        quant_addr + 4,
        multiplier,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_bdc_set_C(
        quant_addr + 8,
        shift,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_poll();
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_int_pc_dequant(
                    dst_addr,
                    src_addr,
                    quant_addr,
                    &shape,
                    (data_type_t)high_dtypes[d0],
                    (data_type_t)low_dtypes[d1],
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_int_pc_dequant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   high_dtypes_str[d0],
                   low_dtypes_str[d1],
                   MAX_ITER,
                   et - st);
        }
    }
    tpu_bdc_set_C(
        quant_addr,
        scale,
        &quant_single_shape,
        &quant_stride,
        DT_FP32);
    tpu_bdc_set_C(
        quant_addr + 4,
        offset,
        &quant_single_shape,
        &quant_stride,
        DT_FP32);
    tpu_poll();
    for (unsigned int d0 = 0; d0 < sizeof(high_dtypes) / sizeof(high_dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_fp32_pc_requant(
                    dst_addr,
                    src_addr,
                    quant_addr,
                    &shape,
                    (data_type_t)low_dtypes[d1],
                    (data_type_t)high_dtypes[d0],
                    RM_HALF_AWAY_FROM_ZERO,
                    RM_HALF_AWAY_FROM_ZERO);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "H=%d "
                   "W=%d "
                   "dst_dtype=%s "
                   "src_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_fp32_pc_requant",
                   shape.n,
                   shape.c,
                   shape.h,
                   shape.w,
                   low_dtypes_str[d1],
                   high_dtypes_str[d0],
                   MAX_ITER,
                   et - st);
        }
    }
    tpu_bdc_set_C(
        quant_addr,
        offset,
        &quant_single_shape,
        &quant_stride,
        DT_INT32);
    tpu_bdc_set_C(
        quant_addr + 4,
        scale,
        &quant_single_shape,
        &quant_stride,
        DT_FP32);
    tpu_poll();
    for (unsigned int d1 = 0; d1 < sizeof(low_dtypes) / sizeof(low_dtypes[0]); ++d1) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_fp32_pc_dequant(
                dst_addr,
                src_addr,
                quant_addr,
                &shape,
                (data_type_t)low_dtypes[d1],
                RM_HALF_AWAY_FROM_ZERO);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "N=%d "
               "C=%d "
               "H=%d "
               "W=%d "
               "src_dtype=%s "
               "loops=%d "
               "elapsed time=%lldus"
               "\n",
               "tpu_bdc_fp32_pc_dequant",
               shape.n,
               shape.c,
               shape.h,
               shape.w,
               low_dtypes_str[d1],
               MAX_ITER,
               et - st);
    }
}
