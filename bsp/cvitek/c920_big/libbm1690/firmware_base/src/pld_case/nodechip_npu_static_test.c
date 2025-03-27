#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"
#define MAX_ITER (10)
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
void nodechip_npu_static_test() {
    tpu_initialize();
    dim4 shape = {.n = 4, .c = 32, .h = 32, .w = 64};
    float cpy_len = shape.n * shape.c * shape.h * shape.w;
    dim4 shape_bcast_src = {.n = shape.n, .c = 1, .h = shape.h, .w = shape.w};
    dim4 stride, stride_bcast_src, stride_cross_src;
    tpu_aligned_stride(&stride, 0, &shape, DT_FP32);
    tpu_aligned_stride(&stride_bcast_src, 0, &shape_bcast_src, DT_FP32);
    local_addr_t dst_addr = 0;
    local_addr_t bcast_src_addr = ALIGN(dst_addr + stride.n * shape.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    TPUKERNEL_ASSERT(bcast_src_addr + stride_bcast_src.n * shape_bcast_src.n * 4 <= (unsigned int)LOCAL_MEM_SIZE);
    int start_idx = 32;
    tpu_aligned_stride(&stride_cross_src, start_idx, &shape, DT_FP32);
    local_addr_t cross_src_addr = start_idx * LOCAL_MEM_SIZE + ALIGN(dst_addr + stride.n * shape.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
    TPUKERNEL_ASSERT(cross_src_addr % LOCAL_MEM_SIZE + stride_cross_src.n * shape.n * 4 <= (unsigned int)LOCAL_MEM_SIZE);
    u64 st = 0, et = 0;
    for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_npu_bcast(
                dst_addr,
                bcast_src_addr,
                &shape,
                (data_type_t)dtypes[d]);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "N=%d "
               "C=%d "
               "H=%d "
               "W=%d "
               "dtype=%s "
               "loops=%d "
               "elapsed time=%lldus "
               "Tops=%.6f"
               "\n",
               "tpu_bdc_npu_bcast",
               shape.n,
               shape.c,
               shape.h,
               shape.w,
               dtypes_str[d],
               MAX_ITER,
               et - st,
               cpy_len / ((et - st) / (float)MAX_ITER * 1e6));
    }
    for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_cpy_cross_npu(
                dst_addr,
                cross_src_addr,
                &shape,
                (data_type_t)dtypes[d]);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "N=%d "
               "C=%d "
               "H=%d "
               "W=%d "
               "dtype=%s "
               "loops=%d "
               "elapsed time=%lldus "
               "Tops=%.6f"
               "\n",
               "tpu_bdc_cpy_cross_npu",
               shape.n,
               shape.c,
               shape.h,
               shape.w,
               dtypes_str[d],
               MAX_ITER,
               et - st,
               cpy_len / ((et - st) / (float)MAX_ITER * 1e6));
    }
    int len = 256;
    for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_npu_bcast_from_static(
                dst_addr,
                0,
                NPU_NUM,
                len,
                (data_type_t)dtypes[d]);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "npu_num=%d "
               "len=%d "
               "dtype=%s "
               "loops=%d "
               "elapsed time=%lldus "
               "Tops=%.6f"
               "\n",
               "tpu_bdc_npu_bcast_from_static",
               NPU_NUM,
               len,
               dtypes_str[d],
               MAX_ITER,
               et - st,
               (len * NPU_NUM) / ((et - st) / (float)MAX_ITER * 1e6));
    }
    for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
        st = firmware_timer_get_time_us();
        for (int maxit = 0; maxit < MAX_ITER; ++maxit)
            tpu_bdc_npu_distribute_from_static(
                dst_addr,
                0,
                len,
                (data_type_t)dtypes[d]);
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("%s: "
               "len=%d "
               "dtype=%s "
               "loops=%d "
               "elapsed time=%lldus "
               "Tops=%.6f"
               "\n",
               "tpu_bdc_npu_distribute_from_static",
               len,
               dtypes_str[d],
               MAX_ITER,
               et - st,
               len / ((et - st) / (float)MAX_ITER * 1e6));
    }
    tpu_poll();
}
