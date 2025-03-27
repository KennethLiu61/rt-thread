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
void nodechip_scatter_test(global_addr_t index_global) {
    tpu_initialize();
    dim4 stride;
    u64 st = 0, et = 0;
    dim4 full_shape = {.n = 1, .c = 32, .h = 128, .w = 256};
    {
        dim4 output_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        dim4 param_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        dim4 index_shape = {.n = 1, .c = 2048, .h = 1, .w = 1};
        local_addr_t output_addr = 0;
        tpu_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        local_addr_t param_addr = ALIGN(output_addr + output_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        local_addr_t index_addr = ALIGN(param_addr + param_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_compact_stride(&stride, 0, &index_shape);
        ASSERT(index_addr + index_shape.n * stride.n * 2 <= (unsigned int)LOCAL_MEM_SIZE);
        // LOAD INDEX FROM GLOBAL MEMORY
        tpu_gdma_cpy_S2L(
            0,
            index_global,
            &full_shape,
            NULL,
            NULL,
            DT_FP32);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_w_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT8);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_w_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT8",
                   MAX_ITER,
                   et - st);
        }
        // INDEX MASK
        scalar_t C = {.u16 = 2047};
        tpu_compact_stride(&stride, 0, &index_shape);
        tpu_bdc_and_C(
            index_addr,
            index_addr,
            C,
            &index_shape,
            &stride,
            &stride,
            DT_UINT16);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_w_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT16);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_w_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT16",
                   MAX_ITER,
                   et - st);
        }
    }
    /*
    {
        dim4 output_shape = {.n = 4, .c = 32, .h = 64, .w = 32};
        dim4 param_shape = {.n = 4, .c = 32, .h = 64, .w = 32};
        dim4 index_shape = {.n = 1, .c = 32 * 32 * 2, .h = 1, .w = 1};
        local_addr_t output_addr = 0;
        tpu_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        local_addr_t param_addr = ALIGN(output_addr + output_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        local_addr_t index_addr = ALIGN(param_addr + param_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_compact_stride(&stride, 0, &index_shape);
        ASSERT(index_addr + index_shape.n * stride.n * 2 <= (unsigned int)LOCAL_MEM_SIZE);
        // LOAD INDEX FROM GLOBAL MEMORY
        tpu_gdma_cpy_S2L(
            0,
            index_global,
            &full_shape,
            NULL,
            NULL,
            DT_FP32);
        // INDEX MASK
        scalar_t C = {.u16 = 31};
        tpu_compact_stride(&stride, 0, &index_shape);
        tpu_bdc_and_C(
            index_addr,
            index_addr,
            C,
            &index_shape,
            &stride,
            &stride,
            DT_UINT16);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_hw_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.w,
                    param_shape.h,
                    (data_type_t)dtypes[d]);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OH=%d "
                   "OW=%d "
                   "PHW=%d "
                   "dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_hw_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.h,
                   output_shape.w,
                   param_shape.h * param_shape.w,
                   dtypes_str[d],
                   MAX_ITER,
                   et - st);
        }
    } */
    {
        dim4 output_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        dim4 param_shape = {.n = 1, .c = 32, .h = 1, .w = 2048};
        dim4 index_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        local_addr_t output_addr = 0;
        tpu_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        local_addr_t param_addr = ALIGN(output_addr + output_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        local_addr_t index_addr = ALIGN(param_addr + param_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &index_shape, DT_UINT16);
        ASSERT(index_addr + index_shape.n * stride.n * 2 <= (unsigned int)LOCAL_MEM_SIZE);
        // LOAD INDEX FROM GLOBAL MEMORY
        tpu_gdma_cpy_S2L(
            0,
            index_global,
            &full_shape,
            NULL,
            NULL,
            DT_FP32);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_batch_bcast_w_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT8,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_batch_bcast_w_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT8",
                   MAX_ITER,
                   et - st);
        }
        // INDEX MASK
        scalar_t C = {.u16 = 2047};
        tpu_bdc_and_C(
            index_addr,
            index_addr,
            C,
            &index_shape,
            NULL,
            NULL,
            DT_UINT16);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_batch_bcast_w_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT16,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_batch_bcast_w_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT16",
                   MAX_ITER,
                   et - st);
        }
    }
    {
        dim4 output_shape = {.n = 4, .c = 32, .h = 256, .w = 4};
        dim4 param_shape = {.n = 1, .c = 32, .h = 256, .w = 4};
        dim4 index_shape = {.n = 4, .c = 32, .h = 256, .w = 1};
        local_addr_t output_addr = 0;
        tpu_line_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        local_addr_t param_addr = ALIGN(output_addr + output_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_line_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        local_addr_t index_addr = ALIGN(param_addr + param_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &index_shape, DT_UINT16);
        ASSERT(index_addr + index_shape.n * stride.n * 2 <= (unsigned int)LOCAL_MEM_SIZE);
        // LOAD INDEX FROM GLOBAL MEMORY
        tpu_gdma_cpy_S2L(
            0,
            index_global,
            &full_shape,
            NULL,
            NULL,
            DT_FP32);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_batch_bcast_h_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.h,
                    (data_type_t)dtypes[d],
                    DT_UINT8,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OH=%d "
                   "OW=%d "
                   "PH=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_batch_bcast_h_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.h,
                   output_shape.w,
                   param_shape.h,
                   dtypes_str[d],
                   "DT_UINT8",
                   MAX_ITER,
                   et - st);
        }
        // INDEX MASK
        scalar_t C = {.u16 = 255};
        tpu_bdc_and_C(
            index_addr,
            index_addr,
            C,
            &index_shape,
            NULL,
            NULL,
            DT_UINT16);
        tpu_poll();
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_batch_bcast_h_scatter(
                    output_addr,
                    param_addr,
                    index_addr,
                    &output_shape,
                    param_shape.h,
                    (data_type_t)dtypes[d],
                    DT_UINT16,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "N=%d "
                   "C=%d "
                   "OH=%d "
                   "OW=%d "
                   "PH=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_batch_bcast_h_scatter",
                   output_shape.n,
                   output_shape.c,
                   output_shape.h,
                   output_shape.w,
                   param_shape.h,
                   dtypes_str[d],
                   "DT_UINT16",
                   MAX_ITER,
                   et - st);
        }
    }
}
