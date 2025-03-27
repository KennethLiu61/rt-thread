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
static const int mask_dtypes[] = {
    DT_UINT8,
    DT_UINT16,
    DT_UINT32
};
static const char * const mask_dtypes_str[] = {
    "DT_UINT8",
    "DT_UINT16",
    "DT_UINT32"
};
void nodechip_sg_other_test(global_addr_t index_global) {
    tpu_initialize();
    dim4 stride;
    u64 st = 0, et = 0;
   // scalar_t E = {.u32 = 0};
    dim4 full_shape = {.n = 1, .c = 32, .h = 128, .w = 256};
    {
        dim4 output_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        dim4 param_shape = {.n = 1, .c = 32, .h = 1, .w = 2048};
        dim4 mask_shape = {.n = 4, .c = 32, .h = 1, .w = 2048};
        dim4 count_shape = {.n = 4, .c = 32, .h = 1, .w = 1};
        local_addr_t output_addr = 0;
        tpu_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        local_addr_t count_addr = ALIGN(output_addr + output_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_compact_stride(&stride, 0, &count_shape);
        local_addr_t param_addr = ALIGN(count_addr + count_shape.n * stride.n * 2, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        local_addr_t mask_addr = ALIGN(param_addr + param_shape.n * stride.n * 4, LOCAL_MEM_SIZE / LOCAL_MEM_BANKS);
        tpu_aligned_stride(&stride, 0, &mask_shape, DT_FP32);
        ASSERT(mask_addr + mask_shape.n * stride.n * 4 <= (unsigned int)LOCAL_MEM_SIZE);
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
            for (unsigned int md = 0; md < sizeof(mask_dtypes) / sizeof(mask_dtypes[0]); ++md) {
                st = firmware_timer_get_time_us();
                for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                    tpu_bdc_batch_bcast_w_mask_select(
                        output_addr,
                        count_addr,
                        param_addr,
                        mask_addr,
                        &output_shape,
                        (data_type_t)dtypes[d],
                        (data_type_t)mask_dtypes[md],
                        true);
                tpu_poll();
                et = firmware_timer_get_time_us();
                printf("%s: "
                       "N=%d "
                       "C=%d "
                       "W=%d "
                       "dtype=%s "
                       "index_dtype=%s "
                       "loops=%d "
                       "elapsed time=%lldus"
                       "\n",
                       "tpu_bdc_batch_bcast_w_mask_select(no conflicting)",
                       output_shape.n,
                       output_shape.c,
                       output_shape.w,
                       dtypes_str[d],
                       mask_dtypes_str[md],
                       MAX_ITER,
                       et - st);
            }
        }
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            for (unsigned int md = 0; md < sizeof(mask_dtypes) / sizeof(mask_dtypes[0]); ++md) {
                st = firmware_timer_get_time_us();
                for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                    tpu_bdc_batch_bcast_w_mask_select(
                        output_addr,
                        count_addr,
                        output_addr,
                        mask_addr,
                        &output_shape,
                        (data_type_t)dtypes[d],
                        (data_type_t)mask_dtypes[md],
                        true);
                tpu_poll();
                et = firmware_timer_get_time_us();
                printf("%s: "
                       "N=%d "
                       "C=%d "
                       "W=%d "
                       "dtype=%s "
                       "index_dtype=%s "
                       "loops=%d "
                       "elapsed time=%lldus"
                       "\n",
                       "tpu_bdc_batch_bcast_w_mask_select(conflicting)",
                       output_shape.n,
                       output_shape.c,
                       output_shape.w,
                       dtypes_str[d],
                       mask_dtypes_str[md],
                       MAX_ITER,
                       et - st);
            }
        }
    }
    /* // remove in A2
    {
        dim4 output_shape = {.n = 1, .c = 64, .h = 1, .w = 4096};
        dim4 param_shape = {.n = 1, .c = 64, .h = 1, .w = 4096};
        dim4 index_shape = {.n = 1, .c = 64, .h = 1, .w = 4096};
        local_addr_t output_addr = LOCAL_MEM_SIZE / 8 * 2;
        tpu_aligned_stride(&stride, 0, &output_shape, DT_FP32);
        ASSERT(output_shape.n * stride.n * 4 <= LOCAL_MEM_SIZE / 8 * 2);
        local_addr_t index_addr = LOCAL_MEM_SIZE / 8 * 0;
        tpu_aligned_stride(&stride, 0, &index_shape, DT_UINT16);
        ASSERT(index_shape.n * stride.n * 2 <= LOCAL_MEM_SIZE / 8 * 2);
        local_addr_t param_addr = LOCAL_MEM_SIZE / 8 * 4;
        tpu_aligned_stride(&stride, 0, &param_shape, DT_FP32);
        ASSERT(param_shape.n * stride.n * 4 <= LOCAL_MEM_SIZE / 8);
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
                tpu_bdc_4bank_w_gather(
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
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_4bank_w_gather",
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT8",
                   MAX_ITER,
                   et - st);
        }
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_4bank_w_gather_exception(
                    output_addr,
                    param_addr,
                    index_addr,
                    E,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT8,
                    true,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_4bank_w_gather_exception",
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT8",
                   MAX_ITER,
                   et - st);
        }
        // INDEX MASK
        scalar_t C = {.u16 = 4095};
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
                tpu_bdc_4bank_w_gather(
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
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_4bank_w_gather",
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT16",
                   MAX_ITER,
                   et - st);
        }
        for (unsigned int d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); ++d) {
            st = firmware_timer_get_time_us();
            for (int maxit = 0; maxit < MAX_ITER; ++maxit)
                tpu_bdc_4bank_w_gather_exception(
                    output_addr,
                    param_addr,
                    index_addr,
                    E,
                    &output_shape,
                    param_shape.w,
                    (data_type_t)dtypes[d],
                    DT_UINT16,
                    true,
                    true);
            tpu_poll();
            et = firmware_timer_get_time_us();
            printf("%s: "
                   "C=%d "
                   "OW=%d "
                   "PW=%d "
                   "dtype=%s "
                   "index_dtype=%s "
                   "loops=%d "
                   "elapsed time=%lldus"
                   "\n",
                   "tpu_bdc_4bank_w_gather_exception",
                   output_shape.c,
                   output_shape.w,
                   param_shape.w,
                   dtypes_str[d],
                   "DT_UINT16",
                   MAX_ITER,
                   et - st);
        }
    }
    */
    tpu_poll();
}
