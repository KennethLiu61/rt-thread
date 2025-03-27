
#include "nodechip_pld_test.h"
#include "tpu_kernel.h"
#include "firmware_timer.h"

char *get_dtype_name_v1(data_type_t dtype)
{
    char *res = malloc(sizeof(*res));
    if (dtype == DT_FP16)
        res = "DT_FP16";
    else if (dtype == DT_FP32)
        res = "DT_FP32";
    else if (dtype == DT_INT8)
        res = "DT_INT8";
    else if (dtype == DT_INT16)
        res = "DT_INT16";
    else if (dtype == DT_INT32)
        res = "DT_INT32";
    else if (dtype == DT_UINT8)
        res = "DT_UINT8";
    else if (dtype == DT_UINT16)
        res = "DT_UINT16";
    else if (dtype == DT_BFP16)
        res = "DT_BFP16";
    else if (dtype == DT_UINT32)
        res = "DT_UINT32";
    else if (dtype == DT_FP8E4M3)
        res = "DT_FP8E4M3";
    else if (dtype == DT_FP8E5M2)
        res = "DT_FP8E5M2";
    else
    {
        printf("unknown dtype");
        TPUKERNEL_ASSERT(false);
    }
    return res;
}

//   bw = loops * (float)tensor_size * sizeof(float) / (float)((s2s_end_time - s2s_start_time) * 1e-6);
//   printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

void nodechip_random_mask_perf_test(
    global_addr_t r_out_addr,
    global_addr_t m_out_addr)
{
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

    int format_v[] = {
        DT_FP32,
        DT_FP16,
        DT_BFP16,
        DT_INT16,
        DT_INT32,
        DT_FP8E4M3,
        DT_FP8E5M2,
    };

    dim4 mask_shape = {
        .n = 1,
        .c = 64,
        .h = 128,
        .w = 128};
    
    float bw = 0;
    dim4 mask_stride;
    local_addr_t dst_addr = 6 * bank_size;
    int size = mask_shape.n * mask_shape.c * mask_shape.h * mask_shape.w;
    unsigned long long st, med_0, med_1,et;

    tpu_initialize();
    for (int i = 0; i < 7; i++)
    {
        printf("\n================= start random mask perf test =================\n");
        tpu_gdma_random_mask_set_seed(i * 0x12756);
        int epoch = 0;
        tpu_aligned_stride(&mask_stride, 0, &mask_shape, format_v[i]);
        st = firmware_timer_get_time_us();
        while (epoch < 10)
        {
            tpu_gdma_random_mask_init_seed_S2L(r_out_addr, dst_addr, &mask_shape, size, &mask_stride, format_v[i]);
            epoch ++;
        }
        tpu_poll();
        med_0 = firmware_timer_get_time_us();
        bw = 10 * (float)size * sizeof(float) / (float)((med_0 - st) * 1e-6);
        printf("\n==========tpu_gdma_random_mask_init_seed_S2L case=========\n");
        printf("--- dtype: %s, loop times: %d, total time: %llu us, avg time: %llu us Average bandwidth : %.3fGB/s\n", get_dtype_name_v1(format_v[i]), i, med_0 - st, (med_0 - st) / 10, bw / 1024 / 1024 / 1024.f);
        epoch = 0;
        while (epoch < 10)
        {
            tpu_gdma_random_mask_S2L(m_out_addr, dst_addr, &mask_shape, size, &mask_stride, 0, format_v[i]);
            epoch ++;
        }
        tpu_poll();
        med_1 = firmware_timer_get_time_us();
        bw = 10 * (float)size * sizeof(float) / (float)((med_1 - med_0) * 1e-6);
        printf("\n==========tpu_gdma_random_mask_S2L use_iter_state=0 case=========\n");
        printf("--- dtype: %s, loop times: %d, total time: %llu us, avg time: %llu us Average bandwidth : %.3fGB/s\n", get_dtype_name_v1(format_v[i]), i, med_1 - med_0, (med_1 - med_0) / 10, bw / 1024 / 1024 / 1024.f);
        epoch = 0;
        while (epoch < 10)
        {
            tpu_gdma_random_mask_S2L(m_out_addr, dst_addr, &mask_shape, size, &mask_stride, 1, format_v[i]);
            epoch ++;
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        bw = 10 * (float)size * sizeof(float) / (float)((et - med_1) * 1e-6);
        printf("\n==========tpu_gdma_random_mask_S2L use_iter_state=1 case=========\n");
        printf("--- dtype: %s, loop times: %d, total time: %llu us, avg time: %llu us Average bandwidth : %.3fGB/s\n", get_dtype_name_v1(format_v[i]), i, et - med_1, (et - med_1) / 10, bw / 1024 / 1024 / 1024.f);
    }
    printf("\n================= end start random mask perf test =================\n");
}