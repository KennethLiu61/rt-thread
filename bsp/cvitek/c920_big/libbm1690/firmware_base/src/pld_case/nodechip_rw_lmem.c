#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "base_def.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_rw_lmem_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us()
#define END(info, prec) \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s lmem %s\n", #info, #prec);            \
    printf("Total %s time: %lldus\n\n", #info, (end_time - start_time))

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time; (void)end_time;
    const u32 cnt = sizeof(u8) + sizeof(u16) + sizeof(u32) + sizeof(u64);
    const u32 size = 256;

    u8* data8 = malloc(cnt * size);
    for (u32 i = 0; i < cnt * size; i++) {
        data8[i] = i % 256;
    }
    // step 1: write lmem
    int offset = 0;
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           *(volatile u8*)GET_LOCAL_ADDR(ic, offset + i) =  data8[i];
        }
    }
    END(write, INT8);

    offset += sizeof(u8) * size;
    u16* data16 = (u16*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           *(volatile u16*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u16)) =  data16[i];
        }
    }
    END(write, INT16);

    offset += sizeof(u16) * size;
    u32* data32 = (u32*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           *(volatile u32*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u32)) =  data32[i];
        }
    }
    END(write, INT32);

    offset += sizeof(u32) * size;
    u64* data64 = (u64*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           *(volatile u64*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u64)) =  data64[i];
        }
    }
    END(write, INT64);

    // step 2: read lmem
    offset = 0;
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
            data8[i] = *(volatile u8*)GET_LOCAL_ADDR(ic, offset + i);
        }
    }
    END(read, INT8);

    offset += sizeof(u8) * size;
    data16 = (u16*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
            data16[i] = *(volatile u16*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u16));
        }
    }
    END(read, INT16);

    offset += sizeof(u16) * size;
    data32 = (u32*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           data32[i] = *(volatile u32*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u32));
        }
    }
    END(read, INT32);

    offset += sizeof(u32) * size;
    data64 = (u64*)(data8 + offset);
    BEGIN();
    for(int ic = 0; ic < NPU_NUM; ic++)
    {
        for (u32 i = 0; i < size; i++) {
           data64[i] = *(volatile u64*)GET_LOCAL_ADDR(ic, offset + i * sizeof(u64));
        }
    }
    END(read, INT64);

    // dim4 shape = {1, NPU_NUM, cnt, size};
    // tpu_gdma_cpy_L2S(output_addr, 0, &shape, NULL, NULL, DT_INT8);
    // tpu_poll();
    for(int ic = 0; ic < NPU_NUM; ic++) {
        memcpy(GET_GLOBAL_ADDR(output_addr + ic * cnt * size),
            GET_LOCAL_ADDR(ic, 0),
            cnt * size);
    }
    tpu_flush_cache(output_addr, NPU_NUM * cnt * size);

#define WRITE_LMEM(times, byte, info) \
    printf("write %dbyte for %d times with memcpy\n", byte, times); \
    start_time = firmware_timer_get_time_us(); \
    for (u32 i = 0; i < times; i++) { \
        memcpy(((u8*)GET_LOCAL_ADDR(0, 0)), data8, byte); \
    } \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s L2mem\n", #info);            \
    printf("Total %s time: %lldus, bw=%.3fMB/s\n\n", #info, (end_time - start_time), (float)byte*(float)times/(float)((float)(end_time-start_time) * 1e-6)/(float)1024/1024)

    // test write with memcpy
    const u32 times = 100;
    WRITE_LMEM(times, 64, write_with_memcpy);
    WRITE_LMEM(times, 256, write_with_memcpy);
    WRITE_LMEM(times, 512, write_with_memcpy);
    WRITE_LMEM(times, 1024, write_with_memcpy);
 

    // test read with memcpy
#define READ_LMEM(times, byte, info) \
    printf("read %dbyte for %d times with memcpy\n", byte, times); \
    start_time = firmware_timer_get_time_us(); \
    for (u32 i = 0; i < times; i++) { \
        memcpy(data8, ((u8*)GET_LOCAL_ADDR(1,0)), byte); \
    } \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s smem\n", #info);            \
    printf("Total %s time: %lldus, bw=%.3fMB/s\n\n", #info, (end_time - start_time),  (float)byte* (float)times/(float)((float)(end_time-start_time) * 1e-6)/(float)1024/1024)

    READ_LMEM(times, 64, read_with_memcpy);
    READ_LMEM(times, 256, read_with_memcpy);
    READ_LMEM(times, 512, read_with_memcpy);
    READ_LMEM(times, 1024, read_with_memcpy);
    printf("%d end test rw local mem\n", data8[0]);

    free(data8);
#undef BEGIN
#undef END
}
