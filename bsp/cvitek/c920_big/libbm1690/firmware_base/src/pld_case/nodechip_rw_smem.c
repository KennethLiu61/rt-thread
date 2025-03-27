#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "atomic_gdma_gen_cmd.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_rw_smem_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us()
#define END(info, prec) \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s smem %s\n", #info, #prec);            \
    printf("Total %s time: %lldus\n\n", #info, (end_time - start_time))

    tpu_initialize();
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time; (void)end_time;
    const u32 cnt = sizeof(u8) + sizeof(u16) + sizeof(u32) + sizeof(u64);
    const u32 size = 1024;

    // step 1: write smem
    u8* data8 = malloc(cnt * size);
    for (u32 i = 0; i < cnt * size; i++) {
        data8[i] = i % 256;
    }
    u64 addr8 = STATIC_MEM_START_ADDR;
    BEGIN();
    for (u32 i = 0; i < size; i++) {
        *((u8*)GET_SMEM_ADDR(addr8 + i)) = data8[i];
    }
    END(write, INT8);

    u64 addr16 = addr8 + size;
    u16* data16 = (u16*)(data8 + size);
    BEGIN();
    for (u32 i = 0; i < size; i++) {
        *((u16*)GET_SMEM_ADDR(addr16 + i * sizeof(u16))) = data16[i];
    }
    END(write, INT16);

    u64 addr32 = addr8 + (sizeof(u8) + sizeof(u16)) * size;
    u32* data32 = (u32*)(data8 + (sizeof(u8) + sizeof(u16)) * size);
    BEGIN();
    for (u32 i = 0; i < size; i++) {
        *((u32*)GET_SMEM_ADDR(addr32 + i * sizeof(u32))) = data32[i];
    }
    END(write, INT32);

    u64 addr64 = addr8 + (sizeof(u8) + sizeof(u16) + sizeof(u32)) * size;
    u64* data64 = (u64*)(data8 + (sizeof(u8) + sizeof(u16) + sizeof(u32)) * size);
    BEGIN();
    for (u32 i = 0; i < size; i++) {
        *((u64*)GET_SMEM_ADDR(addr64 + i * sizeof(u64))) = data64[i];
    }
    END(write, INT64);


#define WRITE_SMEM(times, byte, info) \
    printf("write %dbyte for %d times with memcpy\n", byte, times); \
    start_time = firmware_timer_get_time_us(); \
    for (u32 i = 0; i < times; i++) { \
        memcpy(((u8*)GET_SMEM_ADDR(addr8)), data8, byte); \
    } \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s smem\n", #info);            \
    printf("Total %s time: %lldus, bw=%.3fMB/s\n\n", #info, (end_time - start_time),   \
        (float)byte*(float)times/(float)((float)(end_time-start_time) * 1e-6)/(float)1024/1024)

    // test write with memcpy
    const u32 times = 32;
    WRITE_SMEM(times, 64, write_with_memcpy);
    WRITE_SMEM(times, 256, write_with_memcpy);
    WRITE_SMEM(times, 512, write_with_memcpy);
    WRITE_SMEM(times, 1024, write_with_memcpy);


    // step 2: read smem
    BEGIN();
    for (u32 i = 0; i < size; i++) {
        data8[i] = *((u8*)GET_SMEM_ADDR(addr8 + i));
    }
    END(read, INT8);

    BEGIN();
    for (u32 i = 0; i < size; i++) {
        data16[i] = *((u16*)GET_SMEM_ADDR(addr16 + i * sizeof(u16)));
    }
    END(read, INT16);

    BEGIN();
    for (u32 i = 0; i < size; i++) {
        data32[i] = *((u32*)GET_SMEM_ADDR(addr32 + i * sizeof(u32)));
    }
    END(read, INT32);

    BEGIN();
    for (u32 i = 0; i < size; i++) {
        data64[i] = *((u64*)GET_SMEM_ADDR(addr64 + i * sizeof(u64)));
    }
    END(read, INT64);


    // test read with memcpy
#define READ_SMEM(times, byte, info) \
    printf("read %dbyte for %d times with memcpy\n", byte, times); \
    start_time = firmware_timer_get_time_us(); \
    for (u32 i = 0; i < times; i++) { \
        memcpy(data8, ((u8*)GET_SMEM_ADDR(addr8)), byte); \
    } \
    end_time = firmware_timer_get_time_us();    \
    printf("test %s smem\n", #info);            \
    printf("Total %s time: %lldus, bw=%.3fMB/s\n\n", #info, (end_time - start_time),  \
        (float)byte* (float)times/(float)((float)(end_time-start_time) * 1e-6)/(float)1024/1024)

    READ_SMEM(times, 64, read_with_memcpy);
    READ_SMEM(times, 256, read_with_memcpy);
    READ_SMEM(times, 512, read_with_memcpy);
    READ_SMEM(times, 1024, read_with_memcpy);

    u64 addr_list[15] = {0};
    for (int i = 0; i < 15; i++) {
        if (i == 0) addr_list[0] = STATIC_MEM_START_ADDR;
        else addr_list[i] = addr_list[i - 1] + 1024;
    }
    BEGIN();
    for (u32 i = 0; i < 1024; i++) {
        for (u32 j = 0; j < 15; j++) {
            data8[j] = *((u8*)GET_SMEM_ADDR(addr_list[j] + i));
        }
    }
    END(read_1024byte_15_per_byte, INT8); printf("avg time: %lldus\n", (end_time - start_time) / 15);

    // CMD_ID_NODE pid_node;
    // tpu_get_id_node(&pid_node);
    // general_gdma_gen_cmd(STATIC_MEM_START_ADDR, output_addr, 0, cnt * size, &pid_node);
    // tpu_set_id_node(&pid_node);
    // tpu_poll();
    memcpy(GET_GLOBAL_ADDR(output_addr), data8, cnt * size);
    flush_cache(output_addr, ALIGN(cnt * size, 64));

    free(data8);
#undef BEGIN
#undef END
#undef READ_SMEM
#undef WRITE_SMEM
}
