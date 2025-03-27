#include "nodechip_pld_test.h"
#include "firmware_common.h"
#include "firmware_timer.h"

#define U8_PTR(p) ((u8*)(p))
#define U16_PTR(p) ((u16*)(p))
#define U32_PTR(p) ((u32*)(p))
#define U64_PTR(p) ((u64*)(p))
#define SIZE 1024

void nodechip_rw_ddr(
        unsigned long long input_addr,
        unsigned long long output_addr) {
    // ------------------------------------------------------
    // test function
    // ------------------------------------------------------
    // step 1: read from input
    invalidate_cache(input_addr, SIZE * 4);
    u8 data8[SIZE / sizeof(u8)];
    for (u32 i = 0; i < SIZE / sizeof(u8); i++) {
        data8[i] = *U8_PTR(GET_GLOBAL_ADDR(input_addr + i));
    }
    u16 data16[SIZE / sizeof(u16)];
    for (u32 i = 0; i < SIZE / sizeof(u16); i++) {
        data16[i] = *U16_PTR(GET_GLOBAL_ADDR(input_addr + SIZE + i * sizeof(u16)));
    }
    u32 data32[SIZE / sizeof(u32)];
    for (u32 i = 0; i < SIZE / sizeof(u32); i++) {
        data32[i] = *U32_PTR(GET_GLOBAL_ADDR(input_addr + SIZE * 2 + i * sizeof(u32)));
    }
    u64 data64[SIZE / sizeof(u64)];
    for (u32 i = 0; i < SIZE / sizeof(u64); i++) {
        data64[i] = *U64_PTR(GET_GLOBAL_ADDR(input_addr + SIZE * 3 + i * sizeof(u64)));
    }
    // step 2: write to output
    for (u32 i = 0; i < SIZE / sizeof(u8); i++) {
        *U8_PTR(GET_GLOBAL_ADDR(output_addr + i)) = data8[i];
    }
    for (u32 i = 0; i < SIZE / sizeof(u16); i++) {
        *U16_PTR(GET_GLOBAL_ADDR(output_addr + SIZE + i * sizeof(u16))) = data16[i];
    }
    for (u32 i = 0; i < SIZE / sizeof(u32); i++) {
        *U32_PTR(GET_GLOBAL_ADDR(output_addr + SIZE * 2 + i * sizeof(u32))) = data32[i];
    }
    for (u32 i = 0; i < SIZE / sizeof(u64); i++) {
        *U64_PTR(GET_GLOBAL_ADDR(output_addr + SIZE * 3 + i * sizeof(u64))) = data64[i];
    }
    flush_cache(output_addr, SIZE * 4);
    // ------------------------------------------------------
    // test performance
    // ------------------------------------------------------
    int data[1024];
    int loop = 1000;
    u64 st, et;
    // only read 1 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        invalidate_cache(input_addr, 64);
        data[0] = *U32_PTR(GET_GLOBAL_ADDR(input_addr));
    }
    et = firmware_timer_get_time_us();
    printf("read ddr 4bytes for %d loops using time %lldus\n", loop, et - st);
    // only read 1 number, but invalidate_cache 64*64 bytes
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        invalidate_cache(input_addr, 64 * 64);
        data[0] = *U32_PTR(GET_GLOBAL_ADDR(input_addr));
    }
    et = firmware_timer_get_time_us();
    printf("read ddr 4bytes but invalidate 64*64 bytes for %d loops using time %lldus\n", loop, et - st);
    // only read 64 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        invalidate_cache(input_addr, 64 * sizeof(int));
        memcpy(data, GET_GLOBAL_ADDR(input_addr), 64 * sizeof(int));
    }
    et = firmware_timer_get_time_us();
    printf("read ddr 64*4bytes for %d loops using time %lldus bw %fMB/s\n", loop, et - st, loop * 64 * 4.0 * 1e6 / (float)(et - st) / (float)(1024 * 1024));
    // only read 1024 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        invalidate_cache(input_addr, 1024 * sizeof(int));
        memcpy(data, GET_GLOBAL_ADDR(input_addr), 1024 * sizeof(int));
    }
    et = firmware_timer_get_time_us();
    printf("read ddr 1024*4bytes for %d loops using time %lldus bw %fMB/s\n", loop, et - st, loop * 1024 * 4.0 * 1e6 / (float)(et - st) / (float)(1024 * 1024));
    // only write 1 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        *U32_PTR(GET_GLOBAL_ADDR(input_addr)) = data[0];
        flush_cache(input_addr, 64);
    }
    et = firmware_timer_get_time_us();
    printf("write ddr 4bytes for %d loops using time %lldus\n", loop, et - st);
    // only write 1 number, but flush 64*64 bytes
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        *U32_PTR(GET_GLOBAL_ADDR(input_addr)) = data[0];
        flush_cache(input_addr, 64 * 64);
    }
    et = firmware_timer_get_time_us();
    printf("write ddr 4bytes but flush 64*64 bytes for %d loops using time %lldus\n", loop, et - st);
    // only read 64 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        memcpy(GET_GLOBAL_ADDR(input_addr), data, 64 * sizeof(int));
        flush_cache(input_addr, 64 * sizeof(int));
    }
    et = firmware_timer_get_time_us();
    printf("write ddr 64*4bytes for %d loops using time %lldus bw %fMB/s\n", loop, et - st, loop * 64 * 4.0 * 1e6 / (float)(et - st) / (float)(1024 * 1024));
    // only read 1024 number
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        memcpy(GET_GLOBAL_ADDR(input_addr), data, 1024 * sizeof(int));
        flush_cache(input_addr, 1024 * sizeof(int));
    }
    et = firmware_timer_get_time_us();
    printf("write ddr 1024*4bytes for %d loops using time %lldus bw %fMB\n", loop, et - st, loop * 1024 * 4.0 * 1e6 / (float)(et - st) / (float)(1024 * 1024));
    // memset 1024*4 bytes
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        memset(GET_GLOBAL_ADDR(input_addr), U8_PTR(data)[0], 4 * SIZE);
        flush_cache(input_addr, 4 * SIZE);
    }
    et = firmware_timer_get_time_us();
    printf("memset ddr 1024*4 bytes for %d loops using time %lldus\n", loop, et - st);
    // memcpy 1024*4 bytes
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        invalidate_cache(output_addr, 4 * SIZE);
        memcpy(GET_GLOBAL_ADDR(input_addr), GET_GLOBAL_ADDR(output_addr), 4 * SIZE);
        flush_cache(input_addr, 4 * SIZE);
    }
    et = firmware_timer_get_time_us();
    printf("memcpy ddr 1024*4 bytes for %d loops using time %lldus\n", loop, et - st);
}
