#ifndef __FIRMWARE_TOP_H__
#define __FIRMWARE_TOP_H__

#include <stdint.h>

/* gp timer */
uint64_t timer_frequency(void);

/* tick */
uint64_t timer_get_tick(void);

/* delay */
void timer_delay_tick(uint64_t tick);
void timer_mdelay(uint32_t ms);
void timer_udelay(uint32_t us);

/* fast method to real tiem */
uint64_t timer_tick2us(uint64_t tick);
uint64_t timer_tick2ms(uint64_t tick);
uint64_t timer_tick2s(uint64_t tick);

#define mdelay(__ms)	timer_mdelay(__ms)
#define udelay(__us)	timer_udelay(__us)

/* 128bit type, stdint.h style */
typedef __uint128_t uint128_t;
typedef __int128_t int128_t;

/* load or store 128bit data to device or memory */
/* gcc built-in 128bit integer is an alternative solution, see __uint128_t */
static inline void read128(uint64_t _addr, uint64_t _l64, uint64_t _h64) {
#if defined(__riscv)
    asm volatile ("ldp %0, %1, [%2]"
             : "=r"(_l64), "=r"(_h64)
             : "r"(_addr));
#else
    __uint128_t *val = (__uint128_t *)(_addr);
    *(uint64_t*)_h64 = *val >> 64;
    *(uint64_t*)_l64 = (uint64_t)*val;
#endif
}

static inline void write128(uint64_t _addr, uint64_t _l64, uint64_t _h64) {
#if defined(__riscv)
    asm volatile ("sdd %0, %1, (%2), 0, 4"
              :
              : "r"(_l64), "r"(_h64), "r"(_addr));
#else
    __uint128_t val = _h64;
    val = val << 64 | _l64;
    *(__uint128_t *)(_addr) = val;
#endif
}
static inline void write32(uint64_t _addr, uint32_t value) {
#if defined(__riscv)
    asm volatile ("sw %0, (%1)"
              :
              : "r"(value), "r"(_addr));
#else
    *(uint32_t *)(_addr) = value;
#endif
}
/* output log to pcie host on pcie mode, to application processor on soc mode */
void fw_log(char *fmt, ...);

enum {
	WORK_MODE_PCIE,
	WORK_MODE_SOC,
};

int get_work_mode(void);

#endif
