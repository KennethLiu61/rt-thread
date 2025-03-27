#include "firmware_timer.h"
#include "firmware_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(USING_EDA) && !defined(USING_PLD_TEST)

#include <time.h>
#define TIMER_FREQUENCY_ (1)
#define TIMER_PERIOD_NS_ (1)
u64 firmware_timer_get_cycle() {
    struct timespec now;

    //clock_gettime(CLOCK_REALTIME, &now);
    clock_gettime(CLOCK_MONOTONIC, &now);

    return now.tv_sec * 1000 * 1000 * 1000 + now.tv_nsec;
}
u64 firmware_timer_get_time_ms() { return (firmware_timer_get_cycle() / 1e6); }
u64 firmware_timer_get_time_us() { return (firmware_timer_get_cycle() / 1e3); }

#else

#include "firmware_top.h"
#define TIMER_FREQUENCY_ (1)//(timer_frequency())
#define TIMER_PERIOD_NS_ (1e9 / TIMER_FREQUENCY_)
u64 firmware_timer_get_cycle(void) {
    unsigned long n;

    asm volatile("rdtime %0" : "=r"(n));
    return n * 20;
}
u64 firmware_timer_get_time_ms() { return (firmware_timer_get_cycle() * 1e3 / TIMER_FREQUENCY_);}
u64 firmware_timer_get_time_us() { return (firmware_timer_get_cycle() * 1e6 / TIMER_FREQUENCY_);}

#endif

u64 firmware_time_get_frequency() { return TIMER_FREQUENCY_; }
u64 firmware_time_get_ns() { return TIMER_PERIOD_NS_; }


void firmware_timer_print() {
    FW_INFO("Time: %lld ms\n", firmware_timer_get_time_ms());
}


#ifdef __cplusplus
}
#endif
