#ifndef FIRMWARE_TIMER_H_
#define FIRMWARE_TIMER_H_

#include "common.h"


#ifdef __cplusplus
extern "C" {
#endif

u64 firmware_time_get_frequency();
u64 firmware_time_get_ns();
u64 firmware_timer_get_cycle();
u64 firmware_timer_get_time_ms();
u64 firmware_timer_get_time_us();
void firmware_timer_print();
#define TIMER_FREQUENCY firmware_time_get_frequency()

#ifndef TIMER_PERIOD_NS
#define TIMER_PERIOD_NS firmware_time_get_ns()
#endif

#ifdef __cplusplus
}
#endif

#endif
