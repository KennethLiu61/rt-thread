#ifndef FIRMWARE_PMU_H_
#define FIRMWARE_PMU_H_

#include "firmware_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void enable_tpu_perf_monitor();
void disable_tpu_perf_monitor();
void enable_pmu(unsigned enable_bits, void (*before_func)(int), int param);
bool cdma_port_is_valid(int port);
void set_pmu_param(void* api);

void show_tpu_perf_data(int gdma_max_num, int sdma_max_num, int tiu_max_num, int cdma_max_num);

#ifdef __cplusplus
}
#endif

#endif