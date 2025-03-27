#ifndef __FIRMWARE_CACHE_COMMAND_H__
#define __FIRMWARE_CACHE_COMMAND_H__

#include "firmware_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

void set_cache_command(int enable);
int cache_command_enabled();
void cache_command(int engine_type, const void* command_data, u32 len);
void get_cached_command_info(int engine_type, u32* cmd_num, u32* cmd_bytes);
void get_cached_command_data(int engine_type, u8* data, u32 len);

#ifdef __cplusplus
}
#endif

#endif
