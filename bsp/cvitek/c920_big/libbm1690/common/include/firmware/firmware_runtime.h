#ifndef FIRMWARE_RUNTIME_H
#define FIRMWARE_RUNTIME_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void firmware_lock(int index);
void firmware_unlock(int index);

int tpu_physical_core_id();
#if (defined(USING_CMODEL) && defined(DEBUG)) || (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
#define CORE_PRINT(fmt, ...)                             \
    do                                                  \
    {                                                   \
        firmware_lock(0);                               \
        printf("[%d]" fmt, tpu_physical_core_id(), ##__VA_ARGS__);     \
        firmware_unlock(0);                             \
    } while (0)
#define CORE_PRINT_CORE(core_idx, fmt, ...)                  \
    do                                                      \
    {                                                       \
        if (tpu_physical_core_id() == core_idx)                            \
        {                                                   \
            firmware_lock(0);                               \
            printf("[%d]" fmt, tpu_physical_core_id(), ##__VA_ARGS__);     \
            firmware_unlock(0);                             \
        }                                                   \
    } while (0)
#else
    #define CORE_PRINT(fmt, ...)
    #define CORE_PRINT_CORE(core_idx, fmt, ...)
#endif

typedef struct {
    int                     nodechip_idx;
    int                     bad_plane_id;
    volatile int            terminated;
} firmware_runtime_param_t;

typedef struct sg_kapi_header {
    unsigned int api_id;
    unsigned int api_size;  // size of payload, not including header
    unsigned long long api_handle;
    unsigned int api_seq;
    unsigned int duration;
    unsigned int result;
} __attribute__((packed)) sg_kapi_header_t;

typedef sg_fw_status_t (*api_handler_t)(
    firmware_runtime_param_t   *param,
    unsigned char              *api_buf,
    int                         size);

typedef enum {
    SG_FW_EVENT_NULL          = 0,
    SG_FW_EVENT_MSG           = 1,
    SG_FW_EVENT_CLI           = 2,
    SG_FW_EVENT_QUIT          = 3,
    SG_FW_EVENT_INVALID
} sg_fw_event_t;

void load_lookup_tables();

extern void * firmware_main(void *p);

#ifdef __cplusplus
}
#endif

#endif /* FIRMWARE_RUNTIME_H */
