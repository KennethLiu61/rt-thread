#ifndef __FIRMWARE_PROFILE_H__
#define __FIRMWARE_PROFILE_H__

#include "firmware_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PROFILE_UNKNOWN   (-1)
#define PROFILE_FUNC       (0)
#define PROFILE_NODE_SET   (1)
#define PROFILE_NODE_WAIT  (2)
#define PROFILE_CDMA       (3)
#define PROFILE_GDE        (4)
#define PROFILE_SORT       (5)
#define PROFILE_NMS        (6)
#define PROFILE_VSDMA      (7)

#define PROFILE_CUSTOM   (100)

#pragma pack(1)
#if defined(__sg2260__) || defined(__sg2260e__)
typedef struct {
    int32_t type;
    uint32_t inst_id; 
  // packed node info into type param
} fw_profile_time_info_t;
# else
typedef struct {
    unsigned int profile_id; // refer to extra data
    unsigned int type;
    unsigned long long id;  // detail info: layer_type, group, cmd_id ...
    unsigned long long begin_cycle;
    unsigned long long end_cycle;
} fw_profile_time_info_t;
#endif
#pragma pack()

void set_profile_time_enabled(int enable);
int profile_time_enabled();

void profile_time_add_extra_data(fw_profile_time_info_t* info, int type, const unsigned char* data, size_t len);
void profile_time_add_extra_string(fw_profile_time_info_t* info, const char* data);
#if defined(__sg2260__) || defined(__sg2260e__)
void profile_time_set_node(int engine, int func, int special_func, int info, CMD_ID_NODE* pid_node, u64* high, u64* low, int elt);
# else
fw_profile_time_info_t* profile_time_begin_set_node(CMD_ID_NODE* pid_node, int engine);
fw_profile_time_info_t* profile_time_begin_wait_node(CMD_ID_NODE* pid_node);
fw_profile_time_info_t* profile_time_begin_cdma(unsigned long long id);
fw_profile_time_info_t* profile_time_begin_gde(unsigned long long id);
fw_profile_time_info_t* profile_time_begin_sort(unsigned long long id);
fw_profile_time_info_t* profile_time_begin_nms(unsigned long long id);
void profile_time_add_extra_binary(fw_profile_time_info_t* info, const unsigned char* data, int len);
#endif
fw_profile_time_info_t* profile_time_begin_func(unsigned long long id);
fw_profile_time_info_t *profile_time_begin_custom(unsigned long long data);
void profile_time_end(void* handle);
//return the real len
size_t get_profile_time_data_len();
size_t get_profile_time_data(unsigned char* buffer, size_t buffer_len, size_t offset);
size_t get_profile_extra_data_len();
size_t get_profile_extra_data(unsigned char* buffer, size_t buffer_len, size_t offset);

sg_fw_status_t sg_api_get_profile_data(
    unsigned char *api_buf, int size);
sg_fw_status_t sg_api_set_profile(
    unsigned char *api_buf, int size);

#ifdef __cplusplus
}
#endif

#endif
