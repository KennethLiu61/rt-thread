#ifndef __API_H__
#define __API_H__

#include "common.h"
#include "uthash.h"
#include "sg_types.h"
#include "md5.h"

#define LIB_MAX_NAME_LEN 64
#define FUNC_MAX_NAME_LEN 64

enum API_ID {
	API_ID_LOAD_LIB = 0x90000001,
	API_ID_GET_FUNC = 0x90000002,
	API_ID_LAUNCH_FUNC = 0x90000003,
	API_ID_UNLOAD_LIB = 0x90000004,
};

enum TASK_TYPE {
	ERROR_TASK_TYPE = 0,
	TASK_S2D,
	TASK_D2S,
	TASK_D2D,
	LAUNCH_KERNEL,
	TRIGGER_TASK,
	SYNC_TASK,
	POLL_ENGINE_DONE,
};

struct cc_sys_info {
	uint32_t group_id;
	uint32_t block_id;
};

struct task_header {
	union {
		struct {
			uint64_t task_type:8;
			uint64_t task_dest:8;
			uint64_t task_async:8;
			uint64_t cdma_mode:8;
			uint64_t reserved:32;
		};
		uint64_t task_info;
	};
	uint64_t task_id;
	uint64_t group_num;
	uint64_t block_num;
	struct cc_sys_info request_cc_info;
	uint64_t stream_id;
	uint64_t task_body_size;
} __attribute__((packed));

struct task {
	struct task_header task_header;
	char *task_body;
} __attribute__((packed));

struct task_item {
	struct list_head list;
	struct task task;
	int task_enable;
	int start_time;
	int end_time;
	int need_response;
};

struct api_header {
	enum API_ID api_id;
	uint32_t api_size;  // size of payload, not including header
} __attribute__((packed));

struct task_response {
	uint64_t stream_id;
	uint64_t task_id;
	uint32_t group_id;
	uint32_t block_id;
	uint64_t start_time;
	uint64_t end_time;
	uint64_t kr_time;
	uint64_t result;
};

struct load_module_internal {
	uint8_t *library_path;
	void *library_addr;
	uint32_t size;
	uint8_t library_name[LIB_MAX_NAME_LEN];
	unsigned char md5[MD5SUM_LEN];
	int cur_rec;
} __attribute__((packed));

struct launch_func_internal {
	unsigned char fun_name[FUNC_MAX_NAME_LEN];
	unsigned char lib_name[LIB_MAX_NAME_LEN];
	unsigned char lib_md5[MD5SUM_LEN];
	unsigned int size;
	uint8_t param[4096];
} __attribute__((packed));

struct lib_info {
	void *handle;
	unsigned char lib_name[LIB_MAX_NAME_LEN];
	unsigned char md5[MD5SUM_LEN];
};

struct library_item {
	struct list_head list;
	struct lib_info lib;
};

struct func_name {
	struct list_head list;
	unsigned char func_name[FUNC_MAX_NAME_LEN];
};

struct func_item {
	unsigned char func_name[FUNC_MAX_NAME_LEN];
	unsigned char lib_md5[MD5SUM_LEN];
};

struct func_record {
	struct func_item f_item; // key
	int (*f_ptr)(void *addr, unsigned int size); // value
	UT_hash_handle hh;
};

int run_empty_kernel(void);
int load_lib_process(struct task_item *task_item);
int launch_func_process(struct task_item *task_item);
int unload_lib_process(struct task_item *task_item);

#endif /* __API_H__ */