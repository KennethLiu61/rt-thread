#ifndef __COMMON_H__
#define __COMMON_H__

#include <rtthread.h>
#include <stdarg.h>
#include "sg_types.h"
#include "msp_list.h"
#include "sys/time.h"

#define pr_debug(format, ...) rt_kprintf(format, ##__VA_ARGS__)
#define pr_err(format, ...)

extern struct thread_item *cur_thread;

#define TPU_CORE_ID			(cur_thread->tpu_id)

#define get_time(time) \
({	struct timespec now;\
	clock_gettime(CLOCK_REALTIME, &now);\
	time = now.tv_sec * 1000 * 1000 * 1000 + now.tv_nsec;\
})

#define time_simple() \
({	struct timespec now;\
	clock_gettime(CLOCK_REALTIME, &now);\
	now.tv_sec * 1000 * 1000 * 1000 + now.tv_nsec;\
})

enum {
	SYNC_MODE = 0,
	ASYNC_MODE
};

struct tp_config {
	uint64_t log_addr;
	uint64_t memory_type;
	uint64_t share_memory_addr;
};

struct tp_status {
	uint64_t kernel_exec_time;
	uint64_t tp_alive_time;
	uint64_t tp_heart_beat;
};

struct thread_item {
	struct list_head list;
	int tpu_id;
	struct list_head load_lib_list;
	struct list_head task_list;
	struct func_record *func_table;
	int kernel_running;
	int sync_mode;
	struct tp_status *tp_status;
	uint64_t kernel_exec_time;
	uint64_t tp_alive_time;
	uint64_t start_time;
	uint64_t end_time;
	uint64_t read_loop;
	int (*task_barrier)(int msg_id, int block_num);
	void (*poll_engine_done)(void);
};

//for log
#define LOG_MEMORY_CACHEABLE	0x0
#define LOG_MEMORY_NONE_CACHEABLE	0x1

struct tp_log_struct {
	char *addr;
	uint64_t size;
	uint64_t record_len;
	uint64_t start_time;
	int fd;
	int log_type;
	int (*record)(const char *format, va_list args);
	int (*time)(const char *format, va_list args);
};
extern struct tp_log_struct tp_log;

int log_init(uint64_t start_addr, uint64_t len, int type, int memory_type);
int fix_log_time(uint64_t delate_time);
int tp_debug(const char *format, ...);

//for cache
void invalidate_dcache_range(void *__start, uint64_t size);
void clean_dcache_range(void *__start, uint64_t size);

#endif /* __COMMON_H__ */