#include <rtthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "common.h"

#define L1_CACHE_SHIFT	   6
#define L1_CACHE_BYTES	   (1 << L1_CACHE_SHIFT)

static inline void CBO_flush(unsigned long start, unsigned long size)
{
#ifdef __riscv_xlen
	register unsigned long i asm("a0") = start & ~(L1_CACHE_BYTES - 1);
	for (; i < (start+size); i += L1_CACHE_BYTES)
		asm volatile (".long 0x025200f");
#endif
}

static inline void CBO_clean(unsigned long start, unsigned long size)
{
#ifdef __riscv_xlen
	register unsigned long i asm("a0") = start & ~(L1_CACHE_BYTES - 1);
	for (; i < (start+size); i += L1_CACHE_BYTES)
		asm volatile (".long 0x015200f");
#endif
}

static inline void CBO_inval(unsigned long start, unsigned long size)
{
#ifdef __riscv_xlen
	register unsigned long i asm("a0") = start & ~(L1_CACHE_BYTES - 1);
	for (; i < (start+size); i += L1_CACHE_BYTES)
		asm volatile (".long 0x05200f");
#endif
}

static inline void thead_sync(void)
{
	asm volatile (".long 0x0180000b");
}

static inline void sync_is(void)
{
	asm volatile (".long 0x01b0000b");
}

void clean_dcache_range(void *__start, uint64_t size)
{
	CBO_flush((unsigned long)__start, size);

	thead_sync();
}

void invalidate_dcache_range(void *__start, uint64_t size)
{
	CBO_inval((unsigned long)__start, size);

	thead_sync();
}

#define LOG_NONE	0x0
#define LOG_MEM		0x1
#define LOG_UART	0x2

static int none_log(const char *format, va_list args)
{
	return 0;
}

struct tp_log_struct tp_log = {
	.record = none_log,
	.time = none_log,
	.start_time = 0,
};
static pthread_spinlock_t lock;

static int mem_log(const char *format, va_list args)
{
	int ret;

	pthread_spin_lock(&lock);
	ret = sprintf(tp_log.addr + tp_log.record_len, "[%15lu]", time_simple() - tp_log.start_time);
	tp_log.record_len += ret;
	tp_log.record_len &= (tp_log.size - 1);
	ret = vsprintf(tp_log.addr + tp_log.record_len, format, args);
	tp_log.record_len += ret;
	tp_log.record_len &= (tp_log.size - 1);
	pthread_spin_unlock(&lock);

	return ret;
}

static int uart_log(const char *format, va_list args)
{
	int ret = 0;

	fprintf(stderr, "[tp]: ");
	ret = vfprintf(stderr, format, args);

	return ret;
}

int log_init(uint64_t start_addr, uint64_t len, int type, int memory_type)
{
	__attribute__((unused)) uint64_t time1;
	__attribute__((unused)) uint64_t time2;

	if (type == LOG_NONE) {
		tp_log.record = none_log;
		tp_log.time = none_log;
	} else if (type == LOG_MEM) {
		if (tp_log.addr == NULL) {
			if (memory_type == LOG_MEMORY_NONE_CACHEABLE)
				tp_log.addr = (void *)(start_addr);
			else
				tp_log.addr = (void *)(start_addr);

			if (tp_log.addr == NULL) {
				pr_err("failed to map memory for log:0x%lx\n", start_addr);
				return -1;
			}
		}

		tp_log.record = mem_log;
		tp_log.size = len;
		tp_log.record_len = 0;

		get_time(time1);
		asm_memset(tp_log.addr + 4096, 0x0, 0x1000);
		get_time(time2);
		tp_debug("clean 4K use %lu ns\n", time2 - time1);
	} else if (type == LOG_UART) {
		tp_log.record = uart_log;
		tp_log.record = uart_log;
	} else {
		pr_err("error log type:%d\n", type);
	}

	return 0;
}

int fix_log_time(uint64_t delate_time)
{
	tp_log.start_time = delate_time;

	return 0;
}

int tp_debug(const char *format, ...)
{
	int ret = 0;
	va_list args;

	va_start(args, format);
	ret = tp_log.record(format, args);
	va_end(args);

	return ret;
}
RTM_EXPORT(tp_debug);

//for musl toolchain
void __assert_fail(const char *expr, const char *file, int line, const char *func)
{
    rt_kprintf("Assertion failed: %s (%s: %s: %d)\n", expr, file, func, line);
    abort();
}
RTM_EXPORT(__assert_fail);