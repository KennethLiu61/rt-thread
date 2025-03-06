/*
 * Copyright (c) 2006-2023, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2023/06/25     flyingcys    first version
 */

#include <rtthread.h>
#include <stdio.h>
#include <drivers/dev_pin.h>
#include <dlfcn.h>

#if defined(BOARD_TYPE_MILKV_DUO256M) || defined(BOARD_TYPE_MILKV_DUO256M_SPINOR)
#define LED_PIN     "E02" /* Onboard LED pins */
#elif defined(BOARD_TYPE_MILKV_DUO) || defined(BOARD_TYPE_MILKV_DUO_SPINOR)
#define LED_PIN     "C24" /* Onboard LED pins */
#elif defined(BOARD_TYPE_MILKV_DUOS)
#define LED_PIN     "A29" /* Onboard LED pins */
#endif

uint64_t c920_riscv_gettime(void)
{
    uint64_t time_elapsed;
    __asm__ __volatile__(
        "rdtime %0"
        : "=r"(time_elapsed));
    return time_elapsed;
}

int main(void)
{
#ifdef RT_USING_SMART
    rt_kprintf("Hello RT-Smart!\n");
#else
    rt_kprintf("Hello RISC-V!\n");
#endif

	void *handle;
	void (*hello_func)() = NULL;
	int (*add_func)(int, int) = NULL;
	handle = dlopen("/libtest.so", RTLD_LAZY);
	if(!handle) {
		rt_kprintf("dlopen failed! /libtest.so \n");
		handle = dlopen("libtest.so", RTLD_LAZY);
		if(!handle)
			rt_kprintf("dlopen failed! libtest.so \n");
	} 

	if(handle){
		// 获取函数的地址
		*(void **)(&hello_func) = dlsym(handle, "lib_func");
		if(*hello_func == NULL) {
			rt_kprintf("dlsym failed\n");
		} else {
			// 调用函数
			hello_func();
		}

		*(int **)(&add_func) = dlsym(handle, "add_func");
		if(*add_func == NULL) {
			rt_kprintf("dlsym add func failed!\n");
		} else {
			int value = add_func(12, 34);
			rt_kprintf("add_func: value = %d\n", value);
		}
		// 关闭库
		dlclose(handle);
	}

	// daemon_main();
	extern int tpu_daemon_run(void);
	tpu_daemon_run();

	extern void msi_irq_init(void);
	msi_irq_init();

	while (1)
	{
		rt_thread_mdelay(1000);
	}

	return 0;
}
