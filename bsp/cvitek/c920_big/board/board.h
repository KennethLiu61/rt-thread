/*
 * Copyright (c) 2006-2023, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2023/06/25     flyingcys    first version
 */

#ifndef BOARD_H__
#define BOARD_H__

#include <rtconfig.h>
#include "drv_uart.h"
#include "tick.h"

extern unsigned int __bss_start;
extern unsigned int __bss_end;

#ifndef RT_USING_SMART
#ifdef C920_BOOT_ADDR
#define KERNEL_VADDR_START (C920_BOOT_ADDR)
#else
#error "you must define C920_BOOT_ADDR!!!"
#endif
#endif

#define RT_HW_HEAP_BEGIN ((void *)&__bss_end)
//定义了从RT_HW_HEAP_BEGIN到内存末尾(1MB)的地址范围，即从__bss_end到内存末尾(1MB)的地址范围。
//这个地址范围通常用于分配动态内存，例如在RT-Thread中使用内存池来管理内存。
#define RT_HW_HEAP_END   ((void *)(KERNEL_VADDR_START + (TOTAL_MEMORY - 1 * 1024 * 1024)))
#define RT_HW_PAGE_START RT_HW_HEAP_END
#define RT_HW_PAGE_END   ((void *)(KERNEL_VADDR_START + 32 * 1024 * 1024))

void rt_hw_board_init(void);

#endif
