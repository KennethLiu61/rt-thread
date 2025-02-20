/*
 * Copyright (c) 2006-2023, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2023/06/25     flyingcys    first version
 */
#include <rthw.h>
#include <rtthread.h>

#include "board.h"

#include "sbi.h"

#ifdef RT_USING_SMART
#include "riscv_mmu.h"
#include "mmu.h"
#include "page.h"
#include "lwp_arch.h"

/* respect to boot loader, must be 0xFFFFFFC000200000 */
RT_STATIC_ASSERT(kmem_region, KERNEL_VADDR_START == 0xFFFFFFC000200000);

rt_region_t init_page_region = {(rt_size_t)RT_HW_PAGE_START, (rt_size_t)RT_HW_PAGE_END};

extern size_t MMUTable[];

struct mem_desc platform_mem_desc[] = {
    {KERNEL_VADDR_START, (rt_size_t)RT_HW_PAGE_END - 1, (rt_size_t)ARCH_MAP_FAILED, NORMAL_MEM},
};

#define NUM_MEM_DESC (sizeof(platform_mem_desc) / sizeof(platform_mem_desc[0]))

#endif /* RT_USING_SMART */

void init_bss(void)
{
    unsigned int *dst;

    dst = &__bss_start;
    while ((rt_ubase_t)dst < (rt_ubase_t)&__bss_end)
    {
        *dst++ = 0;
    }
}

static void __rt_assert_handler(const char *ex_string, const char *func, rt_size_t line)
{
    rt_kprintf("(%s) assertion failed at function:%s, line number:%d \n", ex_string, func, line);
    asm volatile("ebreak" ::
                     : "memory");
}

void primary_cpu_entry(void)
{
    /* disable global interrupt */
    rt_hw_interrupt_disable();
    rt_assert_set_hook(__rt_assert_handler);

    entry();
}

#define IOREMAP_SIZE (1ul << 30)

#ifndef ARCH_REMAP_KERNEL
#define IOREMAP_VEND USER_VADDR_START
#else
#define IOREMAP_VEND 0ul
#endif

#ifdef ARCH_RISCV_VECTOR
#include "vector_encoding.h"

#define read_csr(reg) ({ unsigned long __tmp; \
    asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
    __tmp; })

#define write_csr(reg, val) ({ \
    asm volatile ("csrw " #reg ", %0" :: "rK"(val)); })

#define set_csr(reg, bit) ({ unsigned long __tmp; \
    asm volatile ("csrrs %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
    __tmp; })

#define clear_csr(reg, bit) ({ unsigned long __tmp; \
    asm volatile ("csrrc %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
    __tmp; })

void riscv_v_enable(void)
{
	set_csr(sstatus, SSTATUS_VS);
}

void riscv_v_disable(void)
{
	clear_csr(sstatus, SSTATUS_VS);
}

void read_sstatus(void)
{
    rt_kprintf("sstatus = 0x%lx\n",read_csr(sstatus));
}
#endif

void rt_hw_board_init(void)
{
#ifdef RT_USING_SMART
    /* init data structure */
    rt_hw_mmu_map_init(&rt_kernel_space, (void *)(IOREMAP_VEND - IOREMAP_SIZE), IOREMAP_SIZE, (rt_size_t *)MMUTable, PV_OFFSET);

    /* init page allocator */
    rt_page_init(init_page_region);

    /* setup region, and enable MMU */
    rt_hw_mmu_setup(&rt_kernel_space, platform_mem_desc, NUM_MEM_DESC);
#endif

    /* initialize memory system */
#ifdef RT_USING_HEAP
    rt_system_heap_init(RT_HW_HEAP_BEGIN, RT_HW_HEAP_END);
#endif

    /* initalize interrupt */
    rt_hw_interrupt_init();

    /* init rtthread hardware */
    rt_hw_tick_init();

#if defined(SOC_TYPE_BM1690_AP) || (TPU_INDEX == 0)
#ifdef RT_USING_SERIAL
    rt_hw_uart_init();
#endif

#ifdef RT_USING_CONSOLE
    /* set console device */
    rt_console_set_device(RT_CONSOLE_DEVICE_NAME);
#endif /* RT_USING_CONSOLE */
#endif

#ifdef RT_USING_COMPONENTS_INIT
    rt_components_board_init();
#endif

#ifdef RT_USING_HEAP
    rt_kprintf("heap: [0x%lx - 0x%lx]\n", (rt_ubase_t)RT_HW_HEAP_BEGIN, (rt_ubase_t)RT_HW_HEAP_END);
#endif /* RT_USING_HEAP */
}
