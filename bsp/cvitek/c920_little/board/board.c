/*
 * Copyright (c) 2006-2024, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2024/01/11     flyingcys    The first version
 */
#include <rthw.h>
#include <rtthread.h>

#include "board.h"

#include "mmio.h"

int my_test_irq_flag = 0;
int my_test_irq_cnt = 0;

#define SOFT_RESET_REG0	    0x7050003000
#define TOP_MISC_CONTROL_REGISTER 0x7050000008
void watchdog_stop(void)
{
	mmio_write_32(TOP_MISC_CONTROL_REGISTER, mmio_read_32(TOP_MISC_CONTROL_REGISTER) & ~(1 << 2));
	mmio_write_32(SOFT_RESET_REG0, mmio_read_32(SOFT_RESET_REG0) | (0x1 << 14));
}

void rt_hw_board_init(void)
{
#ifdef RT_USING_HEAP
    /* initialize memory system */
    rt_system_heap_init(RT_HW_HEAP_BEGIN, RT_HW_HEAP_END);
#endif

    /* initalize interrupt */
    rt_hw_interrupt_init();

#ifdef RT_USING_SERIAL
    rt_hw_uart_init();
#endif

    /* Set the shell console output device */
#if defined(RT_USING_CONSOLE) && defined(RT_USING_DEVICE)
    rt_console_set_device(RT_CONSOLE_DEVICE_NAME);
#endif
    /* init rtthread hardware */
    rt_hw_tick_init();

#ifdef RT_USING_COMPONENTS_INIT
    // rt_components_board_init();
#endif

#ifdef RT_USING_HEAP
    /* initialize memory system */
    rt_kprintf("RT_HW_HEAP_BEGIN:%x RT_HW_HEAP_END:%x size: %d\r\n", RT_HW_HEAP_BEGIN, RT_HW_HEAP_END, RT_HW_HEAP_END - RT_HW_HEAP_BEGIN);
#endif
    // do_gpio_irq_test();
}
