#include "mmio.h"
#include <rthw.h>
#include <rtthread.h>

#include "board.h"

#define REG_GPIO0_BASE      0x07040009000
#define REG_GPIO1_BASE      0x0704000A000
#define REG_GPIO2_BASE      0x0704000B000
#define SOFT_RESET_REG0     0x7050003000
#define CLK_EN_REG0         0x7050002000

#define GPIO_SWPORTA_DR     0x00 // 数据寄存器（Data Register）
#define GPIO_SWPORTA_DDR    0x04 // 数据方向寄存器（Data Direction Register）
#define GPIO_SWPORTA_CTL    0x08 // 控制寄存器（Control Register）
#define GPIO_INTEN          0x30 // 中断使能寄存器（Interrupt Enable Register）
#define GPIO_INTMASK        0x34 // 中断屏蔽寄存器（Interrupt Mask Register）
#define GPIO_INTTYPE_LEVEL  0x38 // 中断类型级别寄存器（Interrupt Type Level Register）
#define GPIO_INT_POLITY     0x3c // 中断极性寄存器（Interrupt Polarity Register）
#define GPIO_INTSTATUS      0x40 // 中断状态寄存器（Interrupt Status Register）
#define GPIO_RAW_INTSTATUS  0x44 // 原始中断状态寄存器（Raw Interrupt Status Register）
#define GPIO_DEBOUNCE       0x48 // 去抖动寄存器（Debounce Register）
#define GPIO_PORTA_EOI      0x4c // 中断清除寄存器（End of Interrupt Register）
#define GPIO_EXT_PORTA      0x50 // 外部端口寄存器（External Port Register）
#define GPIO_LS_SYNC        0x60 // 低功耗同步寄存器（Low Power Synchronous Register）
#define GPIO_ID_CODE        0x64 // ID代码寄存器（ID Code Register）
#define GPIO_VER_ID_CODE    0x6C // 版本ID代码寄存器（Version ID Code Register）
#define GPIO_CONFIG_REG1    0x74 // 配置寄存器1（Configuration Register 1）
#define GPIO_CONFIG_REG2    0x70 // 配置寄存器2（Configuration Register 2）

#define GPIO0_INTR_FLAG 26
#define GPIO1_INTR_FLAG 27
#define GPIO2_INTR_FLAG 28

static void gpio0_irq_handler(int irqn, void *priv)
{
    mmio_write_32(REG_GPIO0_BASE + GPIO_INTEN, 0x0);
    mmio_write_32(0x70101F0000, 0x1234);
    mmio_write_32(0x70101F0004, 0x5678);
    return;
}

int do_gpio_irq_test(void)
{
    rt_kprintf("gpio irq start\n");
    rt_hw_interrupt_install(GPIO0_INTR_FLAG, gpio0_irq_handler, RT_NULL, "gpio_0 interrupt");
    rt_hw_interrupt_umask(GPIO0_INTR_FLAG);

        // set gpio input
    mmio_write_32(REG_GPIO0_BASE + GPIO_SWPORTA_DDR, 0x0);
    
    // clr gpio intr
    mmio_write_32(REG_GPIO0_BASE + GPIO_PORTA_EOI, 0xffffffff);
    
    // enable gpio intr
    mmio_write_32(REG_GPIO0_BASE + GPIO_INTEN, 0xffffffff);
    
    // activity low
    mmio_write_32(REG_GPIO0_BASE + GPIO_INT_POLITY, 0x0);
    rt_hw_us_delay(100);
    // udelay(10); // pld time delay

    // activity high
    mmio_write_32(REG_GPIO0_BASE + GPIO_INT_POLITY, 0xffffffff);

    rt_hw_us_delay(100);
    //udelay(10); // pld time delay

    // disable gpio intr
    mmio_write_32(REG_GPIO0_BASE + GPIO_INTEN, 0x0);
    rt_kprintf("gpio irq end\n");

    return 0;
}