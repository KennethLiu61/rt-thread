#ifndef __MEMMAP_H__

#define DEVICE_BASE	0x00000000UL
#define DEVICE_SIZE	0x100000000UL

#define MEMORY_BASE	0x100000000UL
#define MEMORY_SIZE	(16UL * 1024 * 1024 * 1024)

#define UART_BASE	0x5022000
// #define UART_BASE	0x29180000
#define TBOX_FORMAT_OUT	0x29180020

#if defined(CONFIG_TARGET_PALLADIUM)

#define UART_PCLK	153600
#define UART_BAUDRATE	9600

#elif defined(CONFIG_TARGET_ASIC)

#define UART_PCLK	500000000
#define UART_BAUDRATE	115200

#else
#error "no target specified"
#endif

#define EFUSE_BASE      0x27040000UL
#define DRAM_BASE	0x100000000UL

#define LOG_BUFFER_ADDR	(DRAM_BASE + 0xc00000)
#define LOG_BUFFER_SIZE	(4 * 1024 * 1024)
#define LOG_LINE_SIZE	512

#define TOP_BASE			0x28100000UL

// GP30 for C906
#define TOP_GP30_SET			(TOP_BASE + 0xf8)
#define TOP_GP30_CLR			(TOP_BASE + 0x78)
// GP31 for A53
#define TOP_GP31_SET			(TOP_BASE + 0xfc)
#define TOP_GP31_CLR			(TOP_BASE + 0x7c)

#define SHARE_REG_BASE			(TOP_BASE + 0x80)
#define SHARE_REG_MESSAGE_WP		(0 + CONFIG_CORE_ID * 5)
#define SHARE_REG_MESSAGE_RP		(1 + CONFIG_CORE_ID * 5)
#define SHARE_REG_FW_STATUS		(2 + CONFIG_CORE_ID * 5)
#define SHARE_REG_C906_FW_LOG_RP       	(3 + CONFIG_CORE_ID * 5)
#define SHARE_REG_C906_FW_LOG_WP       	(4 + CONFIG_CORE_ID * 5)
#define SHARE_REG_C906_FW_MODE		10

#define C906_START_STEP_ENTER_MAIN        1
#define C906_START_STEP_TIMER_INIT        2
#define C906_START_STEP_MMU_INIT          3
#define C906_START_STEP_FIQ_INIT          4
#define C906_START_STEP_FIQ_INIT_DONE     41
#define C906_START_STEP_FIQ_INIT_DOING    42
#define C906_START_STEP_ENTER_BMDNN       5
#define C906_START_STEP_GET_BAD_NPU       51
#define C906_START_STEP_INIT_LOCAL_IRQ    52
#define C906_START_STEP_I2C_SLAVE_INIT    53
#define C906_START_STEP_PCIE_TABLE_INIT   54
#define C906_START_STEP_UNMASK_ALL_INTC   55
#define C906_START_STEP_TEST_FAILED       0x0B0B0B0B
#define C906_START_STEP_TEST_SUCCEED      0x0A0A0A0A

#define CLINT_BASE              0x14000000UL
#define PLIC_BASE               0x10000000UL

/* CLINT */
#define CLINT_TIMECMPL0         (CLINT_BASE + 0x4000)
#define CLINT_TIMECMPH0         (CLINT_BASE + 0x4004)

#define CLINT_MTIME(cnt)             asm volatile("csrr %0, time\n" : "=r"(cnt) :: "memory");

/* PLIC */
#define PLIC_PRIORITY0          (PLIC_BASE + 0x0)
#define PLIC_PRIORITY1          (PLIC_BASE + 0x4)
#define PLIC_PRIORITY2          (PLIC_BASE + 0x8)
#define PLIC_PRIORITY3          (PLIC_BASE + 0xc)
#define PLIC_PRIORITY4          (PLIC_BASE + 0x10)

#define PLIC_PENDING1           (PLIC_BASE + 0x1000)
#define PLIC_PENDING2           (PLIC_BASE + 0x1004)
#define PLIC_PENDING3           (PLIC_BASE + 0x1008)
#define PLIC_PENDING4           (PLIC_BASE + 0x100C)

#define PLIC_ENABLE1            (PLIC_BASE + 0x2000)
#define PLIC_ENABLE2            (PLIC_BASE + 0x2004)
#define PLIC_ENABLE3            (PLIC_BASE + 0x2008)
#define PLIC_ENABLE4            (PLIC_BASE + 0x200C)

#define PLIC_THRESHOLD          (PLIC_BASE + 0x200000)
#define PLIC_CLAIM              (PLIC_BASE + 0x200004)


#endif
