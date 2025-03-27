#include <stdint.h>
#include <stdarg.h>
#include <irq.h>
#include <arch.h>
#include <timer.h>
#include <platform.h>
#include <memmap.h>
#include <firmware_top.h>
#include <framework/module.h>
#include <framework/common.h>
#include <framework/plic.h>

static inline void cpu_write32(unsigned long addr, uint32_t value)
{
	writel(value, addr);
}

static inline uint32_t cpu_read32(unsigned long addr)
{
	return readl(addr);
}

static inline void shm_write32(unsigned long index, uint32_t value)
{
	cpu_write32(SHARE_REG_BASE + index * 4, value);
}

static inline uint32_t shm_read32(unsigned long index)
{
	return cpu_read32(SHARE_REG_BASE + index * 4);
}

static int work_mode;

int get_work_mode(void)
{
	return work_mode;
}

static struct tpu_runtime tpu_runtime = {0};

struct tpu_runtime *get_tpu_runtime(void)
{
	return &tpu_runtime;
}

/* used by firmware core */  
/*void fw_log(char *fmt, ...)
{
	u32 index = shm_read32(SHARE_REG_C920_FW_LOG_WP);
	index = index % LOG_BUFFER_SIZE;
	char *ptr = (char *)(LOG_BUFFER_ADDR + index);
	char log_buffer[LOG_LINE_SIZE - 8] = "";
	va_list p;
	va_start(p,fmt);
	vsnprintf(log_buffer, LOG_LINE_SIZE, fmt, p);
	unsigned long long timestamp = timer_tick2us(timer_get_tick());
	printf("[%llx] %s", timestamp, log_buffer);
	snprintf(ptr, LOG_LINE_SIZE,
		 "[%llx] %s", timestamp, log_buffer);
	va_end(p);
	index = index + LOG_LINE_SIZE;
	shm_write32(SHARE_REG_C920_FW_LOG_WP, index);
}*/

static int irq_handler(int irqn, void *priv)
{
	writel(0, 0x260501d8ULL);
	// ready to die
	if (irqn == 183) {
                pr_info("receive irqn = %d, enter wfi\n", irqn);
		// writel(1 << (6 + CONFIG_CORE_ID * 4), TOP_GP30_CLR);
		while (true)
			asm volatile ("wfi");
	} else {
		pr_info("receive irqn = %d\n", irqn);
		// writel(0xf << (4 + CONFIG_CORE_ID * 4), TOP_GP30_CLR);
	}

#if 0
	/* send pcie interrupt */
	if (get_work_mode() == WORK_MODE_PCIE)
		cpu_write32(pcie_msi_addr, pcie_msi_data);
#endif
    return 0;
}

void setup_plic(void)
{
	pr_debug("setup interrupt\n");

	// request irq
	// writel(0xf << (4 + CONFIG_CORE_ID * 4), TOP_GP30_CLR);
	for (int i = 181; i < 185; i++) {
		if (request_irq(i, irq_handler, 0, "plic irq", NULL))
			pr_err("request_irq fail %d\n", i);
	}

	shm_write32(SHARE_REG_FW_STATUS, C920_START_STEP_UNMASK_ALL_INTC);
}

#if 0
void setup_pcie(void)
{
}

int is_normal_memory(const void *addr)
{
	return (unsigned long)addr >= MEMORY_BASE &&
		((unsigned long)addr < (MEMORY_BASE + MEMORY_SIZE));
}
#endif

int sg2260_plat_init(void)
{
#if 1
	work_mode = WORK_MODE_SOC;
#else
	work_mode = shm_read32(SHARE_REG_C920_FW_MODE);
#endif

#if 0
	if (get_work_mode() == WORK_MODE_PCIE)
		setup_pcie();
#endif

#if 0
	setup_plic();
#endif
	return 0;
}

#ifndef CONFIG_TARGET_EMULATOR
plat_init(sg2260_plat_init);
#endif
