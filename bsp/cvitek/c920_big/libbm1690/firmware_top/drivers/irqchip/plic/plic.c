#include <framework/plic.h>
#include <framework/common.h>
#include <framework/module.h>
#include <irq.h>
#include <arch.h>
#include <memmap.h>
#include <string.h>

static void plic_mask_irq(int irq_num)
{
	uint32_t mask = irq_num;
	uint32_t value = 0;
	if (irq_num < 16) {
		pr_err("mask irq_num is %d\n", irq_num);
		return;
	}
	value = readl(PLIC_ENABLE1 + 4 * (mask / 32));
	value &= ~(0x1 << (mask % 32));
	writel(value, (PLIC_ENABLE1 + (mask / 32) * 4));
}

static void plic_unmask_irq(int irq_num)
{
	uint32_t mask = irq_num;
	uint32_t value = 0;
	if (irq_num < 16) {
		pr_err("unmask irq_num is %d\n", irq_num);
		return;
	}
	value = readl(PLIC_ENABLE1 + 4 * (mask / 32));
	value |= (0x1 << (mask % 32));
	writel(value, (PLIC_ENABLE1 + (mask / 32) * 4));
}

static int plic_ack_irq()
{
	return readl(PLIC_CLAIM);
}

static void plic_eoi_irq(int irq_num)
{
	writel(irq_num, PLIC_CLAIM);
}

static void plic_set_priority_irq(int irq_num, int priority)
{
	pr_debug("plic_set_priority_irq addr(%x) = %d\n", PLIC_PRIORITY0 + irq_num * 4, priority);
	writel(priority, (PLIC_PRIORITY0 + irq_num * 4));
}

static void plic_set_threshold(uint32_t threshold)
{
	writel(threshold, PLIC_THRESHOLD);
}

static struct irq_chip plic_chip = {
	.name           	= "RISCV PLIC",
	.irq_mask       	= plic_mask_irq,
	.irq_unmask     	= plic_unmask_irq,
	.irq_ack        	= plic_ack_irq,
	.irq_set_priority 	= plic_set_priority_irq,
	.irq_eoi        	= plic_eoi_irq,
	.irq_set_threshold 	= plic_set_threshold,
};

static struct irq_action g_irq_action[NUM_IRQ];

void cpu_enable_irqs(void)
{
    	set_csr(mstatus, MSTATUS_MIE);
    	set_csr(mie, MIP_MEIE);
}

void cpu_disable_irqs(void)
{
    	clear_csr(mstatus, MSTATUS_MIE);
    	clear_csr(mie, MIP_MEIE);
}

int request_irq(unsigned int irqn, irq_handler_t handler, unsigned long flags,
		const char *name, void *priv)
{
	if ((irqn < 0) || (irqn >= NUM_IRQ))
		return -1;

	pr_debug("request_irq irqn = %d\n handler=%lx name = %s\n", irqn, (long) handler, name);
	g_irq_action[irqn].handler = handler;
	if (name) {
		memcpy(g_irq_action[irqn].name, name, sizeof(g_irq_action[irqn].name));
		g_irq_action[irqn].name[sizeof(g_irq_action[irqn].name) - 1] = 0;
	}
	g_irq_action[irqn].irq = irqn;
	g_irq_action[irqn].flags = flags;
	g_irq_action[irqn].priv = priv;
	// set highest priority
	plic_chip.irq_set_priority(irqn, 7);
	// unmask irq
	plic_chip.irq_unmask(irqn);

	return 0;
}

void printf_once(char *string, int irq)
{
        static uint8_t __print_once = false;
        static int id = 0;

        if (!__print_once || (id != irq)) {
                __print_once = true;
		id = (int) irq;
                printf("%s %d\n", string, irq);
        }
}

void do_irq(void)
{
	int irqn;

	do {
		irqn = plic_chip.irq_ack();
		if (g_irq_action[irqn].handler && irqn) {
			pr_debug("do_irq irqn = %d\n", irqn);
			g_irq_action[irqn].handler(g_irq_action[irqn].irq, g_irq_action[irqn].priv);
		} else if (irqn)
			pr_info("g_irq_action[%i] NULL", irqn);
		else // plic_claim =0
			break;
		// clear plic pending
		plic_chip.irq_eoi(irqn);
	} while (1);
	// clear external interrupt pending
	clear_csr(mip, MIP_MEIE);
}

void disable_irq(unsigned int irqn)
{
	plic_chip.irq_mask(irqn);
}

void enable_irq(unsigned int irqn)
{
	plic_chip.irq_unmask(irqn);
}

static int plic_init(void)
{
	int i;
	// clear interrupt enable
	write_csr(mie, 0);
	// clear interrupt pending
	write_csr(mip, 0);

	// Clean the setting of all IRQ
	for (i = 0; i < 256 * 4; i = i + 4)
		writel(0, ((uintptr_t)PLIC_PRIORITY0 + i));
	for (i = 0; i <= 256 / 32; i++) {
		writel(0, (PLIC_PENDING1  + i * 4));
		writel(0, (PLIC_ENABLE1  + i * 4));
	}
	memset(g_irq_action, 0, sizeof(struct irq_action) * 256);
	plic_chip.irq_set_threshold(0);
	cpu_enable_irqs();
	return 0;
}

arch_init(plic_init);
