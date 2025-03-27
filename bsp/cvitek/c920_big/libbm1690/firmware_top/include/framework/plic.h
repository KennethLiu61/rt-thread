#ifndef __PLIC_H__
#define __PLIC_H__

#include <arch.h>
#include <memmap.h>
#include <irq.h>
#include <asm/csr.h>

#define NUM_IRQ 256

struct irq_chip {
	const char  *name;
	void        (*irq_mask)(int irq_num);
	void        (*irq_unmask)(int irq_num);
	int         (*irq_ack)(void);
	void        (*irq_set_priority)(int irq_num, int priority);
	void        (*irq_eoi)(int irq_num);
	void        (*irq_set_threshold)(uint32_t threshold);
};

void do_irq(void);
int request_irq(unsigned int irqn, irq_handler_t handler, unsigned long flags,
		const char *name, void *priv);

#endif
