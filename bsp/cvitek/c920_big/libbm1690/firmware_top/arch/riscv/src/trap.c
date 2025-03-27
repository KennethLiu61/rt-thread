#include <arch.h>
#include <framework/common.h>
#include <framework/plic.h>
#include <ptrace.h>
#include <asm/csr.h>

// extern void do_irq(void);

static void exit_trap(int code, uint64_t epc, struct pt_regs *regs)
{
	static const char *exception_code[] = {
		"Instruction address misaligned", /*  0 */
		"Instruction access fault",       /*  1 */
		"Illegal instruction",            /*  2 */
		"Breakpoint",                     /*  3 */
		"Load address misaligned",        /*  4 */
		"Load access fault",              /*  5 */
		"Store/AMO address misaligned",   /*  6 */
		"Store/AMO access fault",         /*  7 */
		"Environment call from U-mode",   /*  8 */
		"Environment call from S-mode",   /*  9 */
		"Reserved",                       /* 10 */
		"Environment call from M-mode",   /* 11 */
		"Instruction page fault",         /* 12 */
		"Load page fault",                /* 13 */
		"Reserved",                       /* 14 */
		"Store/AMO page fault",           /* 15 */
		"Reserved",                       /* 16 */
	};

	pr_info("exception code: %d , %s , epc %08lx , ra %08lx, badaddr %08lx\n",
		code, exception_code[code], epc, regs->ra, regs->sbadaddr);

	if (code == 7)	/* Store/AMO access fault */
		while(1);
}

uint64_t handle_trap(uint64_t scause, uint64_t epc, struct pt_regs *regs)
{
	uint64_t is_int;

	pr_debug("handle_trap: scause=0x%08lx, epc=0x%08lx\n", scause, epc);
        is_int = (scause & MCAUSE_INT);
        if ((is_int) && ((scause & MCAUSE_CAUSE) == IRQ_M_EXT))
                do_irq();
        else
                exit_trap(scause, epc, regs);

        return epc;
}
