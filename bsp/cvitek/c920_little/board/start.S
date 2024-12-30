/*
 * Copyright (c) 2006-2024, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2024/01/11     flyingcys    The first version
 */
#include "riscv-virt.h"
.org 0

	.section .vectors, "ax"
	.globl _start
	.type _start,@function
_start:
	.cfi_startproc
	.cfi_undefined ra
.option push
.option norelax
//	la  gp, __global_pointer$
.option pop
	// Continue primary hart
	csrr a0, mhartid
	li   a1, PRIM_HART
	bne  a0, a1, secondary

    fence.i
    fence iorw, iorw

    li  x1, 0
    li  x2, 0
    li  x3, 0
    li  x4, 0
    li  x5, 0
    li  x6, 0
    li  x7, 0
    li  x8, 0
    li  x9, 0
    li  x10, 0
    li  x11, 0
    li  x12, 0
    li  x13, 0
    li  x14, 0
    li  x15, 0
    li  x16, 0
    li  x17, 0
    li  x18, 0
    li  x19, 0
    li  x20, 0
    li  x21, 0
    li  x22, 0
    li  x23, 0
    li  x24, 0
    li  x25, 0
    li  x26, 0
    li  x27, 0
    li  x28, 0
    li  x29, 0
    li  x30, 0
    li  x31, 0		

	/*
	 * select L1I and L1D and invalidate them
	 * invalidate IBP data, BTB data and BHT data
	 */
	li	t0, 0x70013
	csrw	mcor, t0

	/*
	 * enable L1I and L1D
	 * write allocate, write back
	 * enable return stack
	 * enable branch prediction
	 * enable indirect branch predition
	 * enable write burst
	 * disable L0BTB to workround auipc bug
	 */
	li	t0, 0x1ff
	csrw	mhcr, t0

	/*
	 * enable L2, inclusive with L1D
	 * enable L2 TLB prefetch
	 * enable L2 instuction prefetch (3 cache line)
	 */
	li	t0, 0xe0410009
	csrw	mccr2, t0

	/*
	 * disable broadcast of fence.i, fence, and sfence.vma
	 * enable spec fail prediction
	 * enable L2 prefetch (32 cache line)
	 * enable L1D prefetch (16 cache line)
	 * enable L1I prefetch
	 * enable loop speed-up
	 * enable AMR (adaptive migration–replication?)
	 */
	li	t0, 0xe6e30c
	csrw	mhint, t0

	/*
	 * same as reset value, just in case.
	 * enable unaligned access
	 * enable cmo instructions
	 * enable CLINT S-mode software interrupt and timer interrupt
	 * enable MMU extension attributes
	 * enable C920 extension instructions
	 */
	li	t0, 0x638000
	csrw	mxstatus, t0
	
	# enable fp
	li x3, 0x1 << 13
	csrs mstatus, x3

	fence.i
	fence iorw, iorw

	/* Setup trap handler */
	csrw mie, 0
	csrw mip, 0
   	la t0, c920_trap_entry
 	csrw mtvec, t0

	// Primary hart
	la sp, _stack_top

	// Load data section
	la a0, _data_lma
	la a1, _data
	la a2, _edata
	bgeu a1, a2, 2f
1:
	LOAD t0, (a0)
	STOR t0, (a1)
	addi a0, a0, REGSIZE
	addi a1, a1, REGSIZE
	bltu a1, a2, 1b
2:

	// Clear bss section
	la a0, _bss
	la a1, _ebss
	bgeu a0, a1, 2f
1:
	// reduce branch time, be sure about bss alignment in linker script
	STOR zero, 0x00 (a0)
	STOR zero, 0x08 (a0)
	STOR zero, 0x10 (a0)
	STOR zero, 0x18 (a0)
	STOR zero, 0x20 (a0)
	STOR zero, 0x28 (a0)
	STOR zero, 0x30 (a0)
	STOR zero, 0x38 (a0)
	addi a0, a0, REGSIZE * 8
	bltu a0, a1, 1b
2:
	jal watchdog_stop

	// argc, argv, envp is 0
	li  a0, 0
	li  a1, 0
	li  a2, 0
	jal entry
1:
	wfi
	j 1b

	.align 4
c920_trap_entry:
	j trap_entry

secondary:
	// TODO: Multicore is not supported
	wfi
	j secondary
	.cfi_endproc