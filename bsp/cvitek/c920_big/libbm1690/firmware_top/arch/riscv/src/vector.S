#include "asm/csr.h"

	.section .vector, "ax"
	.global _start
_start:
        fence.i
        li x1, 0
        li x2, 0
        li x3, 0
        li x4, 0
        li x5, 0
        li x6, 0
        li x7, 0
        li x8, 0
        li x9, 0
        li x10, 0
        li x11, 0
        li x12, 0
        li x13, 0
        li x14, 0
        li x15, 0
        li x16, 0
        li x17, 0
        li x18, 0
        li x19, 0
        li x20, 0
        li x21, 0
        li x22, 0
        li x23, 0
        li x24, 0
        li x25, 0
        li x26, 0
        li x27, 0
        li x28, 0
        li x29, 0
        li x30, 0
        li x31, 0
        csrw mscratch, x0

        csrw mie, x0

        # write mtvec and make sure it sticks
        la t0, trap_vector
        csrw mtvec, t0

        # this setting is for RISCV C920
        # invalid I-cache
        li x3, 0x33
        csrc mcor, x3
        li x3, 0x11
        csrs mcor, x3
        # enable I-cache
        li x3, 0x1
        csrs mhcr, x3
        # invalid D-cache
        li x3, 0x33
        csrc mcor, x3
        li x3, 0x12
        csrs mcor, x3
        # enable D-cache
        li x3, 0x2
        csrs mhcr, x3

        # enable fp
        li x3, 0x1 << 13
        csrs mstatus, x3

        fmv.w.x f0,zero
        fmv.w.x f1,zero
        fmv.w.x f2,zero
        fmv.w.x f3,zero
        fmv.w.x f4,zero
        fmv.w.x f5,zero
        fmv.w.x f6,zero
        fmv.w.x f7,zero
        fmv.w.x f8,zero
        fmv.w.x f9,zero
        fmv.w.x f10,zero
        fmv.w.x f11,zero
        fmv.w.x f12,zero
        fmv.w.x f13,zero
        fmv.w.x f14,zero
        fmv.w.x f15,zero
        fmv.w.x f16,zero
        fmv.w.x f17,zero
        fmv.w.x f18,zero
        fmv.w.x f19,zero
        fmv.w.x f20,zero
        fmv.w.x f21,zero
        fmv.w.x f22,zero
        fmv.w.x f23,zero
        fmv.w.x f24,zero
        fmv.w.x f25,zero
        fmv.w.x f26,zero
        fmv.w.x f27,zero
        fmv.w.x f28,zero
        fmv.w.x f29,zero
        fmv.w.x f30,zero
        fmv.w.x f31,zero

        la sp, __ld_stack_top-8
        # double-word aligned in rom.lds
        la a2, __ld_data_load_start
        la a3, __ld_data_start
        la a4, __ld_data_size

        csrr t0, mhartid
        li  t1, 0
        beq  t0, t1, primary_core

primary_core:
        j core_init_done
        wfi

core_init_done:
	# jal clear_bss
	jal load_data
        jal system_init
        jal __libc_init_array
        jal main
        j die

trap_vector:
        j trap_entry
        # never reach here
        j die


#define LREG                    ld
#define SREG                    sd
#define REGBYTES                8
#define RELOC_TYPE              R_RISCV_64
#define SYM_INDEX               0x20
#define SYM_SIZE                0x18
/*
 * trap entry
 */
trap_entry:
        addi sp, sp, -35*REGBYTES
        SREG x1, 1*REGBYTES(sp)
        SREG x2, 2*REGBYTES(sp)
        SREG x3, 3*REGBYTES(sp)
        SREG x4, 4*REGBYTES(sp)
        SREG x5, 5*REGBYTES(sp)
        SREG x6, 6*REGBYTES(sp)
        SREG x7, 7*REGBYTES(sp)
        SREG x8, 8*REGBYTES(sp)
        SREG x9, 9*REGBYTES(sp)
        SREG x10, 10*REGBYTES(sp)
        SREG x11, 11*REGBYTES(sp)
        SREG x12, 12*REGBYTES(sp)
        SREG x13, 13*REGBYTES(sp)
        SREG x14, 14*REGBYTES(sp)
        SREG x15, 15*REGBYTES(sp)
        SREG x16, 16*REGBYTES(sp)
        SREG x17, 17*REGBYTES(sp)
        SREG x18, 18*REGBYTES(sp)
        SREG x19, 19*REGBYTES(sp)
        SREG x20, 20*REGBYTES(sp)
        SREG x21, 21*REGBYTES(sp)
        SREG x22, 22*REGBYTES(sp)
        SREG x23, 23*REGBYTES(sp)
        SREG x24, 24*REGBYTES(sp)
        SREG x25, 25*REGBYTES(sp)
        SREG x26, 26*REGBYTES(sp)
        SREG x27, 27*REGBYTES(sp)
        SREG x28, 28*REGBYTES(sp)
        SREG x29, 29*REGBYTES(sp)
        SREG x30, 30*REGBYTES(sp)
        SREG x31, 31*REGBYTES(sp)
        csrr a1, mepc
        SREG a1,   0*REGBYTES(sp)
        csrr t0, mstatus
        SREG t0,  32*REGBYTES(sp)
        csrr t1, mbadaddr
        SREG t1,  33*REGBYTES(sp)
        csrr a0, mcause
        SREG a0,  34*REGBYTES(sp)
        mv a2, sp
        jal handle_trap

	LREG t0, 32*REGBYTES(sp)
	LREG a1,  0*REGBYTES(sp)
        csrw mstatus, t0
        csrw mepc, a1

        LREG x1, 1*REGBYTES(sp)
        LREG x2, 2*REGBYTES(sp)
        LREG x3, 3*REGBYTES(sp)
        LREG x4, 4*REGBYTES(sp)
        LREG x5, 5*REGBYTES(sp)
        LREG x6, 6*REGBYTES(sp)
        LREG x7, 7*REGBYTES(sp)
        LREG x8, 8*REGBYTES(sp)
        LREG x9, 9*REGBYTES(sp)
        LREG x10, 10*REGBYTES(sp)
        LREG x11, 11*REGBYTES(sp)
        LREG x12, 12*REGBYTES(sp)
        LREG x13, 13*REGBYTES(sp)
        LREG x14, 14*REGBYTES(sp)
        LREG x15, 15*REGBYTES(sp)
        LREG x16, 16*REGBYTES(sp)
        LREG x17, 17*REGBYTES(sp)
        LREG x18, 18*REGBYTES(sp)
        LREG x19, 19*REGBYTES(sp)
        LREG x20, 20*REGBYTES(sp)
        LREG x21, 21*REGBYTES(sp)
        LREG x22, 22*REGBYTES(sp)
        LREG x23, 23*REGBYTES(sp)
        LREG x24, 24*REGBYTES(sp)
        LREG x25, 25*REGBYTES(sp)
        LREG x26, 26*REGBYTES(sp)
        LREG x27, 27*REGBYTES(sp)
        LREG x28, 28*REGBYTES(sp)
        LREG x29, 29*REGBYTES(sp)
        LREG x30, 30*REGBYTES(sp)
        LREG x31, 31*REGBYTES(sp)
        addi sp, sp, 35*REGBYTES
        mret

die:
        wfi
        j die

