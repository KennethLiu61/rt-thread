.text
.balign 4
.global rvv_memcpy
# void *rvv_memcpy(void* dest, const void* src, size_t n)
# a0=dest, a1=src, a2=n
#
#if defined(__riscv_v) && (__riscv_v >= 1000000)
# define VLDB vle8.v
# define VSTB vse8.v
#else
# define VLDB vlb.v
# define VSTB vsb.v
#endif
rvv_memcpy:
	mv	a3, a0
	sltiu	a4, a2, 16
	bnez	a4, .loop_cpy
	andi	a5, a0, 15
	li	a6, 16
	beqz	a5, .loop_cpy
	sub	a5, a6, a5
	vsetvli	t0, a5, e8, m4
	VLDB	v0, (a1)
	add	a1, a1, t0
	sub	a2, a2, t0
	VSTB	v0, (a3)
	add	a3, a3, t0
.loop_cpy:
	vsetvli	t0, a2, e8, m4
	VLDB	v0, (a1)
	add	a1, a1, t0
	sub	a2, a2, t0
	VSTB	v0, (a3)
	add	a3, a3, t0
	bnez	a2, .loop_cpy
	ret
