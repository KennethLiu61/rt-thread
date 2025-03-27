#include <cache.h>

void dcache_disable(void)
{
    	asm volatile("csrc mhcr, %0"::"rk"(DCACHE));
}

void dcache_enable(void)
{
    	asm volatile("csrs mhcr, %0"::"rK"(DCACHE));
}

void inv_dcache_all(void)
{
    	asm volatile("csrs mcor, %0"::"rk"(MCOR_DC_SEL | MCOR_INV_SEL));
}

void clean_dcache_all(void)
{
    	asm volatile("csrs mcor, %0"::"rk"(MCOR_DC_SEL | MCOR_CLR_SEL));
}

void inv_clean_dcache_all(void)
{
    	asm volatile("csrs mcor, %0"::"rk"(MCOR_DC_SEL | MCOR_INV_SEL | MCOR_CLR_SEL));
}


void invalidate_dcache_range(unsigned long start, unsigned long stop)
{
	register unsigned long i asm("a0") = start & ~(L1_CACHE_BYTES - 1);

	for (; i < stop; i += L1_CACHE_BYTES)
		asm volatile ("dcache.cipa a0");

	asm volatile ("sync.is");
}

void flush_dcache_range(unsigned long start, unsigned long stop)
{
	register unsigned long i asm("a0") = start & ~(L1_CACHE_BYTES - 1);

	for (; i < stop; i += L1_CACHE_BYTES)
		asm volatile ("dcache.cpa a0");

	asm volatile ("sync.is");
}
