#ifndef _CACHE_H
#define _CACHE_H

#define L1_CACHE_SHIFT		6
#define L1_CACHE_BYTES		(1 << L1_CACHE_SHIFT)

#define ICACHE  (1 << 0)
#define DCACHE  (1 << 1)

#define MCOR_IC_SEL  (1 << 0)
#define MCOR_DC_SEL  (1 << 1)
#define MCOR_INV_SEL (1 << 4)
#define MCOR_CLR_SEL (1 << 5)

void dcache_disable(void);
void dcache_enable(void);
void inv_dcache_all(void);
void clean_dcache_all(void);
void inv_clean_dcache_all(void);

void invalidate_dcache_range(unsigned long start, unsigned long stop);
void flush_dcache_range(unsigned long start, unsigned long stop);

#endif
