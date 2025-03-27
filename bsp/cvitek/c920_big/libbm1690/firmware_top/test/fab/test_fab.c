#include <stdio.h>
#include <timer.h>
#include <arch.h>
#include <cache.h>
#include <framework/module.h>
#include <framework/common.h>
#include <asm/csr.h>

#define READ_ONLY(addr, size, name) \
	{addr, size, -1, -1, NULL, name}
#define READ_CHECK(addr, size, expected, name) \
	{addr, size, -1, expected, NULL, name}
#define WRITE_CHECK(addr, size, name) \
	{addr, size, 0xdeadbeef, 0xdeadbeef, NULL, name}
#define WRITE_CHECK_VALUE(addr, size, value, name) \
	{addr, size, value, value, NULL, name}

enum {
	FAB_TEST_NT		= 0,
	FAB_TEST_PASS,
	FAB_TEST_FAIL,
};

struct region {
	unsigned long addr;
	uint32_t size;
	int32_t write;
	int32_t expected;
	int (*func)(struct region *region);
	char *name;
	int result;
};

struct region regions[] = {
	READ_ONLY(0x25000000, 0x4, "TPU0 local"),
	READ_ONLY(0x25400000, 0x4, "TPU1 local"),
	READ_ONLY(0x25800000, 0x4, "TPU0 static"),
	READ_ONLY(0x25810000, 0x4, "TPU1 static"),
	READ_ONLY(0x26050218, 0x4, "906b0 reset addr low"),
	READ_ONLY(0x2605021c, 0x4, "906b0 reset addr high"),
	READ_ONLY(0x26050220, 0x4, "906b1 reset addr low"),
	READ_ONLY(0x26050224, 0x4, "906b1 reset addr high"),
	READ_ONLY(0x28100000, 0x4, "TOP reg"),
	READ_ONLY(0x270f0000, 0x4, "Mailbox"),

	// check device lock property: 16 * 4KB
	// expected: READ 0 -> READ 1 -> WRITE 0 -> READ 0
	READ_CHECK(0x260501d8, 0x40, 0x0, "device lock read"),
	READ_CHECK(0x260501d8, 0x40, 0x1, "device lock read"),
	WRITE_CHECK_VALUE(0x260501d8, 0x40, 0x0, "device lock write"),

	WRITE_CHECK(0x25000000, 0x400000, "TPU0 local write"),
	WRITE_CHECK(0x25400000, 0x400000, "TPU1 local write"),
	WRITE_CHECK(0x25800000, 0x10000, "TPU0 static write"),
	WRITE_CHECK(0x25810000, 0x10000, "TPU1 static write"),

	READ_ONLY(0x26000000, 0x1000, "TPU0 reg read"),
	READ_ONLY(0x26010000, 0x1000, "TPU1 reg read"),
	READ_ONLY(0x26020000, 0x1000, "GDMA0 reg read"),
	READ_ONLY(0x26030000, 0x1000, "GDMA1 reg read"),
	READ_ONLY(0x26040000, 0x2000, "HAU reg read"),
	READ_ONLY(0x26050000, 0x1000, "TPUSYS reg read"),
	READ_ONLY(0x26060000, 0x1000, "MSG reg read"),
};

static inline const char *test_result(struct region *r)
{
	if (r->result == FAB_TEST_PASS)
		return "PASS";
	else if (r->result == FAB_TEST_FAIL)
		return "FAIL";
	else
		return "NOT-TESTED";
}

void show_region(struct region *r)
{
	pr_info("0x%08lx %8x",
			r->addr, r->size);
}

static int test_region(struct region *r)
{
	int err = 0;
	uint32_t value;

	show_region(r);

	for (int i = 0; i < r->size; i += sizeof(int)) {
		if (r->func)
			err |= r->func(r);

		if (r->write >= 0)
			writel(r->write, r->addr + i);

		value = readl(r->addr + i);
		if (i == 0)
			pr_info("  0x%08x", value);

		if (r->expected >= 0 && r->expected != value) {
			pr_err("\nexpected 0x%08x but get 0x%08x\n", r->expected, value);
			err |= 1;
		}

		if (err)
			break;
	}

	r->result = err ? FAB_TEST_FAIL : FAB_TEST_PASS;

	pr_info("  %25s    [%s]\n", r->name, test_result(r));

	return err;
}

enum {
	NONCACHEABLE,
	CACHEABLE,
};

struct perf_region {
	unsigned long addr;
	int gran;
	char *name;
	int cacheable;
};

struct perf_region perf_regions[] = {
	{0x180000000, 16, "DDR", CACHEABLE},
	{0x190000000, 4, "DDR", CACHEABLE},
	{0x25000000, 16, "TPU0 local 128", NONCACHEABLE},
	{0x25000000, 4, "TPU0 local 32", NONCACHEABLE},
	{0x25800000, 16, "TPU0 static 128", NONCACHEABLE},
	{0x25800000, 4, "TPU0 static 32", NONCACHEABLE},
	{0x26000100, 4, "TPU0 ctrl reg 32", NONCACHEABLE},
	{0x26020000, 4, "GDMA0 reg 32", NONCACHEABLE},
	{0x26050004, 4, "TPUSYS reg 32", NONCACHEABLE},
};

// cacheable region like DDR, read/write contiguous memory of size PERF_DEVICE_COUNT
// non-cacheable region, read/write PERF_DEVICE_COUNT times of same address
#define PERF_DEVICE_COUNT	(1 * 1024 * 1024)

int perf_device_region(struct perf_region *r)
{
	unsigned long ptr;
	unsigned long i = 0;
	uint64_t time;
	register uint32_t __attribute__((unused)) tmp32;
	register uint128_t __attribute__((unused)) tmp128;

	ptr = r->addr;

	pr_info("**************************************\n");
	pr_info("0x%08lx %16s\n",
			r->addr, r->name);

	time = timer_get_tick();

	if (r->cacheable) {
		if (r->gran == 4) {
			for (i = 0; i < PERF_DEVICE_COUNT; i += 4)
				*(volatile uint32_t *)(ptr + i) = 0xdeadbeefULL;
		} else if (r->gran == 16) {
			for (i = 0; i < PERF_DEVICE_COUNT; i += 16 * 32) {
				*(volatile uint128_t *)(ptr + i + 16 * 0) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 1) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 2) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 3) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 4) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 5) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 6) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 7) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 8) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 9) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 10) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 11) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 12) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 13) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 14) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 15) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 16) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 17) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 18) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 19) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 20) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 21) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 22) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 23) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 24) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 25) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 26) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 27) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 28) = 0xdeadbeefULL;
        			*(volatile uint128_t *)(ptr + i + 16 * 29) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 30) = 0xdeadbeefULL;
				*(volatile uint128_t *)(ptr + i + 16 * 31) = 0xdeadbeefULL;
                        }
		} else {
			pr_err("unsupported write granularity %d\n",
					r->gran);
			return -1;
		}
	} else {
		if (r->gran == 4) {
			for (i = 0; i < PERF_DEVICE_COUNT; ++i)
				*(volatile uint32_t *)ptr = 0xdeadbeefUL;
		} else if (r->gran == 16) {
			for (i = 0; i < PERF_DEVICE_COUNT; ++i)
				*(volatile uint128_t *)ptr = 0xdeadbeefUL;
		} else {
			pr_err("unsupported write granularity %d\n",
					r->gran);
			return -1;
		}
	}

	time = timer_get_tick() - time;
	time = timer_tick2us(time);

	if (r->cacheable) {
		pr_info("%ld times write%d in %ldus\n", i / r->gran, r->gran * 8, time);
		pr_info("%ld write-ops/s %ldB/s\n",
			i * 1000 * 1000 / (time * r->gran),
			i * 1000 * 1000 / time);
	} else {
		pr_info("%ld times write%d in %ldus\n", i, r->gran * 8, time);
		pr_info("%ld write-ops/s %ldB/s\n",
			i * 1000 * 1000 / time,
			i * 1000 * 1000 * r->gran / time);
	}

	time = timer_get_tick();

	if (r -> cacheable) {
		if (r->gran == 4) {
			for (i = 0; i < PERF_DEVICE_COUNT; i += 4)
				tmp32 = *(volatile uint32_t *)(ptr + i);
		} else if (r->gran == 16) {
                        for (i = 0; i < PERF_DEVICE_COUNT; i += 16 * 32) {
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 0);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 1);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 2);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 3);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 4);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 5);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 6);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 7);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 8);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 9);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 10);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 11);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 12);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 13);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 14);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 15);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 16);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 17);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 18);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 19);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 20);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 21);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 22);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 23);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 24);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 25);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 26);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 27);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 28);
        			tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 29);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 30);
				tmp128 = *(volatile uint128_t *)(ptr + i + 16 * 31);
                        }
		} else {
			pr_err("unsupported read granularity %d\n",
					r->gran);
			return -1;
		}
	} else {
		if (r->gran == 4) {
			for (i = 0; i < PERF_DEVICE_COUNT; ++i)
				tmp32 = *(volatile uint32_t *)ptr;
		} else if (r->gran == 16) {
			for (i = 0; i < PERF_DEVICE_COUNT; ++i)
				tmp128 = *(volatile uint128_t *)ptr;
		} else {
			pr_err("unsupported read granularity %d\n",
					r->gran);
			return -1;
		}
	}

	time = timer_get_tick() - time;
	time = timer_tick2us(time);

	if (r->cacheable) {
		pr_info("%ld times read%d in %ldus\n", i / r->gran, r->gran * 8, time);
		pr_info("%ld read-ops/s %ldB/s\n",
			i * 1000 * 1000 / (time * r->gran),
			i * 1000 * 1000 / time);	
	} else {
		pr_info("%ld times read%d in %ldus\n", i, r->gran * 8, time);
		pr_info("%ld read-ops/s %ldB/s\n",
			i * 1000 * 1000 / time,
			i * 1000 * 1000 * r->gran / time);		
	}
	pr_info("**************************************\n");

	return 0;
}

static int test_fab(void)
{
	int i;
	int mhartid = read_csr(mhartid);

	pr_info("I'm C906 %d.\n", mhartid - 1);

	pr_info("FAB Access Test\n\n");

	pr_info("enable clk of TPU C906 GDMA HAU\n");
	writel(readl(0x26050000) | 0xff, 0x26050000);

	pr_info("enable TPU0\n");
	writel(readl(0x26000100) | 1, 0x26000100);

	pr_info("enable TPU1\n");
	writel(readl(0x26010100) | 1, 0x26010100);

	pr_info("\n\nsysmap test\n\n");
	pr_info("%10s %8s  %10s  %25s  %8s\n",
		"addr", "size", "value", "name", "result");
	for (i = 0; i < ARRAY_SIZE(regions); ++i)
		test_region(&regions[i]);

	int mhcr = read_csr(mhcr);
	pr_info("mhcr: %08x\n\n", mhcr);

	pr_info("performance test with dcache enable\n\n");
	for (i = 0; i < ARRAY_SIZE(perf_regions); ++i)
		perf_device_region(&perf_regions[i]);

	pr_info("FAB Access Test Done\n\n");

	return 0;
}

test_case(test_fab);
