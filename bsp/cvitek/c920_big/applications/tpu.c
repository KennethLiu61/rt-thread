// SPDX-License-Identifier: GPL-2.0

#include "sg_io.h"
#include "tpu.h"
#include "memmap.h"

static void enable_clk(int clk_id)
{
	uint32_t clk = 0;

	clk = sg_tpu_sys_read(TPU_SYS_CLK);
	sg_tpu_sys_write(TPU_SYS_CLK, clk | clk_id);
}

static void enable_l2m(void)
{
	uint64_t addr;
	uint32_t reg;

	addr = TPU_SYS_REG_BASE_ADDR + TPU_SYS_L2M_CFG_CTRL_OFFSET;
	reg = sg_tpu_sys_read(addr);
	sg_tpu_sys_write(addr, reg | 0x1);

	addr = TPU_SYS_REG_BASE_ADDR + TPU_SYS_HNF_L2M_CTRL_OFFSET;
	reg = sg_tpu_sys_read(addr);
	sg_tpu_sys_write(addr, reg | 0xe);

	reg = sg_tpu_sys_read(TPU_SYS_RESET);
	sg_tpu_sys_write(TPU_SYS_RESET, reg & 0x0);
}

static void enable_tpu(void)
{
	uint32_t value;

	value = sg_tpu_sys_read(BD_ENGINE_MAIN_CTRL_AHB);
	sg_tpu_sys_write(BD_ENGINE_MAIN_CTRL_AHB, value | 0x1);
}

static void enable_cdma(void)
{
}

void tpu_init(void)
{
	enable_clk(0x7f | (1 << 20));
	enable_l2m();
	enable_tpu();
	enable_cdma();
}
