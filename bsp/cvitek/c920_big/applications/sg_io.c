//cache ops head
#include "rthw.h"
//reg ops head
#include "mmio.h"
//sg common head
#include "common.h"
//sg memory head
#include "memmap.h"

#include "api.h"

uint64_t g_drv_ddr_handle;
uint64_t g_drv_top_handle;
uint64_t g_drv_tpusys_handle;
uint64_t g_drv_vcsys_handle;
uint64_t g_drv_shmem_handle;
uint64_t g_drv_mtli_handle;
uint64_t g_drv_clint_handle;
uint64_t g_drv_c2c_sys0_handle;
uint64_t g_drv_c2c_sys1_handle;
uint64_t g_drv_cxp_sys_handle;
uint64_t g_drv_sram_handle;

int vaddr_init(void)
{
	pr_debug("I'm in %s\n", __func__);

	g_drv_ddr_handle = (uint64_t )DDR_BASE_ADDR_PHYS;
	g_drv_top_handle = (uint64_t )TOP_BASE_ADDR_PHYS;
	g_drv_tpusys_handle = TPU_SYS_BASE_ADDR_PHYS;
	g_drv_vcsys_handle = VC_SYS_BASE_ADDR_PHYS;
    //msg fifo
	// g_drv_shmem_handle = SHMEM_BASE_ADDR_PHYS;
	g_drv_mtli_handle = AP_PLIC_MTLI_CFG;
	g_drv_clint_handle = CLINT_BASE_ADDR;
	g_drv_c2c_sys0_handle = C2C_SYS0_BASE_ADDR_PHYS;
	g_drv_c2c_sys1_handle = C2C_SYS1_BASE_ADDR_PHYS;
	g_drv_cxp_sys_handle = CXP_SYS_BASE_ADDR_PHYS;
	g_drv_sram_handle = SRAM_BASE_ADDR_PHYS;

	pr_debug("%s success!\n", __func__);
	return 0;
}

void vaddr_destroy(void)
{
}

void *sg_get_base_addr(void)
{
	return (void *)g_drv_ddr_handle;
}

void *map_share_memory(uint64_t pa)
{
	g_drv_shmem_handle = pa;

	return (void *)g_drv_shmem_handle;
}

void *map_to_kaddr(unsigned long long addr)
{
	if ((addr >= DDR_BASE_ADDR_PHYS) && (addr < (DDR_BASE_ADDR_PHYS + MMAP_DDR_LEN)))
		return (void *)(g_drv_ddr_handle + addr);
	else if ((addr >= TPU_SYS_BASE_ADDR_PHYS) && (addr < (TPU_SYS_BASE_ADDR_PHYS + MMAP_TPU_SYS_LEN)))
		return (void *)(g_drv_tpusys_handle + addr - TPU_SYS_BASE_ADDR_PHYS);
	else if ((addr >= VC_SYS_BASE_ADDR_PHYS) && (addr < (VC_SYS_BASE_ADDR_PHYS + MMAP_VC_SYS_LEN)))
		return (void *)(g_drv_vcsys_handle + addr - VC_SYS_BASE_ADDR_PHYS);
	else if ((addr >= C2C_SYS0_BASE_ADDR_PHYS) && (addr < (C2C_SYS0_BASE_ADDR_PHYS + MMAP_C2C_SYS_LEN)))
		return (void *)(g_drv_c2c_sys0_handle + addr - C2C_SYS0_BASE_ADDR_PHYS);
	else if ((addr >= C2C_SYS1_BASE_ADDR_PHYS) && (addr < (C2C_SYS1_BASE_ADDR_PHYS + MMAP_C2C_SYS_LEN)))
		return (void *)(g_drv_c2c_sys1_handle + addr - C2C_SYS1_BASE_ADDR_PHYS);
	else if ((addr >= CXP_SYS_BASE_ADDR_PHYS) && (addr < (CXP_SYS_BASE_ADDR_PHYS + MMAP_CXP_SYS_LEN)))
		return (void *)(g_drv_cxp_sys_handle + addr - CXP_SYS_BASE_ADDR_PHYS);

	pr_err("%s error: invalid addr: 0x%llx\n", __func__, addr);
	return (void *)addr;
}
RTM_EXPORT(map_to_kaddr);


int sg_msgfifo_update_ptr(uint32_t cur_ptr)
{
	return cur_ptr % MSG_FIFO_SIZE;
}

uint32_t sg_read(uint64_t addr)
{
    return mmio_read_32(g_drv_ddr_handle + addr);
}

void sg_write(uint64_t addr, uint32_t val)
{
    mmio_write_32(g_drv_ddr_handle + addr, val);
}

uint32_t gp_reg_read_idx(uint32_t idx)
{
    return mmio_read_32(g_drv_top_handle + GP_REG_BASE_ADDR_OFFSET + idx * sizeof(uint32_t));
}

void gp_reg_write_idx(uint32_t idx, uint32_t val)
{
	mmio_write_32(g_drv_top_handle + GP_REG_BASE_ADDR_OFFSET + idx * sizeof(uint32_t), val);
}

uint32_t sg_tpu_sys_read(uint64_t addr)
{
	return mmio_read_32(g_drv_tpusys_handle + addr - TPU_SYS_BASE_ADDR_PHYS);
}
RTM_EXPORT(sg_tpu_sys_read);

void sg_tpu_sys_write(uint64_t addr, uint32_t val)
{
	mmio_write_32(g_drv_tpusys_handle + addr, val - TPU_SYS_BASE_ADDR_PHYS);
}
RTM_EXPORT(sg_tpu_sys_write);

uint32_t sg_shmem_read(uint64_t addr)
{
    // rt_hw_cpu_dcache_ops(RT_HW_CACHE_INVALIDATE, (void *)(g_drv_shmem_handle + addr), sizeof(uint64_t));
	invalidate_dcache_range((void *)(g_drv_shmem_handle + addr), sizeof(uint64_t));
	return mmio_read_32(g_drv_shmem_handle + addr);
}

void sg_shmem_write(uint64_t addr, uint32_t val)
{
	mmio_write_32(g_drv_shmem_handle + addr, val);
	// rt_hw_cpu_dcache_ops(RT_HW_CACHE_FLUSH, (void *)(g_drv_shmem_handle + addr), sizeof(uint64_t));
	clean_dcache_range((void *)(g_drv_shmem_handle + addr), sizeof(uint64_t));
}

uint32_t sg_msgfifo_rx_read(uint32_t offset)
{
	return sg_shmem_read(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset));
}

void sg_msgfifo_rx_write(uint32_t offset, uint32_t val)
{
	sg_shmem_write(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset), val);
}

uint64_t sg_msgfifo_rx_read_64(uint32_t offset)
{
	uint32_t high, low;

	low = sg_shmem_read(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset));
	high = sg_shmem_read(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset + sizeof(uint32_t)));

	return ((uint64_t)high << 32) | low;
}

void sg_msgfifo_rx_write_64(uint32_t offset, uint64_t val)
{
	uint32_t high, low;

	low = val & ((1ull << 32) - 1);
	high = val >> 32;

	sg_shmem_write(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset), low);
	sg_shmem_write(MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset + sizeof(uint32_t)), high);
}

uint32_t sg_msgfifo_tx_read(uint32_t offset)
{
	return sg_shmem_read(MSG_FIFO_TX_BASE_OFFSET + sg_msgfifo_update_ptr(offset));
}

void sg_msgfifo_tx_write(uint32_t offset, uint32_t val)
{
	sg_shmem_write(MSG_FIFO_TX_BASE_OFFSET + sg_msgfifo_update_ptr(offset), val);
}

void sg_msgfifo_rx_read_bytes(uint32_t offset, uint8_t *buf, uint32_t len)
{
	offset = sg_msgfifo_update_ptr(offset);

	uint64_t addr = MSG_FIFO_RX_BASE_OFFSET + sg_msgfifo_update_ptr(offset);
	uint32_t left_sized = MSG_FIFO_SIZE - offset;

	if (left_sized >= len) {
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_INVALIDATE, (void *)(g_drv_shmem_handle + addr), len);
		invalidate_dcache_range((void *)(g_drv_shmem_handle + addr), len);
		memcpy(buf, (void *)(g_drv_shmem_handle + addr), len);
	} else {
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_INVALIDATE, (void *)(g_drv_shmem_handle + addr), left_sized);
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_INVALIDATE,
		// 					(void *)(g_drv_shmem_handle + MSG_FIFO_RX_BASE_OFFSET), (len - left_sized));
		invalidate_dcache_range((void *)(g_drv_shmem_handle + addr), left_sized);
		invalidate_dcache_range((void *)(g_drv_shmem_handle + MSG_FIFO_RX_BASE_OFFSET), (len - left_sized));
		memcpy(buf, (void *)(g_drv_shmem_handle + addr), left_sized);
		memcpy(buf + left_sized, (void *)(g_drv_shmem_handle + MSG_FIFO_RX_BASE_OFFSET), len - left_sized);
	}
}

void sg_msgfifo_tx_response(struct task_response *task_response)
{
	uint32_t current_wp = sg_shmem_read(MSG_FIFO_TX_WP_OFFSET);
	uint32_t left_size = MSG_FIFO_SIZE - current_wp;
	uint64_t addr = MSG_FIFO_TX_BASE_OFFSET + current_wp;

	if (left_size >= sizeof(struct task_response)) {
		*(struct task_response *)(g_drv_shmem_handle + addr) = *task_response;
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_FLUSH, (void *)(g_drv_shmem_handle + addr), sizeof(struct task_response));
		clean_dcache_range((void *)(g_drv_shmem_handle + addr), sizeof(struct task_response));
		asm volatile("fence iorw, iorw" ::);
		sg_shmem_write(MSG_FIFO_TX_WP_OFFSET, current_wp + sizeof(struct task_response));
	} else {
		memcpy((void *)(g_drv_shmem_handle + addr), task_response, left_size);
		memcpy((void *)(g_drv_shmem_handle + MSG_FIFO_TX_BASE_OFFSET),
			(uint8_t *)task_response + left_size,
			sizeof(struct task_response) - left_size);
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_FLUSH, (void *)(g_drv_shmem_handle + addr), left_size);
		// rt_hw_cpu_dcache_ops(RT_HW_CACHE_FLUSH, (void *)(g_drv_shmem_handle + MSG_FIFO_TX_BASE_OFFSET),
		// 					sizeof(struct task_response) - left_size);
		clean_dcache_range((void *)(g_drv_shmem_handle + addr), left_size);
		clean_dcache_range((void *)(g_drv_shmem_handle + MSG_FIFO_TX_BASE_OFFSET),
							sizeof(struct task_response) - left_size);
		asm volatile("fence iorw, iorw" ::);
		sg_shmem_write(MSG_FIFO_TX_WP_OFFSET,
			current_wp + sizeof(struct task_response) - MSG_FIFO_SIZE);
	}
	pr_debug("update TX head to: 0x%x\n", sg_shmem_read(MSG_FIFO_TX_WP_OFFSET));
}

uint32_t sg_clint_read(uint64_t addr)
{
	return mmio_read_32(g_drv_clint_handle + addr);
}

void sg_sram_read(uint32_t offset, uint32_t size, uint8_t *buf)
{
	memcpy(buf, (void *)(g_drv_sram_handle + offset), size);
}

