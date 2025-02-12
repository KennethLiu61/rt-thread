/* SPDX-License-Identifier: GPL-2.0 */

#ifndef __IO_H__
#define __IO_H__

#include "api.h"

extern uint64_t g_drv_ddr_handle;
extern uint64_t g_drv_top_handle;
extern uint64_t g_drv_tpusys_handle;
extern uint64_t g_drv_shmem_handle;
extern uint64_t g_drv_mtli_handle;
extern uint64_t g_drv_clint_handle;
extern uint64_t g_drv_c2c0_cdma_handle;
extern uint64_t g_drv_c2c1_cdma_handle;

int vaddr_init(void);
void vaddr_destroy(void);

void *sg_get_base_addr(void);
void *map_share_memory(uint64_t pa);

uint32_t sg_read(unsigned long long addr);
void sg_write(unsigned long long addr, unsigned int val);

uint32_t gp_reg_read_idx(uint32_t idx);
void gp_reg_write_idx(uint32_t idx, uint32_t val);

uint32_t sg_tpu_sys_read(uint64_t addr);
void sg_tpu_sys_write(uint64_t addr, uint32_t val);

uint32_t sg_shmem_read(uint64_t addr);
void sg_shmem_write(uint64_t addr, uint32_t val);

int sg_msgfifo_update_ptr(uint32_t cur_ptr);
uint32_t sg_msgfifo_rx_read(uint32_t offset);
void sg_msgfifo_rx_write(uint32_t offset, uint32_t val);
uint64_t sg_msgfifo_rx_read_64(uint32_t offset);
void sg_msgfifo_rx_read_bytes(uint32_t offset, uint8_t *buf, uint32_t len);
void sg_msgfifo_rx_write_64(uint32_t offset, uint64_t val);
uint32_t sg_msgfifo_tx_read(uint32_t offset);
void sg_msgfifo_tx_write(uint32_t offset, uint32_t val);
void sg_msgfifo_tx_response(struct task_response *task_response);

uint32_t sg_clint_read(uint64_t addr);
void sg_sram_read(uint32_t offset, uint32_t size, uint8_t *buf);

#endif