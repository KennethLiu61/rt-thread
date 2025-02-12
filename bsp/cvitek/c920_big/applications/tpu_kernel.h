/* SPDX-License-Identifier: GPL-2.0 */

#ifndef __TPU_KERNEL_H__
#define __TPU_KERNEL_H__

#include <stdint.h>

struct tpu_groupset_info {
	uint32_t tpu_id;
	uint64_t group_num;
	uint64_t block_num;
	uint32_t group_id;
	uint32_t block_id;
} __attribute__((packed));

void set_scaler_scheduler_mode(int mode);
void get_tpu_groupset_info(struct tpu_groupset_info *ptr_groupset_info);
void poll_cur_task_enable(void);
void write_response(int result);

//for C2C
enum {
	C2C_SEND = 0,
	C2C_RECV,
};

struct c2c_port_info {
	uint32_t chip_num;
	uint16_t src_device_id;
	uint16_t dst_device_id;
	uint8_t src_pcie_id;
	uint8_t dst_pcie_id;
	int8_t send_port;
	int8_t recv_port;
};

int c2c_info_init(void);
int get_c2c_dev_num(void);
int get_c2c_port(int myself_devid, int peer_devid, int direction);

#endif
