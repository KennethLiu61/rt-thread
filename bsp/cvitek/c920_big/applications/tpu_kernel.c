// SPDX-License-Identifier: GPL-2.0

#include "runtime_tpu_kernel.h"

#include <pthread.h>
#include <stdio.h>

#include "api.h"
#include "sg_io.h"
#include "msgfifo.h"

#define C2C_INFO_OFFSET 0x1f3000
#define MAX_C2C_DEV_NUM 512

static int c2c_dev_num;
static void *c2c_info_base;

void set_scaler_scheduler_mode(int mode)
{
	cur_thread->sync_mode = mode;
}

void get_tpu_groupset_info(struct tpu_groupset_info *ptr_groupset_info)
{
	struct task_item *cur_task;

	pr_debug("I'm in %s\n", __func__);

	if (cur_thread->kernel_running)
		cur_task = (struct task_item *)(&cur_thread->task_list)->next->next;
	else
		cur_task = (struct task_item *)(&cur_thread->task_list)->next;

	ptr_groupset_info->tpu_id = cur_thread->tpu_id;
	ptr_groupset_info->group_num = cur_task->task.task_header.group_num;
	ptr_groupset_info->block_num = cur_task->task.task_header.block_num;
	ptr_groupset_info->group_id = cur_task->task.task_header.request_cc_info.group_id;
	ptr_groupset_info->block_id = cur_task->task.task_header.request_cc_info.block_id;
}
RTM_EXPORT(get_tpu_groupset_info);

void poll_cur_task_enable(void)
{
	struct task_item *cur_task;
	int ret;
	// todo: no trigger
	return;
}

void write_response(int result)
{
	struct task_response *task_response = (struct task_response *)malloc(sizeof(struct task_response));
	struct task_item *cur_task;

	pr_debug("I'm in %s\n", __func__);

	if (cur_thread->sync_mode == SYNC_MODE) {
		pr_err("%s: SYNC_MODE: should not call this func\n", __func__);
		return;
	}

	// first kernel
	if (!cur_thread->kernel_running) {
		cur_thread->kernel_running = 1;
		return;
	}

	cur_task = (struct task_item *)(&cur_thread->task_list)->next;
	if (!cur_task->need_response)
		return;

	task_response->task_id = cur_task->task.task_header.task_id;
	task_response->group_id = cur_task->task.task_header.request_cc_info.group_id;
	task_response->block_id = cur_task->task.task_header.request_cc_info.block_id;
	task_response->stream_id = cur_task->task.task_header.stream_id;
	task_response->start_time = cur_task->start_time;
	task_response->end_time = cur_task->end_time;
	task_response->result = result;

	pr_debug("%s: task_id: 0x%llx\n", __func__, task_response->task_id);
	pr_debug("%s: group_id: 0x%x\n", __func__, task_response->group_id);
	pr_debug("%s: block_id: 0x%x\n", __func__, task_response->block_id);
	pr_debug("%s: stream_id: 0x%llx\n", __func__, task_response->stream_id);

	msgfifo_finish_api(task_response);

	list_del(&cur_task->list);
	free(cur_task->task.task_body);
	free(cur_task);
	free(task_response);
}
RTM_EXPORT(write_response);

int get_c2c_dev_num(void)
{
	return c2c_dev_num;
}

/**
 * Initialize the C2C port information.
 *
 * This function allocates memory for the C2C port information,
 * reads the information from SRAM.
 *
 * @return 0 on success, -1 on failure.
 */
int c2c_info_init(void)
{
	int buf_size = sizeof(struct c2c_port_info);

	c2c_info_base = malloc(buf_size);
	if (c2c_info_base == NULL) {
		pr_debug("malloc c2c info buf failed\n");
		return -1;
	}

	/* try to get valid c2c_dev_num */
	for (int i = 0; i < MAX_C2C_DEV_NUM; i++) {
		sg_sram_read(C2C_INFO_OFFSET + i * buf_size, buf_size, c2c_info_base);
		c2c_dev_num = ((struct c2c_port_info *)c2c_info_base)->chip_num;
		if (c2c_dev_num < 1 || c2c_dev_num > MAX_C2C_DEV_NUM)
			pr_debug("try to get c2c_dev_num, index %d\n", i);
		else
			break;
	}

	if (c2c_dev_num < 1 || c2c_dev_num > MAX_C2C_DEV_NUM) {
		pr_debug("invalid c2c dev num\n");
		free(c2c_info_base);
		c2c_info_base = NULL;
		return -1;
	}

	buf_size = sizeof(struct c2c_port_info) * c2c_dev_num * c2c_dev_num;
	c2c_info_base = realloc(c2c_info_base, buf_size);
	if (c2c_info_base == NULL) {
		pr_debug("realloc c2c info buf failed\n");
		return -1;
	}
	sg_sram_read(C2C_INFO_OFFSET, buf_size, c2c_info_base);

	return 0;
}

int get_c2c_port(int myself_devid, int peer_devid, int direction)
{
	int ret = 0;

	/* lazy init */
	if (c2c_info_base == NULL) {
		ret = c2c_info_init();
		if (ret) {
			pr_debug("c2c info init failed\n");
			return -1;
		}
	}

	struct c2c_port_info *port = c2c_info_base;

	port = port + c2c_dev_num * myself_devid + peer_devid;
	if (direction == C2C_SEND)
		return (int)port->send_port;
	else if (direction == C2C_RECV)
		return (int)port->recv_port;

	pr_debug("error c2c direction:%d\n", direction);

	return -1;
}
