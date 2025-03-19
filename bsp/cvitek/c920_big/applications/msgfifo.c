// SPDX-License-Identifier: GPL-2.0

#include <stdio.h>
#include <unistd.h>

#include "sg_io.h"
#include "msp_list.h"
#include "common.h"
#include "memmap.h"
#include "runtime_tpu_kernel.h"

int initial_rp;

void msgfifo_init(void)
{
	sg_shmem_write(MSG_FIFO_RX_RP_OFFSET, 0);
	sg_shmem_write(MSG_FIFO_RX_WP_OFFSET, 0);

	initial_rp = 0;
}

int msgfifo_empty(void)
{
	uint64_t cur_time;
	uint32_t rp, wp;

	rp = sg_shmem_read(MSG_FIFO_RX_RP_OFFSET);
	wp = sg_shmem_read(MSG_FIFO_RX_WP_OFFSET);

	if (rp == wp) {
		cur_thread->read_loop++;
		if (cur_thread->read_loop & 0xffff) {
			get_time(cur_time);
			cur_thread->tp_status->tp_heart_beat = cur_time;
		}
		return 1;
	}

	return 0;
}

void send_msi_to_host(void)
{
	uint64_t mtli_offset = (TPU_CORE_ID + 1) * 0x4;
	uint32_t *msi_addr = (uint32_t *)(g_drv_mtli_handle + mtli_offset);

	//pr_debug("I'm in %s\n", __func__);
	*msi_addr = 1;
}

void msgfifo_rx_update(uint32_t msg_size)
{
	uint32_t rp;

	// update rx rp
	rp = sg_shmem_read(MSG_FIFO_RX_RP_OFFSET);
	sg_shmem_write(MSG_FIFO_RX_RP_OFFSET, sg_msgfifo_update_ptr(rp + msg_size));
	pr_debug("After wp/head is %d\n", sg_shmem_read(MSG_FIFO_RX_WP_OFFSET));
	pr_debug("After rp/tail is %d\n", sg_shmem_read(MSG_FIFO_RX_RP_OFFSET));

	// update initial_rp
	initial_rp = sg_msgfifo_update_ptr(initial_rp + msg_size);
	pr_debug("update initial_rp to %d\n", initial_rp);
}

void msgfifo_read_task_header(struct task_header *task_header)
{
	pr_debug("I'm in %s\n", __func__);
	pr_debug("initial_rp now is %d\n", initial_rp);
	sg_msgfifo_rx_read_bytes(initial_rp, (uint8_t *)task_header, sizeof(struct task_header));
}

void msgfifo_read_task_body(char *task_body)
{
	struct api_header *ptr_api_header = (struct api_header *)task_body;
	char *ptr_payload = task_body + sizeof(struct api_header);
	int offset;

	pr_debug("I'm in %s\n", __func__);

	offset = sizeof(struct task_header) + offsetof(struct api_header, api_id);
	ptr_api_header->api_id = (enum API_ID)sg_msgfifo_rx_read(initial_rp + offset);

	offset = sizeof(struct task_header) + offsetof(struct api_header, api_size);
	ptr_api_header->api_size = sg_msgfifo_rx_read(initial_rp + offset);

	offset = sizeof(struct task_header) + sizeof(struct api_header);
	sg_msgfifo_rx_read_bytes(initial_rp + offset, (uint8_t *)ptr_payload, ptr_api_header->api_size * sizeof(uint32_t));
}

int msgfifo_read_task(void)
{
	struct list_head *pos_task;
	struct task_header task_header;
	int msg_size;
	int find = 0;

	while (!msgfifo_empty()) {
		pr_debug("begin read\n");
		get_time(cur_thread->start_time);
		msgfifo_read_task_header(&task_header);
		pr_debug("read task id:0x%lx\n", task_header.task_id);
		if (task_header.task_type == TRIGGER_TASK) {
			list_for_each(pos_task, &cur_thread->task_list) {
				struct task_item *ptr_task_item = (struct task_item *)pos_task;

				if (ptr_task_item->task.task_header.task_id == task_header.task_id &&
					ptr_task_item->task.task_header.request_cc_info.group_id == task_header.request_cc_info.group_id) {
					ptr_task_item->task_enable = 1;
					pr_debug("TRIGGER_TASK: task id found: 0x%llx\n", task_header.task_id);
					find = 1;
					break;
				}
			}
			if (!find) {
				pr_err("TRIGGER_TASK: task id not found: 0x%llx\n", task_header.task_id);
				return -1;
			}
			// update rx rp
			msg_size = sizeof(struct task_header);
			msgfifo_rx_update(msg_size);
		} else if (task_header.task_type == LAUNCH_KERNEL || task_header.task_type == SYNC_TASK ||
			   task_header.task_type == POLL_ENGINE_DONE) {
			struct task_item *ptr_task_item = (struct task_item *)malloc(sizeof(struct task_item));

			if (!ptr_task_item) {
				pr_err("failed to malloc for ptr_task_item size 0x%lx task_id:0x%llx\n",
						sizeof(struct task_item), task_header.task_id);
				return -1;
			}

			if (task_header.task_body_size) {
				ptr_task_item->task.task_body = (char *)malloc(task_header.task_body_size);
				if (!ptr_task_item->task.task_body) {
					pr_err("failed to malloc for task_body_size 0x%llx task_id:0x%llx\n",
							task_header.task_body_size, task_header.task_id);
					free(ptr_task_item);
					return -1;
				}

				msgfifo_read_task_body(ptr_task_item->task.task_body);
			}

			ptr_task_item->task.task_header = task_header;
			ptr_task_item->task_enable = 0;
			ptr_task_item->need_response = 0;
			list_append(&ptr_task_item->list, &cur_thread->task_list);

			// update rx rp
			msg_size = sizeof(struct task_header) + task_header.task_body_size;
			msgfifo_rx_update(msg_size);

			return 0; // only read one task
		} else {
			pr_err("unknown task type:0x%x,task_id:0x%llx\n", task_header.task_type, task_header.task_id);
			return -1;
		}
	}

	return 0;
}

void msgfifo_finish_api(struct task_response *task_response)
{
	pr_debug("I'm in %s\n", __func__);

	// write back tx
	sg_msgfifo_tx_response(task_response);
	asm volatile("fence iorw, iorw" ::);
	// interrupt host
	send_msi_to_host();
}

void msgfifo_task_handle(struct task_item *task_item)
{
	struct task_response task_response = {0};
	struct api_header *ptr_api_header;
	int ret = 0;
	int sync_id;
	uint64_t start_time;
	uint64_t end_time;

	if (task_item->task.task_header.task_type == SYNC_TASK) {
		sync_id = (int)task_item->task.task_header.stream_id;
		if (cur_thread->task_barrier) {
			tp_debug("[barrier]task:0x%lx barrier, sync id:%d, block num:0x%llx\n",
				task_item->task.task_header.task_id, sync_id,
				task_item->task.task_header.block_num);
			get_time(start_time);
			cur_thread->task_barrier(sync_id, task_item->task.task_header.block_num);
			get_time(end_time);
			cur_thread->kernel_exec_time += (end_time - start_time);
			cur_thread->tp_status->kernel_exec_time = cur_thread->kernel_exec_time;
			tp_debug("barrier exit\n");
		} else {
			tp_debug("[barrier]task:0x%lx barrier, sync id:%d, block num:0x%llx, no function\n",
				task_item->task.task_header.task_id, sync_id,
				task_item->task.task_header.block_num);
		}

		list_del(&task_item->list);
		free(task_item);

		return;
	}

	if (task_item->task.task_header.task_type == POLL_ENGINE_DONE) {
		if (cur_thread->poll_engine_done) {
			tp_debug("[poll]task:0x%lx, %u/%lu\n",
				task_item->task.task_header.task_id,
				task_item->task.task_header.request_cc_info.block_id,
				task_item->task.task_header.block_num);
			get_time(start_time);
			cur_thread->poll_engine_done();
			get_time(end_time);
			cur_thread->kernel_exec_time += (end_time - start_time);
			cur_thread->tp_status->kernel_exec_time = cur_thread->kernel_exec_time;
			tp_debug("poll done\n");
		} else {
			tp_debug("[poll]task:0x%lx, %u/%lu, no function\n",
				task_item->task.task_header.task_id,
				task_item->task.task_header.request_cc_info.block_id,
				task_item->task.task_header.block_num);
		}

		task_response.task_id = task_item->task.task_header.task_id;
		task_response.group_id = task_item->task.task_header.request_cc_info.group_id;
		task_response.block_id = task_item->task.task_header.request_cc_info.block_id;
		task_response.stream_id = task_item->task.task_header.stream_id;
		task_response.result = 0;

		msgfifo_finish_api(&task_response);

		list_del(&task_item->list);
		free(task_item);

		return;
	}

	pr_debug("%s: task_id: 0x%llx\n", __func__, task_item->task.task_header.task_id);
	pr_debug("%s: group_num: 0x%llx\n", __func__, task_item->task.task_header.group_num);
	pr_debug("%s: block_num: 0x%llx\n", __func__, task_item->task.task_header.block_num);
	pr_debug("%s: group_id: 0x%x\n", __func__, task_item->task.task_header.request_cc_info.group_id);
	pr_debug("%s: block_id: 0x%x\n", __func__, task_item->task.task_header.request_cc_info.block_id);
	pr_debug("%s: stream_id: 0x%llx\n", __func__, task_item->task.task_header.stream_id);
	pr_debug("%s: task_body_size: 0x%llx\n", __func__, task_item->task.task_header.task_body_size);

	pr_debug("exec task:0x%lx\n", task_item->task.task_header.task_id);

	task_response.task_id = task_item->task.task_header.task_id;
	task_response.group_id = task_item->task.task_header.request_cc_info.group_id;
	task_response.block_id = task_item->task.task_header.request_cc_info.block_id;
	task_response.stream_id = task_item->task.task_header.stream_id;

	ptr_api_header = (struct api_header *)task_item->task.task_body;

	task_response.start_time = time_simple();

	switch (ptr_api_header->api_id) {
	case API_ID_LOAD_LIB:
		ret = load_lib_process(task_item);
		break;

	case API_ID_LAUNCH_FUNC:
		ret = launch_func_process(task_item);
		break;

	case API_ID_UNLOAD_LIB:
		ret = unload_lib_process(task_item);
		break;
	default:
		pr_debug("unknown api id: 0x%x\n", ptr_api_header->api_id);
		ret = -1;
		break;
	}
	task_response.end_time = time_simple();
	task_response.result = ret;

	if ((cur_thread->sync_mode == ASYNC_MODE) && (ptr_api_header->api_id == API_ID_LAUNCH_FUNC)) {
		task_item->start_time = task_response.start_time;
		task_item->end_time = task_response.end_time;
		task_item->need_response = 1;
	} else {
		msgfifo_finish_api(&task_response);

		list_del(&task_item->list);
		free(task_item->task.task_body);
		free(task_item);
	}
}

void msgfifo_process(void)
{
	struct list_head *pos_task, *tmp;
	int ret;
	int flag = 0;

	if (list_empty(&cur_thread->task_list)) {
		ret = msgfifo_read_task();
		if (ret) {
			pr_err("msgfifo_read_task failed\n");
			return;
		}
	}

	list_for_each_safe(pos_task, tmp, &cur_thread->task_list) {
		msgfifo_task_handle((struct task_item *)pos_task);
		flag = 1;
		get_time(cur_thread->end_time);
		cur_thread->tp_alive_time += (cur_thread->end_time - cur_thread->start_time);
		cur_thread->tp_status->tp_alive_time = cur_thread->tp_alive_time;
	}
	if (flag)
		pr_debug("list empty\n");

	run_empty_kernel();
}