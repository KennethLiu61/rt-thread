#include "common.h"
#include "sg_io.h"
#include "memmap.h"
#include "mmio.h"
#include "tpu.h"
#include "msgfifo.h"

#define BM1690_TIMER_CUR_VAL	0x4
#define BM1690_TIMER_INT_STATUS	0xa8

int tp_status_init(struct thread_item *cur_thread)
{
	cur_thread->tp_status = (struct tp_status *)((uint64_t)(0x2000 + 0x1000 * sg_clint_read(CLINT_MHART_ID)));

	return 0;
}

struct thread_item *cur_thread;

uint64_t bm1690_sync_time(void)
{
	uint64_t timer_addr;
	uint32_t timer_cur_val;
	uint32_t int_status;

	timer_addr = 0x7040003000;

	while (1) {
		timer_cur_val = mmio_read_32(timer_addr + BM1690_TIMER_CUR_VAL);
		int_status = mmio_read_32(timer_addr + BM1690_TIMER_INT_STATUS);

		if (int_status == 0 && timer_cur_val <= 1000)
			break;

		if (int_status == 1)
			return 0;
	}

	return timer_cur_val;
}

void daemon_main(void* parameter)
{
	int ret;
	uint32_t status_back = 0x0;
	uint32_t last_val = 0x0;

	cur_thread = (struct thread_item *)malloc(sizeof(struct thread_item));
	memset(cur_thread, 0, sizeof(struct thread_item));

	INIT_LIST_HEAD(&cur_thread->load_lib_list);
	INIT_LIST_HEAD(&cur_thread->task_list);
	cur_thread->func_table = NULL;

	ret = vaddr_init();
	if (ret != 0) {
		pr_err("vaddr init error!\n");
		return;
	}

	cur_thread->tpu_id = sg_clint_read(CLINT_MHART_ID);
	pr_debug("tpu_id = 0x%x\n", cur_thread->tpu_id);
	tpu_init();

	cur_thread->kernel_running = 0;
	cur_thread->sync_mode = SYNC_MODE;

	// timer_init();

	status_back = gp_reg_read_idx(3 + cur_thread->tpu_id);
	last_val = (LAST_INI_REG_VAL) & ~(0xf << 28);
	last_val =  (last_val | (status_back & (0xf << 28)));
	gp_reg_write_idx(3 + cur_thread->tpu_id, last_val);

	uint64_t log_addr;
	uint64_t log_type;
	uint64_t log_memory_type;
	struct tp_config *config;
	uint64_t start_time;
	uint64_t apb_timer;
	void *share_memory;

	apb_timer = bm1690_sync_time();
	get_time(start_time);
	rt_kprintf("start_time = 0x%ld us\n", start_time);
	fix_log_time(start_time);

	config = (struct tp_config *)sg_get_base_addr();
	log_addr = (config->log_addr >> 4) << 4;
	log_type = config->log_addr & 0xf;
	log_memory_type = config->memory_type;
	pr_debug("log_addr[0x%lx], log_type[0x%lx], log_memory_type[0x%lx]\n",
			log_addr, log_type, log_memory_type);
	log_init(log_addr + sg_clint_read(CLINT_MHART_ID) * 0x8000000, 0x8000000, log_type, log_memory_type);
	share_memory = map_share_memory(config->share_memory_addr);
	pr_debug("share memory:0x%lx -> 0x%lx\n", config->share_memory_addr, share_memory);
	pr_debug("tpu core id: %d\n", cur_thread->tpu_id);
	pr_debug("apb timer val:0x%lx, arch timer:0x%lx\n", apb_timer, start_time);
	// tp_debug("share memory:0x%lx -> 0x%lx\n", config->share_memory_addr, share_memory);
	// tp_debug("tpu core id: %d\n", cur_thread->tpu_id);
	// tp_debug("apb timer val:0x%lx, arch timer:0x%lx\n", apb_timer, start_time);
	tp_status_init(cur_thread);

	while (1) {
		msgfifo_process();
		rt_thread_mdelay(1);
		// rt_thread_delay(100000);	//2000us	//实测居然是2s
	}

	free(cur_thread);
	return;
}

int tpu_daemon_run(void)
{
	rt_thread_t tid = RT_NULL;
	tid = rt_thread_create("tp_daemon",
	daemon_main, NULL,
	RT_MAIN_THREAD_STACK_SIZE,
	RT_MAIN_THREAD_PRIORITY, 20);
	if(tid != NULL)
		rt_thread_startup(tid);

	return 0;
}