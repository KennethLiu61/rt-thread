// SPDX-License-Identifier: GPL-2.0
#define _XOPEN_SOURCE 500

#include <stdio.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <ftw.h>

#include "api.h"
#include "sg_io.h"
#include "runtime_tpu_kernel.h"
#include "common.h"

static int removefn(const char *pathname, const struct stat *sbuf, int type, struct FTW *ftwb)
{
	int ret;

	ret = remove(pathname);
	if (ret != 0)
		pr_err("failed to remove %s!\n", pathname);
	return ret;
}

static char *workdir(void)
{
	return "/tmp";
}

#define sprintfcat(var, ...) snprintf(var + strlen(var), sizeof(var) - strlen(var), __VA_ARGS__)

int find_sym_by_name(struct lib_info *lib, unsigned char name[], char **pfunc)
{
	void *tmp;
	struct func_record func_record, *p_func_record;

	pr_debug("%s: func_name: %s\n", __func__, name);

	asm_memset(&func_record, 0, sizeof(struct func_record));
	optimize_memcpy(func_record.f_item.func_name, name, FUNC_MAX_NAME_LEN);
	optimize_memcpy(func_record.f_item.lib_md5, lib->md5, MD5SUM_LEN);

	// first find in hashtable
	HASH_FIND(hh, cur_thread->func_table, &func_record.f_item, sizeof(struct func_item), p_func_record);
	if (p_func_record) {
		*pfunc = (char *)p_func_record->f_ptr;
		return 0;
	}

	tmp = dlsym(lib->handle, (char *)name);
	if (!tmp) {
		pr_err("find function %s in module error!\n", name);
		return -1;
	}
	*pfunc = (char *)tmp;

	// add to hashtable cur_thread->func_table
	p_func_record = (struct func_record *)malloc(sizeof(struct func_record));
	asm_memset(p_func_record, 0, sizeof(struct func_record));
	optimize_memcpy(p_func_record->f_item.func_name, name, FUNC_MAX_NAME_LEN);
	optimize_memcpy(p_func_record->f_item.lib_md5, lib->md5, MD5SUM_LEN);
	p_func_record->f_ptr = tmp;
	HASH_ADD(hh, cur_thread->func_table, f_item, sizeof(struct func_item), p_func_record);

	return 0;
}

int run_empty_kernel(void)
{
	struct task_item *cur_task;
	struct list_head *pos_lib;
	int ret;

	//todo:no trigger
	return 0;
	// pr_debug("I'm in %s\n", __func__);
}

int load_lib_process(struct task_item *task_item)
{
	char local_file[1024] = {0};
	struct stat st = {0};
	uint8_t *map_vaddr;
	char *error;
	int fd;
	int ret = 0;

	pr_debug("I'm in %s\n", __func__);

	run_empty_kernel();
	poll_cur_task_enable();

	struct load_module_internal *load_module =
		(struct load_module_internal *)
		((char *)task_item->task.task_body + sizeof(struct api_header));

	pr_debug("library addr is 0x%llx\n", (uint64_t)load_module->library_addr);
	pr_debug("library size is 0x%x\n", load_module->size);
	tp_debug("[load]:%s\n", load_module->library_name);
	show_md5(load_module->md5);

	map_vaddr = sg_get_base_addr() + (uint64_t)load_module->library_addr;

	strncpy(local_file, workdir(), sizeof(local_file));

	ret = sprintfcat(local_file, "/%s", load_module->library_name);
	if (ret < 0)
		pr_err("snprintf failed\n");

	fd = open(local_file, O_CREAT | O_WRONLY, 0755);
	if (fd < 0) {
		pr_err("open file %s error! %s\n", local_file, strerror(errno));
		return -1;
	}

	pr_debug("%s: lib write to %s\n", __func__, local_file);
	invalidate_dcache_range((void *)map_vaddr, load_module->size);
	if (write(fd, map_vaddr, load_module->size) != load_module->size) {
		perror(NULL);
		pr_err("write file %s error! errno %s\n", local_file, strerror(errno));
		close(fd);
		return -1;
	}
	close(fd);

	struct library_item *ptr_lib_item =
		(struct library_item *)malloc(sizeof(struct library_item));

	ptr_lib_item->lib.handle = dlopen(local_file, RTLD_LOCAL  | RTLD_NOW);
	if (!(ptr_lib_item->lib.handle)) {
		pr_err("dlopen  %s error!\n", local_file);
		return -1;
	}

	optimize_memcpy(ptr_lib_item->lib.lib_name, load_module->library_name, LIB_MAX_NAME_LEN);
	optimize_memcpy(ptr_lib_item->lib.md5, load_module->md5, MD5SUM_LEN);
	list_add(&ptr_lib_item->list, &cur_thread->load_lib_list);

	extern void tpu_kernel_init(int core_idx);
	extern int tpu_core_barrier(int msg_id, int core_num);
	extern void tpu_poll();

	void (*tpu_func_ptr)(int core_id);

	ret = find_sym_by_name(&ptr_lib_item->lib, (unsigned char *)"tpu_kernel_init", (char **)&tpu_func_ptr);
	if (!ret) {
		pr_debug("%s: before tpu_kernel_init\n", __func__);
		//relocate tpu_kernel_init
		tpu_func_ptr = tpu_kernel_init;
		tpu_func_ptr(cur_thread->tpu_id);
		pr_debug("%s: after tpu_kernel_init\n", __func__);
	} else
		pr_err("%s: find_sym_by_name: tpu_kernel_init failed, ret = %d\n", __func__, ret);
	
	ret = find_sym_by_name(&ptr_lib_item->lib, (unsigned char *)"tpu_core_barrier", (char **)&cur_thread->task_barrier);
	ret &= find_sym_by_name(&ptr_lib_item->lib, (unsigned char *)"tpu_poll", (char **)&cur_thread->poll_engine_done);
	if (!ret) {
		//relocate task_barrier
		cur_thread->task_barrier = NULL;
		//relocate poll_engine_done
		cur_thread->poll_engine_done = tpu_poll;
	}

	return ret;
}

int unload_lib_process(struct task_item *task_item)
{
	struct list_head *pos_lib;
	char local_file[1024];
	int ret = 0;

	pr_debug("I'm in %s\n", __func__);

	run_empty_kernel();
	poll_cur_task_enable();

	struct load_module_internal *load_module =
		(struct load_module_internal *)
		((char *)task_item->task.task_body + sizeof(struct api_header));

	pr_debug("library addr is 0x%llx\n", (uint64_t)load_module->library_addr);
	pr_debug("library size is 0x%x\n", load_module->size);
	tp_debug("[unload]:%s\n", load_module->library_name);
	show_md5(load_module->md5);

	list_for_each(pos_lib, &cur_thread->load_lib_list) {
		if (!memcmp(load_module->md5, ((struct library_item *)pos_lib)->lib.md5, MD5SUM_LEN)) {
			void (*tpu_func_ptr)(void);

			ret = find_sym_by_name(&((struct library_item *)pos_lib)->lib,
						(unsigned char *)"tpu_kernel_deinit",
						(char **)&tpu_func_ptr);
			if (!ret) {
				pr_debug("%s: before tpu_kernel_deinit\n", __func__);
				tpu_func_ptr();
				pr_debug("%s: after tpu_kernel_deinit\n", __func__);
			} else
				pr_err("%s: find_sym_by_name: tpu_kernel_deinit failed, ret = %d\n", __func__, ret);
			ret = dlclose(((struct library_item *)pos_lib)->lib.handle);
			if (ret != RT_TRUE)
				pr_err("dlclose error!\n");

			struct func_record *tmp_func_record, *p_func_record;

			HASH_ITER(hh, cur_thread->func_table, p_func_record, tmp_func_record) {
				if (!memcmp(p_func_record->f_item.lib_md5,
					((struct library_item *)pos_lib)->lib.md5, MD5SUM_LEN)) {
					HASH_DEL(cur_thread->func_table, p_func_record);
					free(p_func_record);
				}
			}

			list_del(&((struct library_item *)pos_lib)->list);
			free((struct library_item *)pos_lib);
			break;
		}
	}

	sprintf(local_file, "%s/%s", workdir(), load_module->library_name);
	ret = remove(local_file);
	if (ret != 0)
		pr_err("failed to remove dir %s!\n", local_file);

	cur_thread->task_barrier = NULL;
	cur_thread->poll_engine_done = NULL;

	return ret;
}
typedef struct {
    uint32_t physical_core_id;
    uint64_t group_num;
    uint64_t workitem_num;
    uint32_t group_id;
    uint32_t workitem_id;
} __attribute__((packed)) tpu_groupset_info_t;
int launch_func_process(struct task_item *task_item)
{
	struct list_head *pos_lib;
	int find = 0;
	int ret = 0;
	uint64_t start_time;
	uint64_t end_time;

	pr_debug("I'm in %s\n", __func__);

	struct launch_func_internal *launch_func =
		(struct launch_func_internal *)
		((char *)task_item->task.task_body + sizeof(struct api_header));

	pr_debug("received func_name is %s\n", launch_func->fun_name);
	pr_debug("received lib_name is %s\n", launch_func->lib_name);
	show_md5(launch_func->lib_md5);
	pr_debug("received size is 0x%x\n", launch_func->size);

	if (cur_thread->sync_mode == SYNC_MODE)
		poll_cur_task_enable();

	list_for_each(pos_lib, &cur_thread->load_lib_list) {
		if (!memcmp(launch_func->lib_md5, ((struct library_item *)pos_lib)->lib.md5, MD5SUM_LEN)) {
			struct tpu_groupset_info groupset_info;
			void (*set_tpu_info_func_ptr)(struct tpu_groupset_info *ptr_groupset_info) = NULL;
			int (*func_ptr)(void *addr, unsigned int size) = NULL;

			pr_debug("%s: find module\n", __func__);
			find = 1;
			get_tpu_groupset_info(&groupset_info);
			ret = find_sym_by_name(&((struct library_item *)pos_lib)->lib,
						(unsigned char *)"set_tpu_groupset_info",
						(char **)&set_tpu_info_func_ptr);
			if (!ret) {
				pr_debug("%s: before set_tpu_groupset_info\n", __func__);
				tp_debug("find set tpu info\n");
				void set_tpu_groupset_info( tpu_groupset_info_t *tpu_groupset);
				set_tpu_info_func_ptr = set_tpu_groupset_info;
				set_tpu_info_func_ptr(&groupset_info);
				pr_debug("%s: after set_tpu_groupset_info\n", __func__);
			} else
				pr_err("find_sym_by_name: set_tpu_groupset_info failed, ret = %d\n", ret);

			tp_debug("find sym\n");
			ret = find_sym_by_name(&((struct library_item *)pos_lib)->lib,
						launch_func->fun_name,
						(char **)&func_ptr);
			if (!ret) {
				pr_debug("%s: before %s\n", __func__, launch_func->fun_name);
				tp_debug("[kernel]:%s\n", launch_func->fun_name);
				get_time(start_time);
				ret = func_ptr(launch_func->param, launch_func->size);
				get_time(end_time);
				cur_thread->kernel_exec_time += (end_time - start_time);
				cur_thread->tp_status->kernel_exec_time = cur_thread->kernel_exec_time;
				tp_debug("[exit]%s:%d\n", launch_func->fun_name, ret);
				pr_debug("%s: after %s\n", __func__, launch_func->fun_name);
			} else
				tp_debug("[error]:func %s not found\n", launch_func->fun_name);
			break;
		}
	}

	if (!find) {
		tp_debug("[error]:lib %s not found\n", launch_func->lib_name);
		return -1;
	}

	return ret;
}
