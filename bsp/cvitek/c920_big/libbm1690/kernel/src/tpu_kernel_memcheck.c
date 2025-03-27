#include <stdlib.h>
#include "tpu_kernel.h"

typedef struct mem_record_node_t{
    system_addr_t addr;
    u64 size;
    u64 info;
    struct mem_record_node_t *next;
    u32 checksum;
} mem_record_node_t;

typedef struct mem_record_list_t {
    mem_record_node_t *head;
    mem_record_node_t *tail;
} mem_record_list_t;

static mem_record_list_t __mem_record_list = {NULL, NULL};

static u32 caluculate_checksum(const mem_record_node_t *node) {
    u32 checksum = 0;
    u32* start = (u32*)node;
    u32* end = (u32*)(&node->checksum);
    for(u32 *data = start; data < end; data++) {
        checksum ^= *data;
    }
    return checksum;
}

static inline void print_mem_record_node(const char* prefix, mem_record_node_t *node) {
    TPUKERNEL_LOG("%smem record %p: range=[0x%llx,0x%llx), size=%lld(0x%llx), info=0x%llx, checksum=0x%x\n",
      prefix, node, node->addr, node->addr + node->size, node->size, node->size, node->info, node->checksum);
}

// should not clear all the mem record list, for the memory may be invalid
#define CHECK_NODE_VALID(node)                         \
    do {                                               \
        u32 checksum = caluculate_checksum(node);      \
        if (checksum != node->checksum) {              \
            print_mem_record_node("bad ", node);       \
            TPUKERNEL_ASSERT(0);                       \
        }                                              \
    } while (0)

static mem_record_node_t * add_mem_record_node(mem_record_list_t *list, system_addr_t addr, u64 size, u64 info) {
    mem_record_node_t *mem_record_node = (mem_record_node_t *)malloc(sizeof(mem_record_node_t));
    mem_record_node->addr = addr;
    mem_record_node->size = size;
    mem_record_node->info = info;
    mem_record_node->next = NULL;
    mem_record_node->checksum = caluculate_checksum(mem_record_node);
    if(!list->tail) {
        list->head = mem_record_node;
        list->tail = mem_record_node;
    } else {
        CHECK_NODE_VALID(list->tail);
        list->tail->next = mem_record_node;
        list->tail->checksum = caluculate_checksum(list->tail);
        list->tail = mem_record_node;
    }
    print_mem_record_node("add ", mem_record_node);
    return mem_record_node;
}

static int del_mem_record_node(mem_record_list_t *list, system_addr_t addr, u64 size) {
    if(!list || !list->head) return 0;
    mem_record_node_t *prev_mem_record_node = list->head;
    CHECK_NODE_VALID(prev_mem_record_node);
    if(prev_mem_record_node->addr == addr && (size == 0 || prev_mem_record_node->size == size)) {
        list->head = prev_mem_record_node->next;
        if(list->head == NULL) {
            list->tail = NULL;
        }
        print_mem_record_node("del ", prev_mem_record_node);
        free(prev_mem_record_node);
        return 1;
    }

    mem_record_node_t *mem_record_node = prev_mem_record_node->next;
    while(mem_record_node) {
        CHECK_NODE_VALID(mem_record_node);
        if(mem_record_node->addr == addr && (size==0 || mem_record_node->size == size)) {
            prev_mem_record_node->next = mem_record_node->next;
            prev_mem_record_node->checksum = caluculate_checksum(prev_mem_record_node);
            if(prev_mem_record_node->next == NULL) {
                list->tail = prev_mem_record_node;
            }
            print_mem_record_node("del ", mem_record_node);
            free(mem_record_node);
            return 1;
        }
    }
    return 0;
}

static void clear_mem_record_node(mem_record_list_t *list) {
    if(!list) return;
    mem_record_node_t* mem_record_node = list->head;
    TPUKERNEL_LOG("clearing mem record list:\n");
    while(mem_record_node) {
        CHECK_NODE_VALID(mem_record_node);
        print_mem_record_node("  ", mem_record_node);
        mem_record_node_t *next = mem_record_node->next;
        free(mem_record_node);
        mem_record_node = next;
    }
    list->head = NULL;
    list->tail = NULL;
}

typedef int (*cond_func_t)(mem_record_node_t *node, system_addr_t addr, u64 size);
static mem_record_node_t *tpu_mem_record_find_by_condition(mem_record_list_t* list, system_addr_t addr, u64 size, cond_func_t func) {
    if(!list) return NULL;
    mem_record_node_t *mem_record = list->head;
    while(mem_record) {
        CHECK_NODE_VALID(mem_record);
        if(func(mem_record, addr, size)) {
            return mem_record;
        }
        mem_record = mem_record->next;
    }
    return mem_record;
}

static int is_mem_record_same(mem_record_node_t *mem_record, system_addr_t addr, u64 size) {
    return (mem_record->addr == addr && size == mem_record->size);
}

static int is_mem_record_overlap(mem_record_node_t *mem_record, system_addr_t addr, u64 size) {
    return mem_record->addr + mem_record->size >  addr && addr+size > mem_record->addr + mem_record->size;
}

static int is_mem_record_in_range(mem_record_node_t *mem_record, system_addr_t addr, u64 size) {
        return mem_record->addr <= addr && addr+size <= mem_record->addr + mem_record->size;
}

static mem_record_node_t *tpu_mem_record_find_same(mem_record_list_t* list, system_addr_t addr, u64 size) {
    return tpu_mem_record_find_by_condition(list, addr, size, is_mem_record_same);
}

static mem_record_node_t *tpu_mem_record_find_in_range(mem_record_list_t* list, system_addr_t addr, u64 size) {
    return tpu_mem_record_find_by_condition(list, addr, size, is_mem_record_in_range);
}
static mem_record_node_t *tpu_mem_record_find_overlap(mem_record_list_t* list, system_addr_t addr, u64 size) {
    return tpu_mem_record_find_by_condition(list, addr, size, is_mem_record_overlap);
}

void tpu_mem_record_add(system_addr_t addr, u64 size, u64 info) {
    mem_record_node_t *mem_record = tpu_mem_record_find_overlap(&__mem_record_list, addr, size);
    if(mem_record) {
        print_mem_record_node("overlapping mem record exists: ", mem_record);
        clear_mem_record_node(&__mem_record_list);
        TPUKERNEL_ASSERT_INFO(mem_record==NULL, "mem record added addr=0x%llx, size=0x%llx, info=0x%llx", addr, size, info);
    }
    mem_record = add_mem_record_node(&__mem_record_list, addr, size, info);
}

void tpu_mem_record_del(system_addr_t addr, u64 size) {
    int success = del_mem_record_node(&__mem_record_list, addr, size);
    if(!success){
        clear_mem_record_node(&__mem_record_list);
    }
    TPUKERNEL_ASSERT_INFO(success, "del mem_record failed, cannot find record for addr=0x%llx, size=0x%llx", addr, size);
}

static inline void __do_mem_check(u64 addr, u64 size) {
    if((addr >= tpu_l2_sram_get_start_addr()) && (addr < tpu_l2_sram_get_start_addr() + tpu_l2_sram_size())) {
        TPUKERNEL_LOG("  in L2 sram, ignored\n");
        return;
    }
    mem_record_node_t *mem_record = tpu_mem_record_find_in_range(&__mem_record_list, addr, size);
    if(mem_record == NULL) {
        clear_mem_record_node(&__mem_record_list);
        TPUKERNEL_ASSERT_INFO(mem_record != NULL, "mem record not found");
    } else {
        print_mem_record_node("  found: ", mem_record);
    }
}

void tpu_mem_check_range(system_addr_t addr, u64 size) {
    if(!__mem_record_list.head) return;
    system_addr_t real_addr = tpu_global_mem_real_addr(addr);
    TPUKERNEL_LOG("checking addr=0x%llx, range=[0x%llx, 0x%llx), size=%lld(0x%llx)\n", addr, real_addr, real_addr+size, size, size);
    __do_mem_check(real_addr, size);
}

void tpu_mem_check_matrix(system_addr_t addr, u32 rows, u32 cols, u32 row_stride, data_type_t dtype) {
    if(!__mem_record_list.head) return;
    u64 size = 0;
    size = (rows-1)*row_stride + cols;
    size = size * tpu_data_type_size(dtype);
    system_addr_t real_addr = tpu_global_mem_real_addr(addr);
    TPUKERNEL_LOG("checking addr=0x%llx, range=[0x%llx, 0x%llx), size=%lld(0x%llx), row=%d, col=%d, row_stride=%d, dtype=%d\n",
        addr, real_addr, real_addr+size, size, size,
        rows, cols, row_stride, dtype);
    __do_mem_check(real_addr, size);
}

void tpu_mem_check_tensor(system_addr_t addr, const dim4 *shape, const dim4 *stride, data_type_t dtype) {
    if(!__mem_record_list.head) return;
    u64 size = 0;
    dim4 real_stride = {0};
    if(!stride) {
        tpu_continuous_stride(&real_stride, shape);
        stride = &real_stride;
    }
    size = (shape->n - 1) * stride->n + (shape->c - 1) * stride->c + (shape->h - 1) * stride->h + (shape->w - 1) * stride->w + 1;
    size = size * tpu_data_type_size(dtype);
    system_addr_t real_addr = tpu_global_mem_real_addr(addr);
    TPUKERNEL_LOG("checking addr=0x%llx, range=[0x%llx, 0x%llx), size=%lld(0x%llx), shape=[%d,%d,%d,%d], stride=[%d,%d,%d,%d], dtype=%d\n",
        addr, real_addr, real_addr+size, size, size,
        shape->n, shape->c, shape->h, shape->w,
        stride->n, stride->c, stride->h, stride->w,
        dtype);
    __do_mem_check(real_addr, size);
}
