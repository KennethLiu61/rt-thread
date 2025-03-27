#include "firmware_profile.h"
#include "firmware_common_macro.h"
#include "firmware_common_inline.h"
#include "firmware_timer.h"
#include "sg_api_struct.h"
#include "firmware_pmu.h"
#include "firmware_runtime.h"
#include "tpu_kernel.h"

#define BYTES_PER_BLOCK (sizeof(fw_profile_time_info_t)*1024*2048)

#define FW_PROFILE_DATA_STRING 0
#define FW_PROFILE_DATA_BINARY 1
#define FW_PROFILE_DATA_CUSTOM 100

#define MAX_RECORD_COUNT (0)

#pragma pack(1)
typedef struct {
    int profile;
    int type;
    int len;
} fw_profile_data_head_t;
#pragma pack()

typedef struct fw_profile_data_block_t {
    struct fw_profile_data_block_t* next;
    unsigned int                    used; //unit is byte
    u8                              data[BYTES_PER_BLOCK];
} fw_profile_data_block_t;

typedef struct {
    unsigned int             real_count;
    fw_profile_data_block_t* time_tail;
    fw_profile_data_block_t* time_head;
    fw_profile_data_block_t* data_head;
    fw_profile_data_block_t* data_tail;
} fw_profile_time_context_t;

#ifndef USING_CMODEL
fw_profile_time_context_t* g_profile_context = NULL;
int g_profile_enabled = 0;
#else
// Tpuv7rt cmodel multi-thread method share the same g_profile_context
_Thread_local fw_profile_time_context_t* g_profile_context = NULL;
_Thread_local int g_profile_enabled = 0;
#endif

#define IMPLEMENT_NEW_PROFILE_DATA(type)                                       \
    static inline fw_profile_data_block_t* new_profile_data_for_##type(        \
        fw_profile_time_context_t* ctx) {                                      \
        fw_profile_data_block_t* block = (fw_profile_data_block_t*)malloc(     \
            sizeof(fw_profile_data_block_t));                                  \
        ASSERT_INFO(block, "no free memory for profile");                      \
        block->used = 0;                                                       \
        block->next = 0;                                                       \
        if (ctx->type##_head) {                                                \
            ctx->type##_tail->next = block;                                    \
        } else {                                                               \
            ctx->type##_head = block;                                          \
        }                                                                      \
        ctx->type##_tail = block;                                              \
        return block;                                                          \
    }
IMPLEMENT_NEW_PROFILE_DATA(data)
IMPLEMENT_NEW_PROFILE_DATA(time)

static inline fw_profile_time_info_t* profile_time_begin(int32_t type, uint32_t id) {
    if(!profile_time_enabled() && type != PROFILE_UNKNOWN) return NULL;
    fw_profile_time_context_t* ctx = g_profile_context;
    if(!ctx) return NULL;
    if (MAX_RECORD_COUNT>0 && ctx->real_count>MAX_RECORD_COUNT) return NULL;
    if (!ctx->time_tail || ctx->time_tail->used >= BYTES_PER_BLOCK)
        new_profile_data_for_time(ctx);
    fw_profile_data_block_t* block = ctx->time_tail;
    fw_profile_time_info_t*  info  = (fw_profile_time_info_t*)&block->data[block->used];
    block->used += sizeof(fw_profile_time_info_t);
    ctx->real_count ++;
    info->type                    = type;
    info->inst_id                 = id;
    return info;
}

void init_profile_time_context(fw_profile_time_context_t* ctx) {
    if(!ctx) return;
    ctx->real_count = 0;
    ctx->time_head = NULL;
    ctx->time_tail  = NULL;
    ctx->data_head = NULL;
    ctx->data_tail = NULL;
    profile_time_begin(PROFILE_UNKNOWN, 0);
}

void deinit_profile_time_context(fw_profile_time_context_t* ctx) {
    fw_profile_data_block_t* time_head = ctx->time_head;
    while (time_head) {
        ctx->time_head = time_head->next;
        free(time_head);
        time_head = ctx->time_head;
    }
    ctx->time_tail = NULL;
    fw_profile_data_block_t* data_head = ctx->data_head;
    while (data_head) {
        ctx->data_head = data_head->next;
        free(data_head);
        data_head = ctx->data_head;
    }
    ctx->data_tail = NULL;
}

void set_profile_time_enabled(int enable) {
    if(enable) {
        if(g_profile_context)
            deinit_profile_time_context(g_profile_context);
        else {
            g_profile_context = (fw_profile_time_context_t*)malloc(sizeof(fw_profile_time_context_t));
            ASSERT_INFO(g_profile_context!=NULL, "out of memory for profile context malloc");
        }
        init_profile_time_context(g_profile_context);
    } else {
        if(g_profile_context) {
            deinit_profile_time_context(g_profile_context);
            g_profile_context = 0;
        }
    }
}

static inline void *tpu_memcpy(void *dst, const void *src, size_t n)
{
    volatile u8 *tmp = (u8*)dst;
    volatile u8 *s = (u8*)src;
    while (n--) {
        *tmp++ = *s++;
    }
    return dst;
}


static inline void profile_time_copy_data(fw_profile_time_context_t* ctx, const unsigned char* data, size_t data_len) {
    fw_profile_data_block_t* extra_data = ctx->data_tail;
    if(!extra_data)
        extra_data = new_profile_data_for_data(ctx);
    size_t copied_len = 0;
    while(copied_len < data_len) {
        size_t left_buffer_len = BYTES_PER_BLOCK - extra_data->used;
        size_t left_data_len = data_len - copied_len;
        size_t can_copy_len = sg_min(left_buffer_len, left_data_len);
        tpu_memcpy(extra_data->data+ extra_data->used, data + copied_len, can_copy_len);
        extra_data->used += can_copy_len;
        copied_len += can_copy_len;
        if(extra_data->used == BYTES_PER_BLOCK)
            extra_data = new_profile_data_for_data(ctx);
    }
}

void profile_time_add_extra_data(fw_profile_time_info_t* info, int type, const unsigned char* data, size_t len) {
    if(!profile_time_enabled()) return;
    fw_profile_time_context_t* ctx = g_profile_context;
    if(!ctx) return;
    fw_profile_data_head_t head;
    head.len = len;
    head.type = type;
    head.profile = info->type;
    profile_time_copy_data(ctx, (const unsigned char*)&head, sizeof(head));
    profile_time_copy_data(ctx, data, len);
}

void profile_time_add_extra_string(fw_profile_time_info_t* info, const char* data) {
    return;
    profile_time_add_extra_data(info, FW_PROFILE_DATA_STRING, (const unsigned char*)data, strlen(data));
}

void profile_time_add_extra_binary(fw_profile_time_info_t* info, const unsigned char* data, int len) {
    profile_time_add_extra_data(info, FW_PROFILE_DATA_BINARY, data, len);
}

INLINE  void profile_time_set_node(int engine, int func, int special_func, int info, CMD_ID_NODE* pid_node, u64* high, u64* low, int elt) {
    // |0.....|8.......|11.............|16................|21.............................|31.......|32
    // | type | engine | bd_op/dma_op  | bd_func/dma_dir  | rsvd/dtype && direct sys_info | parellel|
    // | 8    |  3     | 5             | 5                | 10                            | 1
    // rsvd/dtype/sys_info:
    //     bd: dtype (0xf, 4bits)
    //     dma: (direct << 4 | dtype) (0x7f, 7bit)
    //     cdma: (port << 7 |direct << 4 | dtype) (0x7ff, 11bit)
    //     sys: rsvd(0) or send_cnt/ wait_cnt (0x7f 7bits)
    if(!profile_time_enabled()) return;
    if(!pid_node) return;
    // printf("engine %d: func %d special_func %d bd_id %d gdma %d\n", engine, func, special_func, pid_node->bd_cmd_id, pid_node->gdma_cmd_id);
    uint32_t type = PROFILE_NODE_SET;
    type |= (engine&0x7) << 8 | (func&0x1F) << 11 | (special_func&0x1F) << 16;
    uint32_t inst_id = 0;
    switch (engine) {
    case ENGINE_BD:
        type |= (info&0x7f) << 21 | (pid_node->in_parallel_state) << 31;
        inst_id = pid_node->bd_cmd_id & 0x0FFFF;
        break;
    case ENGINE_GDMA:
        type |= (info&0x7f) << 21 | (pid_node->in_parallel_state) << 31;
        inst_id = pid_node->gdma_cmd_id & 0x0FFFF;
        break;
    case ENGINE_SDMA:
        type |= (info&0x7f) << 21 | (pid_node->in_parallel_state) << 31;
        inst_id = pid_node->sdma_cmd_id & 0x0FFFF;
        break;
    case ENGINE_VSDMA:
        type |= (info&0x7f) << 21 | (pid_node->in_parallel_state) << 31;
        inst_id = pid_node->vsdma_cmd_id[tpu_workitem_index()];
        break;
    case ENGINE_CDMA:
        type |=  (info&0x7ff) << 21;
        inst_id = pid_node->cdma_cmd_id[info >> 7] & 0x0FFFF;
        break;
    }
    fw_profile_time_info_t* p_info = profile_time_begin(type, inst_id);
    if (g_profile_enabled == 2) {
        u64 command[16];
        for(int i = 0; i < elt; i++){
            command[2 * i] = low[i];
            command[2 * i + 1] = high[i];
        }
        profile_time_add_extra_binary(p_info, (const unsigned char*)command, 2 * elt * sizeof(command[0]));
    }
}

static inline size_t __get_profile_data_len(fw_profile_data_block_t* head) {
    size_t total_len = 0;
    while(head) {
        total_len += head->used;
        head = head->next;
    }
    return total_len;
}

static inline size_t __get_profile_data(fw_profile_data_block_t* head, unsigned char* buffer, size_t buffer_len, size_t offset) {
    size_t copied_len = 0;
    size_t current_offset = 0;
    fw_profile_data_block_t* block = head;
    while(block && block->used>0) {
        size_t copy_offset  = 0;
        size_t can_copy_len = 0;
        if (current_offset <= offset && offset < block->used + current_offset) {
            copy_offset  = offset - current_offset;
            can_copy_len = block->used - copy_offset;
        } else if (current_offset > offset)
            can_copy_len = block->used;
        if (copied_len + can_copy_len > buffer_len)
            can_copy_len = buffer_len - copied_len;
        if (can_copy_len > 0) {
            tpu_memcpy(buffer + copied_len, block->data + copy_offset, can_copy_len);
            copied_len += can_copy_len;
            if (copied_len == buffer_len)
                break;
        }
        current_offset += block->used;
        block = block->next;
    }
    return copied_len;
}

size_t get_profile_time_data_len() {
    fw_profile_time_context_t *ctx = g_profile_context;
    if(!ctx || !ctx->time_head) return 0;
    return __get_profile_data_len(ctx->time_head);
}

size_t get_profile_extra_data_len() {
    fw_profile_time_context_t *ctx = g_profile_context;
    if(!ctx || !ctx->data_head) return 0;
    return __get_profile_data_len(ctx->data_head);
}

size_t get_profile_time_data(unsigned char* buffer, size_t buffer_len, size_t offset) {
    fw_profile_time_context_t *ctx = g_profile_context;
    if(!ctx || !ctx->time_head) return 0;
    return __get_profile_data(ctx->time_head, buffer, buffer_len, offset);
}

size_t get_profile_extra_data(unsigned char* buffer, size_t buffer_len, size_t offset) {
    fw_profile_time_context_t *ctx = g_profile_context;
    if(!ctx || !ctx->data_head) return 0;
    return __get_profile_data(ctx->data_head, buffer, buffer_len, offset);
}

INLINE int profile_time_enabled() {
    return g_profile_enabled;
}

sg_fw_status_t sg_api_set_profile(unsigned char *api_buf, int size){
    tpu_poll();

    sg_api_set_profile_t *api = (sg_api_set_profile_t*)api_buf;
    ASSERT(size == sizeof(sg_api_set_profile_t));
    bool init_status = !g_profile_context;
    bool pause = (api->enable >> PROFILE_PAUSE) & 0x1;
    enable_pmu((api->enable >> PROFILE_ENGINE_CDMA) & 0x3,
            pause ? NULL: set_profile_time_enabled,
            api->enable & 0x3);
    g_profile_enabled = api->enable & 0x3;
    if (api->enable && init_status && !pause) {

        int msg_id[CONFIG_MAX_CDMA_NUM];
        if (tpu_is_last_workitem()) {
            for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) {
                if (cdma_port_is_valid(port)) {
                    msg_id[port] = tpu_get_ccl_msg_id();
                    tpu_cdma_tx_wait_msg(port, msg_id[port], 1);
                }
            }
        }

        tpu_sync_all();

        if (tpu_is_last_workitem()) {
            for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) {
                if (cdma_port_is_valid(port)) {
                    tpu_vsdma_send_msg(msg_id[port], 1, 0);
                }
            }

            for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) {
                if (cdma_port_is_valid(port)) {
                    tpu_cdma_nop(port);
                    tpu_cdma_port_poll(port);
                }
            }
        }
    }
    return SG_FW_SUCCESS;
}

sg_fw_status_t sg_api_set_engine_profile_param(unsigned char *api_buf,
                                               int size) {
  tpu_poll();
  set_pmu_param(api_buf);
  return SG_FW_SUCCESS;
}

typedef struct {
    u32 read_len;
    u32 total_len; //0 means finished
} sg_profile_data_header_t;

sg_fw_status_t sg_api_get_profile_data(
        unsigned char *api_buf, int size){
  // quit cond
  sg_api_get_profile_data_t *api = (sg_api_get_profile_data_t*)api_buf;
  fw_profile_time_context_t *ctx = g_profile_context;
  // if time_head is init status quit
  if(!ctx || !ctx->time_head || ctx->real_count == 1) {
    int size = sizeof(sg_profile_data_header_t);
    memset(GET_GLOBAL_ADDR(api->output_global_addr), 0, size);
    flush_cache(api->output_global_addr, size);
    return SG_FW_SUCCESS;
  }
  ASSERT(size == sizeof(sg_api_get_profile_data_t));
  int core_id = (api->data_category >> 1);
  if (core_id != tpu_workitem_index()) return SG_FW_SUCCESS;
  api->data_category = api->data_category & 0x1;
  unsigned char* out_buffer = (unsigned char*)GET_GLOBAL_ADDR(api->output_global_addr);

  sg_profile_data_header_t * header = (sg_profile_data_header_t*)out_buffer;
  out_buffer += sizeof(sg_profile_data_header_t);
  size_t buffer_size = api->output_size-sizeof(sg_profile_data_header_t);

  // copy back to global addr
  if(api->data_category == 0){
      header->total_len = get_profile_time_data_len();
      header->read_len = get_profile_time_data(out_buffer, buffer_size, api->byte_offset);
  } else {
      header->total_len = get_profile_extra_data_len();
      header->read_len = get_profile_extra_data(out_buffer, buffer_size, api->byte_offset);
  }
  size_t copy_len = header->read_len + sizeof(sg_profile_data_header_t);
  flush_cache(api->output_global_addr, ALIGN(copy_len, CACHE_LINE_SIZE));

  return SG_FW_SUCCESS;
}

void profile_time_end(void* handle) {
    if (!handle) return;
    fw_profile_time_info_t * info = (fw_profile_time_info_t*) handle;
    profile_time_begin(info->type, info->inst_id);
}

fw_profile_time_info_t* profile_time_begin_func(unsigned long long id) {
    return NULL;
    // return profile_time_begin(PROFILE_FUNC, id);
}

fw_profile_time_info_t *profile_time_begin_custom(unsigned long long data) {
    return NULL;
    // return profile_time_begin(PROFILE_CUSTOM, 0);
}

void profile_time_wait_node(uint32_t id) {
    profile_time_begin(PROFILE_NODE_WAIT, id);
}

void profile_time_wait_vsdma(uint32_t id) {
    profile_time_begin(PROFILE_VSDMA, id);
}

void profile_time_wait_cdma(uint32_t id) {
    profile_time_begin(PROFILE_CDMA, id);
}

void profile_time_wait_gde(uint32_t id) {
    profile_time_begin(PROFILE_GDE, id);
}

void profile_time_wait_sort(uint32_t id) {
    profile_time_begin(PROFILE_SORT, id);
}

void profile_time_wait_nms(uint32_t id) {
    profile_time_begin(PROFILE_NMS, id);
}

void profile_time_wait_func(uint32_t id) {
    profile_time_begin(PROFILE_FUNC, id);
}

void profile_time_wait_custom(uint32_t id) {
    profile_time_begin(PROFILE_CUSTOM, id);
}