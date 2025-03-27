#include "firmware_common.h"

#ifdef USING_CMODEL
#include "store_cmd.h"
#endif

void fix_cmd_node_overflow(CMD_ID_NODE *pid_node) {
    if (pid_node->gdma_cmd_id > CMD_ID_OVERFLOW_VALUE ||
            pid_node->bd_cmd_id > CMD_ID_OVERFLOW_VALUE) {
        poll_all_engine_done(pid_node);
        resync_cmd_id(pid_node);
    }
}

void cmd_id_divide(CMD_ID_NODE * p_cmd_src,
                   CMD_ID_NODE * p_cmd_dst0, CMD_ID_NODE * p_cmd_dst1) {
#ifdef USING_CMODEL
    if (get_enable_profile()) {
      fprintf(get_profile_file(), "[bmprofile] start parallel.\n");
    }
#endif
    p_cmd_dst0->bd_cmd_id = p_cmd_src->bd_cmd_id;
    p_cmd_dst0->gdma_cmd_id = p_cmd_src->gdma_cmd_id;
    p_cmd_dst0->in_parallel_state = true;
    p_cmd_src->in_parallel_state = true;
    p_cmd_dst0->in_sync_state = p_cmd_dst1->in_sync_state
                              = p_cmd_dst0->in_sync_state;
#ifdef SG_STAS_GEN
    p_cmd_dst0->cycle_count = p_cmd_src->cycle_count;
    strncpy(p_cmd_dst0->name_prefix, p_cmd_src->name_prefix, sizeof((*p_cmd_dst0).name_prefix));
#endif
    p_cmd_dst1->bd_cmd_id = p_cmd_src->bd_cmd_id;
    p_cmd_dst1->gdma_cmd_id = p_cmd_src->gdma_cmd_id;
    p_cmd_dst1->in_parallel_state = true;
#ifdef SG_STAS_GEN
    p_cmd_dst1->cycle_count = p_cmd_src->cycle_count;
    strncpy(p_cmd_dst1->name_prefix, p_cmd_src->name_prefix, sizeof((*p_cmd_dst1).name_prefix));
#endif
}

void cmd_id_merge(CMD_ID_NODE *p_cmd_dst,
                  CMD_ID_NODE *p_cmd_src0, CMD_ID_NODE *p_cmd_src1) {
    p_cmd_dst->bd_cmd_id = sg_max(p_cmd_src0->bd_cmd_id, p_cmd_src1->bd_cmd_id);
    p_cmd_dst->gdma_cmd_id = sg_max(p_cmd_src0->gdma_cmd_id, p_cmd_src1->gdma_cmd_id);
    p_cmd_dst->in_parallel_state = false;
    ASSERT(p_cmd_src0->in_sync_state == p_cmd_src1->in_sync_state);
    p_cmd_dst->in_sync_state = p_cmd_src0->in_sync_state;
#ifdef SG_STAS_GEN
    p_cmd_dst->cycle_count = sg_max(p_cmd_src0->cycle_count, p_cmd_src1->cycle_count);
#endif
#ifdef USING_CMODEL
    cmdid_overflow_reset(p_cmd_dst);
    if (get_enable_profile()) {
      fprintf(get_profile_file(), "[bmprofile] end parallel.\n");
    }
#endif
}

void fw_log(char *fmt, ...)
{
    va_list args;
    va_start(args,fmt);
    vprintf(fmt, args);
    va_end(args);
}
