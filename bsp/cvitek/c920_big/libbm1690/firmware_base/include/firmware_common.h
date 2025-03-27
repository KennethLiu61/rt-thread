#ifndef FIRMWARE_COMMON_H
#define FIRMWARE_COMMON_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USING_CMODEL
#undef USING_MULTI_THREAD_ENGINE
#endif

#ifndef MASTER_THREAD
#define MASTER_THREAD 0
#endif

#ifndef SLAVE_THREAD
#define SLAVE_THREAD 1
#endif

#include "firmware_common_macro.h"
#include "firmware_common_inline.h"
#include "firmware_debug_macro.h"
#include "gen_cmd.h"

#define TPU0_CLK_ID (0x1 << 0)
#define TPU1_CLK_ID (0x1 << 1)
#define GDMA0_CLK_ID (0x1 << 2)
#define GDMA1_CLK_ID (0x1 << 3)
#define HAU_CLK_ID (0x1 << 4)
#define TC9200_CLK_ID (0x1 << 5)
#define TC9201_CLK_ID (0x1 << 6)

void fix_cmd_node_overflow(CMD_ID_NODE *pid_node);

void cmd_id_divide(CMD_ID_NODE * p_cmd_src, CMD_ID_NODE * p_cmd_dst0, CMD_ID_NODE * p_cmd_dst1);
void cmd_id_merge(CMD_ID_NODE *p_cmd_dst, CMD_ID_NODE *p_cmd_src0, CMD_ID_NODE *p_cmd_src1);

#ifdef __cplusplus
}
#endif

#endif /* FIRMWARE_COMMON_H */
