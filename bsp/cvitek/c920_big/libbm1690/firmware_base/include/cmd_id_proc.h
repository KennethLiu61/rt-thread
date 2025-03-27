#ifndef CMD_ID_PROC_H_
#define CMD_ID_PROC_H_
#ifdef USING_CMODEL
#include "store_cmd.h"
#endif

#ifndef sg_max
#define sg_max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#define BD_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id) \
  thread_id == 0 ? pid_node->bd_cmd_id++ : pid_node->slave_bd_cmd_id++;    \
  cmd_id0 = pid_node->bd_cmd_id;              \
  cmd_id1 = pid_node->gdma_cmd_id;            \
  cmd_id2 = pid_node->hau_cmd_id;             \
  cmd_id3 = pid_node->slave_bd_cmd_id;

#define GDMA_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id) \
  thread_id == 0 ? pid_node->gdma_cmd_id++ : pid_node->slave_gdma_cmd_id++; \
  cmd_id0 = pid_node->bd_cmd_id;                \
  cmd_id1 = pid_node->gdma_cmd_id;              \
  cmd_id2 = pid_node->hau_cmd_id;               \
  cmd_id3 = pid_node->slave_gdma_cmd_id;

#define HAU_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2) \
  pid_node->hau_cmd_id++;                       \
  cmd_id0 = pid_node->bd_cmd_id;                \
  cmd_id1 = pid_node->gdma_cmd_id;              \
  cmd_id2 = pid_node->hau_cmd_id;

#ifdef SG_STAS_GEN
#define SDMA_CMD_ID_PROC_CORE() \
  pid_node->sdma_cmd_id++;      \
  sg_stas_add_node(pid_node, ENGINE_SDMA);         \
  pid_node->cycle_count += pid_node->cur_op_cycle;
#else
#define SDMA_CMD_ID_PROC_CORE() \
  pid_node->sdma_cmd_id++;
#endif

#define VSDMA_CMD_ID_PROC_CORE(port) \
  pid_node->vsdma_cmd_id[port]++;

#define CDMA_CMD_ID_PROC_CORE(port) \
  pid_node->cdma_cmd_id[port]++;

#ifdef SG_STAS_GEN
#define BD_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)       \
     BD_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)     \
     if (pid_node != NULL) {                            \
       sg_stas_add_node(pid_node, ENGINE_BD);           \
       pid_node->cycle_count += pid_node->cur_op_cycle; \
     }
#else
#define BD_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id) BD_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)
#endif

#ifdef SG_STAS_GEN
#define GDMA_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)    \
    GDMA_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)   \
    if (pid_node != NULL) {                            \
      sg_stas_add_node(pid_node, ENGINE_GDMA);         \
      pid_node->cycle_count += pid_node->cur_op_cycle; \
    }
#else
#define GDMA_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id) GDMA_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2, cmd_id3, thread_id)
#endif

#ifdef SG_STAS_GEN
#define HAU_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2)     \
    HAU_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2)    \
    if (pid_node != NULL) {                            \
      sg_stas_add_node(pid_node, ENGINE_HAU);       \
      pid_node->cycle_count += pid_node->cur_op_cycle; \
    }
#else
#define HAU_CMD_ID_PROC(cmd_id0, cmd_id1, cmd_id2) HAU_CMD_ID_PROC_CORE(cmd_id0, cmd_id1, cmd_id2)
#endif
#endif

#define CDMA_CMD_ID_PROC(port) CDMA_CMD_ID_PROC_CORE(port)