#ifndef GEN_CMD_H
#define GEN_CMD_H
#include "base_def.h"
#ifdef USING_CMODEL
#include <unistd.h>
#include "store_cmd.h"
#else
#include "bd_reg_value.h"
#endif
#include "firmware_profile.h"
#include "gen_cmd_utils.h"
#include "gen_cmd_bd.h"
#include "gen_cmd_gdma.h"
#include "gen_cmd_sdma.h"
#include "gen_cmd_hau.h"
#include "gen_cmd_cdma.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USING_CMODEL

  #define WRITE_CMD_128BIT(base, offset, h64, l64)                                               \
      do {                                                                                       \
          write_cmd_32bit(base, fast_cmd->cmd.buf, (offset << 2), l64 & 0xffffffff);             \
          write_cmd_32bit(base, fast_cmd->cmd.buf, (offset << 2) + 1, (l64 >> 32) & 0xffffffff); \
          write_cmd_32bit(base, fast_cmd->cmd.buf, (offset << 2) + 2, h64 & 0xffffffff);         \
          write_cmd_32bit(base, fast_cmd->cmd.buf, (offset << 2) + 3, (h64 >> 32) & 0xffffffff); \
      } while (0)
  #define WRITE_CMD_CDMA(port, offset, h64, l64) \
          WRITE_CMD_128BIT(CDMA_CMD_BASE_ADDR(port), offset, h64, l64)
   #define WRITE_CMD_EX(addr, offset, h64, l64) WRITE_CMD_128BIT(addr, offset, h64, l64)
   #define WRITE_CMD_EX_32BIT(addr, offset, h64, l64) WRITE_CMD_128BIT(addr, offset, h64, l64)
   #define WRITE_CMD_32BIT(addr, offset, value) \
          write_cmd_32bit(addr, fast_cmd->cmd.buf, offset, value);
#else
  #define WRITE_CMD_CDMA(port, offset, h64, l64) write128((uint64_t)map_to_kaddr(CDMA_CMD_BASE_ADDR(port)) + (offset << 4), l64, h64)
  #define WRITE_CMD_EX(addr, offset, h64, l64) \
          (void)(addr); \
          write128((uint64_t)fast_cmd->cmd.buf + (offset << 4), l64, h64)

  #define WRITE_CMD_32BIT(addr, offset, value) \
          (void)(addr); \
          write32((uint64_t)fast_cmd->cmd.buf + (offset << 2), value);

  #define WRITE_CMD_EX_32BIT(addr, offset, h64, l64) \
          (void)(addr); \
          write32((uint64_t)fast_cmd->cmd.buf + (offset << 4), (l64 & 0xffffffff)); \
          write32((uint64_t)fast_cmd->cmd.buf + (offset << 4) + 4, (l64 >> 32)); \
          write32((uint64_t)fast_cmd->cmd.buf + (offset << 4) + 8, (h64 & 0xffffffff)); \
          write32((uint64_t)fast_cmd->cmd.buf + (offset << 4) + 12, (h64 >> 32)); \

#endif

#define BEGIN_FAST_GEN_CMD_CDMA(port)                         \
    u32        id[3];                                         \
    cdma_gen_cmd_t *fast_cmd;                                 \
    fast_cmd = CDMA_begin_fast_gen_cmd(port, id, pid_node);

#define END_FAST_GEN_CMD_CDMA(port, pid_node)                 \
    CDMA_end_fast_gen_cmd(port, fast_cmd, pid_node);

#define BEGIN_FAST_GEN_CMD(engine)                           \
    u32 id[3];                                               \
    gen_cmd_t *fast_cmd;                                     \
    fast_cmd = engine##_begin_fast_gen_cmd(id, pid_node);

#define BEGIN_FAST_GEN_CMD_BD(thread_id)                          \
    u32        id[4];                                             \
    gen_cmd_t *fast_cmd;                                       \
    fast_cmd = BD_begin_fast_gen_cmd(id, pid_node, thread_id);

#define END_FAST_GEN_CMD_BD(pid_node)                      \
    BD_end_fast_gen_cmd(fast_cmd, pid_node);

#define BEGIN_FAST_GEN_CMD_SDMA(port_id)                          \
    u32        id[3];                                               \
    sdma_gen_cmd_t *fast_cmd;                                       \
    fast_cmd = SDMA_begin_fast_gen_cmd(id, pid_node, port_id);

#define END_FAST_GEN_CMD_SDMA(pid_node)                      \
    SDMA_end_fast_gen_cmd(fast_cmd, pid_node);

#define BEGIN_FAST_GEN_CMD_GDMA(thread_id)                          \
    u32        id[4];                                               \
    gen_cmd_t *fast_cmd;                                       \
    fast_cmd = GDMA_begin_fast_gen_cmd(id, pid_node, thread_id);

#define END_FAST_GEN_CMD_GDMA(pid_node)                      \
    GDMA_end_fast_gen_cmd(fast_cmd, pid_node);

#define END_FAST_GEN_CMD(engine, pid_node)                      \
    engine##_end_fast_gen_cmd(fast_cmd, pid_node);

#define SET_CMD_THREAD_ID(engine, thread_id)                   \
    engine##_set_cmd_thread_id(fast_cmd, thread_id);

#define READ_GDMA_FILTER_RES_NUM(val, pid_node)                                \
  do {                                                                         \
    const reg_id_t reg_filter = GDMA_ID_CFG_FILTER_NUM;                        \
    val = READ_REG(GDMA_ENGINE_MAIN_CTRL + ((reg_filter.where >> 5) << 2));    \
    val >>= (reg_filter.where & 31);                                           \
    val &= ((1ull << reg_filter.len) - 1);                                     \
  } while (0)

#define READ_SDMA_FILTER_RES_NUM(val, port_id, pid_node)                       \
  do {                                                                         \
    if (port_id == -1) {                                                       \
      const reg_id_t reg_filter = SDMA_ID_CFG_FILTER_NUM;                      \
      val = READ_REG(SDMA_ENGINE_MAIN_CTRL + ((reg_filter.where >> 5) << 2));  \
      val >>= (reg_filter.where & 31);                                         \
      val &= ((1ull << reg_filter.len) - 1);                                   \
    } else {                                                                   \
      const reg_id_t reg_filter = VSDMA_ID_CFG_FILTER_NUM;                     \
      val = READ_REG(VSDMA_ENGINE_MAIN_CTRL(port_id) +                         \
                     ((reg_filter.where >> 5) << 2));                          \
      val >>= (reg_filter.where & 31);                                         \
      val &= ((1ull << reg_filter.len) - 1);                                   \
    }                                                                          \
  } while (0)

#ifdef __cplusplus
}
#endif
#endif  // GEN_CMD_H
