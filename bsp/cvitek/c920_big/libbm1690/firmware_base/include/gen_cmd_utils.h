#ifndef GEN_CMD_UTILS_H
#define GEN_CMD_UTILS_H

// #include "firmware_common.h"
//#include "gen_cmd.h"
#include "firmware_common_macro.h"

#ifdef USING_CMODEL
#define THREAD __thread
#else
#define THREAD
#endif

typedef struct {
    P_COMMAND buf;
    int       len;
    void (*write)(P_COMMAND, unsigned int, unsigned int);
    unsigned int (*read)(P_COMMAND, unsigned int);
} cmd_t;

typedef struct {
    cmd_t cmd;
    P_COMMAND (*buf)(int);
    void (*id_proc)(unsigned int *, CMD_ID_NODE *, int);
    void (*common)(cmd_t *);
    void (*start)(cmd_t *);
    int thread_id;
} gen_cmd_t;

typedef struct {
    cmd_t cmd;
    P_COMMAND (*buf)(int);
    void (*id_proc)(int, unsigned int *, CMD_ID_NODE *);
    void (*common)(int, cmd_t *);
    void (*start)(int, cmd_t *);
} cdma_gen_cmd_t;

typedef struct {
    cmd_t cmd;
    P_COMMAND (*buf)(int);
    void (*id_proc)(unsigned int *, CMD_ID_NODE *, ENGINE_TYPE, int);
    void (*common)(cmd_t *);
    void (*start)(cmd_t *);
    ENGINE_TYPE sdma_type;
    int port_id;
} sdma_gen_cmd_t;

static inline void write_cmd_32bit(uint64_t base_addr, void *cmd_buf, u32 offset, u32 data) {
    *((u32 *)cmd_buf + offset) = data;
    #ifdef SG_TV_GEN
    uint64_t reg_addr = (base_addr + offset * 4);
    sg_wr_tv_dump_reg(reg_addr, data, NODECHIP_REG);
    #endif
}

static inline u32 read_cmd_32bit(void *cmd_buf, u32 offset) {
    return *((u32 *)cmd_buf + offset);
}

static inline void write_cmd(uint64_t base_addr, cmd_t *cmd, const reg_pack_t *pack) {
    ASSERT(pack->id.len > 0 && (pack->id.len <= WORD_SIZE || pack->id.len == DWORD_SIZE));
    const unsigned int i = pack->id.where >> WORD_BITS;
    if (pack->id.len == DWORD_SIZE) {
        unsigned int w = pack->val & 0xffffffff;
        write_cmd_32bit(base_addr, cmd->buf, i, w);
        w = pack->val >> WORD_SIZE;
        write_cmd_32bit(base_addr, cmd->buf, i + 1, w);
        return;
    } else if (pack->id.len < WORD_SIZE) {
        if (pack->val >= (unsigned int)(1 << pack->id.len)) {
            FW_ERR("val = %lld, bit len = %d bit where is %d\n",
                    pack->val, pack->id.len, pack->id.where);
            ASSERT(0);
        }
    }
    const unsigned int j = pack->id.where & WORD_MASK;
    const unsigned int j_p_len = j + pack->id.len;
    const unsigned int ws_m_j = WORD_SIZE - j;
    if (j_p_len <= WORD_SIZE) {
        /**
         *  ---------------------------------------------------
         *       WORD_SIZE - j - len    |  len  |      j
         *  ---------------------------------------------------
         *   c[j + len : WORD_SIZE - 1] |  val  | c[0 : j - 1]
         *  ---------------------------------------------------
         */
        const unsigned int c = read_cmd_32bit(cmd->buf, i);
        unsigned int w = pack->val << j;
        if (j_p_len < WORD_SIZE)
            w |= c >> j_p_len << j_p_len;
        if (j > 0)
            w |= c << ws_m_j >> ws_m_j;
        write_cmd_32bit(base_addr, cmd->buf, i, w);
    } else {
        /**
         *                         c1                                          c0
         *  ------------------------------------------------  -----------------------------------
         *       WORD_SIZE - len1     |        len1                   len0       |       j
         *  ------------------------------------------------  -----------------------------------
         *   c1[len1 : WORD_SIZE - 1] | val[len0 : len - 1]    val[0 : len0 - 1] | c0[0 : j - 1]
         *  ------------------------------------------------  -----------------------------------
         */
        const unsigned int c0 = read_cmd_32bit(cmd->buf, i);
        const unsigned int c1 = read_cmd_32bit(cmd->buf, i + 1);
        const unsigned int len0 = ws_m_j;
        const unsigned int len1 = pack->id.len - len0;
        const unsigned int ws_m_len0 = WORD_SIZE - len0;
        unsigned int w0 = pack->val << ws_m_len0 >> ws_m_len0 << j;
        w0 |= c0 << len0 >> len0;
        unsigned int w1 = pack->val >> len0;
        w1 |= c1 >> len1 << len1;
        write_cmd_32bit(base_addr, cmd->buf, i, w0);
        write_cmd_32bit(base_addr, cmd->buf, i + 1, w1);
    }
}
#endif // GEN_CMD_UTILS_H
