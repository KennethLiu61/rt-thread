#ifndef ATOMIC_RANDOM_GEN_GEN_CMD_H
#define ATOMIC_RANDOM_GEN_GEN_CMD_H
#include "firmware_common.h"
#include "bd_reg_value.h"
#ifdef __cplusplus
extern "C" {
#endif

void atomic_random_gen_init_seed_gen_cmd(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    int jump_cnt,
    int c_offset,
    unsigned int store_state_addr,
    int need_store,
    int thread_id,
    CMD_ID_NODE * pid_node);

void atomic_random_gen_gen_cmd(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    unsigned int load_state_addr,
    unsigned int store_state_addr,
    int need_store,
    RAND_OP op_type,
    int thread_id,
    CMD_ID_NODE * pid_node);

#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_RANDOM_GEN_GEN_CMD_H */
