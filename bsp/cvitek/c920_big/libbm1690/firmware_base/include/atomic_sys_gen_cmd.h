#ifndef ATOMIC_SYS_GEN_CMD_H
#define ATOMIC_SYS_GEN_CMD_H

#include "firmware_common.h"
#ifdef __cplusplus
extern "C" {
#endif

void atomic_bd_end_gen_cmd(CMD_ID_NODE *pid_node);
void atomic_bd_nop_gen_cmd(CMD_ID_NODE *pid_node);
// software power boot, this is useless for software
void atomic_bd_spb_gen_cmd(CMD_ID_NODE *pid_node);

void atomic_bd_rand_seed_gen_cmd(int thread_id, CMD_ID_NODE *pid_node);

// void atomic_bd_nop_gen_cmd(int thread_id, CMD_ID_NODE *pid_node);

void atomic_bd_fork_gen_cmd(CMD_ID_NODE *pid_node);

void atomic_bd_join_gen_cmd(CMD_ID_NODE *pid_node);

void atomic_bd_exit_gen_cmd(CMD_ID_NODE *pid_node);

// this atomic can't sync gdma id
void atomic_bd_swr_gen_cmd(
    u32 src_addr, // local addr
    int src_is_const, u64 src_const_value,
    int short_term_valid, // 0: clear valid bit, others: set valid bit
    int long_term_valid,  // 0: clear valid bit, others: set valid bit
    BD_SYS_TYPE sys_type, int thread_id, CMD_ID_NODE *pid_node);

// gen command for TIU general control register
void atomic_bd_trwr_gen_cmd(const u32 *value, const int *indice, int count, int thread_id, 
                            CMD_ID_NODE *pid_node);
void atomic_set_bdid_gen_cmd(u32 bdc_id, CMD_ID_NODE *pid_node);
void atomic_set_tpu_mode_gen_cmd(int32_t mode, CMD_ID_NODE *pid_node);

void atomic_sdma_end_gen_cmd(CMD_ID_NODE *pid_node, int port_id);
void atomic_sdma_nop_gen_cmd(CMD_ID_NODE *pid_node, int port_id);
void atomic_set_sdma_id_gen_cmd(u32 sdma_id, CMD_ID_NODE *pid_node, int port_id);
void atomic_gdma_end_gen_cmd(CMD_ID_NODE *pid_node);
void atomic_gdma_nop_gen_cmd(CMD_ID_NODE *pid_node, int thread_id);
void atomic_set_gdma_id_gen_cmd(u32 gdma_id, CMD_ID_NODE *pid_node);

void atomic_send_msg_gen_cmd(ENGINE_TYPE eng_type, int32_t msg_id,
                             int32_t wait_cnt, int thread_id, CMD_ID_NODE *pid_node);
void atomic_wait_msg_gen_cmd(ENGINE_TYPE eng_type, int32_t msg_id,
                             int32_t send_cnt, int thread_id, CMD_ID_NODE *pid_node);
void atomic_cdma_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_tx_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_tx_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_rx_send_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t wait_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_rx_wait_msg_gen_cmd(int32_t port, int32_t msg_id, int32_t send_cnt,
                                  CMD_ID_NODE *pid_node);
void atomic_cdma_end_gen_cmd(int32_t port, CMD_ID_NODE *pid_node);
void atomic_cdma_nop_gen_cmd(int32_t port, CMD_ID_NODE *pid_node);
void atomic_set_cdma_id_gen_cmd(int32_t port, u32 sdma_id, CMD_ID_NODE *pid_node);

/// @brief set gdma/sort base ddr, so real ddr is ddr[34:0] +  ddr[38:36]
/// @param base_idx
/// @param base_addr base
/// @param count base_ddr count
/// @param eng_type ENGINE_GDMA, ENGINE_HAU support
void atomic_set_base_ddr(const int *base_idx, const u64 *base_addr, int count,
                         ENGINE_TYPE eng_type);

/// @brief set cdma base ddr for specified port
/// @param base_idx
/// @param base_addr base
/// @param count base_ddr count
/// @param cdma_port CDMA port
void atomic_cdma_port_set_base_ddr(const int *base_idx, const u64 *base_addr, int count,
                         int cdma_port);

/// @brief set gdma random mask csr seed, 64bit
/// @param seed
void atomic_set_gdma_random_mask_seed(uint64_t seed);

/// @brief set message base idx, real msg idx = msg_base_idx + current_msg_idx
/// @param base_idx msessage base idx
/// @param eng_type ENGINE_BD, ENGINE_GDMA, ENGINE_HAU support
void atomic_set_base_msg_id(int32_t base_idx, ENGINE_TYPE eng_type);
void atomic_cdma_port_set_base_msg_id(int32_t base_idx, int32_t port);

void atomic_set_bd_random_gen_seed(uint64_t seed);

int atomic_cdma_send_cmodel_gen_cmd(uint8_t dst_chipid,
                             uint32_t src_base_addr,
                             uint32_t src_n, uint32_t src_c, uint32_t src_h, uint32_t src_w,
                             uint32_t src_n_stride, uint32_t src_c_stride, uint32_t src_h_stride,
                             uint8_t opcode,
                             uint8_t dtype);

int atomic_cdma_receive_gen_cmd(uint8_t src_chipid,
                                uint32_t dst_base_addr,
                                uint32_t dst_n, uint32_t dst_c, uint32_t dst_h, uint32_t dst_w,
                                uint32_t dst_n_stride, uint32_t dst_c_stride, uint32_t dst_h_stride,
                                uint8_t dst_mem_type);
#ifdef __cplusplus
}
#endif

#endif
