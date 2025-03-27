#ifndef ATOMIC_GDMA_GEN_CMD_H
#define ATOMIC_GDMA_GEN_CMD_H
#include "firmware_common.h"
#include "gdma_reg_value.h"
#include "atomic_gen_cmd.h"
#ifdef __cplusplus
extern "C" {
#endif

#define GDMA_MAX_N 65535
#define GDMA_MAX_C 65535
#define GDMA_MAX_H 65535
#define GDMA_MAX_W 65535
#ifndef MAX_WSTRIDE_BYTE_LEN
#define MAX_WSTRIDE_BYTE_LEN 128
#endif

void tensor_align_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    int src_format,
    int direction,
    int transpose,  // N/C transpose
    int thread_id,
    CMD_ID_NODE * pid_node);

void tensor_stride_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int src_format,
    int direction,
    int transpose,  // N/C transpose
    CMD_ID_NODE * pid_node);

void tensor_general_move_gen_cmd(
    u64 src_addr, //local_addr or global_addr
    int src_local_idx, //use only from local_mem
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,

    u64 dst_addr, //local_addr or global_addr
    int dst_local_idx, //use only to local_mem
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int direction,
    int transpose,  // N/C transpose
    CMD_ID_NODE * pid_node);

void tensor_compact_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N, int src_C, int src_H, int src_W,
    int src_format,
    int direction,
    int transpose,  // N/C transpose
    int thread_id,
    CMD_ID_NODE * pid_node);

//col_num and row_num mean matrix size in system_mem, no matter transpose
//note: when transpose, S(2x3)-->L(3x2), config row_num=2, col_num=3
//note: when transpose, L(3x2)-->S(2x3), config row_num=2, col_num=3
void matrix_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int sec_size,
    int row_num, int col_num, //means matrix in sys_mem  is row*col,
    int src_format,
    int direction,
    int transpose,
    int thread_id,
    CMD_ID_NODE * pid_node);

void matrix_stride_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int sec_size,
    int row_num, int col_num,
    stride_type global_row_stride,
    stride_type local_row_stride,
    stride_type local_sec_stride,
    int src_format,
    int direction,
    int transpose,
    int thread_id,
    CMD_ID_NODE * pid_node);

void general_matrix_move_gen_cmd(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int sec_size,
    int row_num, int col_num,
    stride_type row_stride,
    int src_format,
    int direction,
    int transpose,
    int thread_id,
    CMD_ID_NODE * pid_node);

// if use_broadcast = 1, dst_W_stride must be 1 or stride_enable == 1, and dst_C + local_mem_idx <= NPU_NUM
void fill_constant_gen_local_cmd_stride(
    int local_mem_start_addr,
    int local_mem_idx,
    const void *const_val,
    int data_format,
    int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int stride_enable,
    int use_broadcast,
    int thread_id,
    CMD_ID_NODE * pid_node);

void fill_constant_gen_global_cmd_stride(
    u64 sys_mem_start_addr,
    const void* const_val,
    int data_format,
    int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int stride_enable,
    int thread_id,
    CMD_ID_NODE * pid_node);

void general_gdma_gen_cmd(
    u64 src_addr, // Absolute addr
    u64 dst_addr, // Absolute addr
    int src_format,
    stride_type src_count, //tv_gen: default=1
    int src_is_const,
    int thread_id,
    CMD_ID_NODE * pid_node);

// This operation broadcast linear data to local memory
// Only support S2L, L2L or Const to lmem
void general_broadcast_gen_cmd(
    u64 src_addr, // src_addr(absolute addr) or constant
    int local_mem_start_addr,
    int local_mem_idx,
    int src_format,
    stride_type src_count,
    int dst_c, // Broadcast src_count data to dst_c lanes, local_idx + dst_c <= NPU_NUM
    int src_is_const, // 0: not const, 1: is const
    int thread_id,
    CMD_ID_NODE * pid_node);

// SRC [N, C, H, W] -> DST [N, W, H, C]
void general_cwtrans_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    int src_N,  int src_C,
    int src_H,  int src_W,
    int src_format,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int stride_enable,
    int direction,  // Support S2S, S2L, L2S, L2L
    int thread_id,
    CMD_ID_NODE * pid_node);

void tensor_general_move_with_mask_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 mask_addr, // local_addr or global_addr
    int mask_local_idx, // use only from local_mem
    int mask_in_lmem, // 1: mask is in local mem, 0: mask is in global mem
    u64 dst_addr, // global addr only
    int src_format,
    int mask_format,
    int N,
    int C,
    int H,
    int W,
    int direction, // src to dst direction, only support L2S, S2S
    int thread_id,
    CMD_ID_NODE * pid_node);

void tensor_move_nonzero_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 dst_addr, // global addr only
    int src_format,
    int dst_format, // Only INT8/INT16/INT32
    int N,
    int C,
    int H,
    int W,
    u32 base_idx,
    int direction, // only support L2S, S2S
    int thread_id,
    CMD_ID_NODE * pid_node);

// This function is used after tensor_general_move_with_mask_gen_cmd or
// tensor_move_nonzero_gen_cmd
unsigned int get_gdma_filter_res_num_gen_cmd(CMD_ID_NODE * pid_node);

// Channel broadcast, src_C = 1
void tensor_broadcast_move_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    int dst_lmem_start_addr, //local_addr
    int dst_local_idx,
    int src_N,
    int src_H,
    int src_W,
    int dst_C, // Restriction: dst_local_idx + dst_C <= NPU_NUM
    stride_type src_N_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_H_stride,
    int data_format,
    int stride_enable,
    int direction, // Only support, S2L, L2L
    CMD_ID_NODE * pid_node);

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_gather_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 index_addr, // local_addr or global_addr, index_format is always UINT32
    int index_local_idx, // use only from local_mem
    int index_in_lmem, // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    u64 const_val, // if index >= src_H, then get const val
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 index_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride, // Must be 1 because of IC constraint if stride_enable = 1
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int direction, // Support S2S, S2L, L2S, L2L
    int thread_id,
    CMD_ID_NODE * pid_node);

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_scatter_gen_cmd(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 index_addr, // local_addr or global_addr, index_format is always UINT32
    int index_local_idx, // use only from local_mem
    int index_in_lmem, // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 dst_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride, // Must be 1 because of IC constraint if stride_enable = 1
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int direction, // Support S2S, S2L, L2S, L2L
    int inplace_add,
    int thread_id,
    CMD_ID_NODE * pid_node);

// only support l2s
void tensor_normal_compress_gen_cmd(
    uint32_t local_mem_addr,
    u64      sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node);

// only support s2l
void tensor_normal_decompress_gen_cmd(
    uint32_t local_mem_addr,
    u64      sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node);

// only support l2s
void tensor_racu_compress_gen_cmd(
    uint32_t local_mem_addr,
    u64      racu_sys_mem_addr,
    u64      meta_sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint32_t racu_n_stride,
    uint32_t racu_c_stride,
    uint32_t racu_h_stride,
    uint32_t meta_n_stride,
    uint32_t meta_c_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node);

// only support s2l
void tensor_racu_decompress_gen_cmd(
    uint32_t local_mem_addr,
    u64      racu_sys_mem_addr,
    u64      meta_sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint32_t racu_n_stride,
    uint32_t racu_c_stride,
    uint32_t racu_h_stride,
    uint32_t meta_n_stride,
    uint32_t meta_c_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format,
    int thread_id,
    CMD_ID_NODE *pid_node);

void tensor_gdma_reverse_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t src_n_stride,
    uint32_t src_c_stride,
    uint32_t src_h_stride,
    uint32_t dst_n_stride,
    uint32_t dst_c_stride,
    uint32_t dst_h_stride,
    int32_t reverse_axis,
    int32_t data_format,
    int32_t direction,
    int thread_id,
    CMD_ID_NODE *pid_node);

void gdma_lossy_compress_gen_cmd(
    u64 src_addr, // local_addr or sys_addr
    int src_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction,
    int thread_id,
    CMD_ID_NODE * pid_node);

void gdma_lossy_decompress_gen_cmd(
    u64 src_addr, // sys_addr
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // local_addr or sys_addr
    int dst_local_idx,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction,
    int thread_id,
    CMD_ID_NODE * pid_node);

void gdma_lossy_compress_reduce_gen_cmd(
    u64 src_addr, // local_addr or sys_addr
    int src_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction,
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node);

void gdma_lossy_decompress_reduce_gen_cmd(
    u64 src_addr, // sys_addr
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // local_addr or sys_addr
    int dst_local_idx,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction,
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node);

void tensor_stride_move_reduce_gen_cmd(
        int local_mem_start_addr,
        int local_mem_idx,
        u64 sys_mem_start_addr,
        int src_N,
        int src_C,
        int src_H,
        int src_W,
        stride_type src_N_stride,
        stride_type src_C_stride,
        stride_type src_H_stride,
        stride_type src_W_stride,
        stride_type dst_N_stride,
        stride_type dst_C_stride,
        stride_type dst_H_stride,
        stride_type dst_W_stride,
        int src_format,
        int direction,
        int transpose,  // N/C transpose
        int reduce_psum_op,
        int reduce_opcode,
        int thread_id,
        CMD_ID_NODE * pid_node);

void tensor_general_move_reduce_gen_cmd(
    u64 src_addr, //local_addr or global_addr
    int src_local_idx, //use only from local_mem
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr, //local_addr or global_addr
    int dst_local_idx, //use only to local_mem
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int direction,
    int transpose,  // N/C transpose
    int reduce_psum_op,
    int reduce_opcode,
    int thread_id,
    CMD_ID_NODE * pid_node);

void tensor_general_move_local_cross_core_gen_cmd(
    u64 src_addr,
    int src_local_idx,
    int src_core_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,

    u64 dst_addr,
    int dst_local_idx,
    int dst_core_idx,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int thread_id,
    CMD_ID_NODE * pid_node);

void atomic_transfer_general_gen_cmd(
    u64 src_addr, //local_addr or smem_addr
    int src_core_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr, //local_addr or smem_addr
    int dst_core_idx,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int thread_id,
    CMD_ID_NODE * pid_node);

void random_mask_init_seed_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int dst_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    uint64_t size,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int format,
    int thread_id,
    CMD_ID_NODE * pid_node);

void random_mask_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int dst_local_idx,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    uint64_t size,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int format,
    int inter_state,
    int thread_id,
    CMD_ID_NODE * pid_node);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_DMA_GEN_CMD_H */
