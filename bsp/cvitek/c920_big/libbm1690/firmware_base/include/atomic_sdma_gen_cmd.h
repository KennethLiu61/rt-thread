#ifndef ATOMIC_SDMA_GEN_CMD_H
#define ATOMIC_SDMA_GEN_CMD_H
#include "firmware_common.h"
#include "sdma_reg_value.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SDMA_MAX_N 65535
#define SDMA_MAX_C 65535
#define SDMA_MAX_H 65535
#define SDMA_MAX_W 65535
#ifndef SDMA_MAX_WSTRIDE_BYTE_LEN
#define SDMA_MAX_WSTRIDE_BYTE_LEN 64
#endif
#define DEFAULT_SDMA_PORT (-1)

void sdma_tensor_general_move_gen_cmd(
    u64 src_addr,
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
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose,  // N/C transpose
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_fill_constant_gen_global_cmd_stride(
    u64 sys_mem_start_addr,
    const void* const_val,
    int data_format,
    int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int stride_enable,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_general_gen_cmd(
    u64 src_addr, // Absolute addr
    u64 dst_addr, // Absolute addr
    int src_format,
    stride_type src_count, //tv_gen: default=1
    int src_is_const,
    int port_id,
    CMD_ID_NODE * pid_node);

// SRC [N, C, H, W] -> DST [N, W, H, C]
void sdma_general_cwtrans_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
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
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_tensor_general_move_with_mask_gen_cmd(
    u64 src_addr,
    u64 mask_addr,
    u64 dst_addr, // global addr only
    int src_format,
    int mask_format,
    int N,
    int C,
    int H,
    int W,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_tensor_move_nonzero_gen_cmd(
    u64 src_addr,
    u64 dst_addr,
    int src_format,
    int dst_format, // Only INT8/INT16/INT32
    int N,
    int C,
    int H,
    int W,
    u32 base_idx,
    int port_id,
    CMD_ID_NODE * pid_node);

// This function is used after tensor_general_move_with_mask_gen_cmd or
// tensor_move_nonzero_gen_cmd
unsigned int get_sdma_filter_res_num_gen_cmd(int port_id, CMD_ID_NODE * pid_node);

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_gather_gen_cmd(
    u64 src_addr,
    u64 index_addr,
    u64 dst_addr,
    u32 const_val, // if index >= src_H, then get const val
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 index_H,
    u32 start_pos, // if index < start_pos, get const val
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
    int port_id,
    CMD_ID_NODE * pid_node);

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_scatter_gen_cmd(
    u64 src_addr,
    u64 index_addr, // index_format is always UINT32
    u64 dst_addr,
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
    int port_id,
    int inplace_add,
    CMD_ID_NODE * pid_node);

void sdma_tensor_reverse_gen_cmd(
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
    int port_id,
    CMD_ID_NODE *pid_node);

void sdma_lossy_compress_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_lossy_decompress_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_lossy_compress_reduce_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_lossy_decompress_reduce_gen_cmd(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node);

void sdma_tensor_reduce_gen_cmd(
    u64 src_addr, //local_addr or global_addr
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
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose,  // N/C transpose
    int reduce_psum_op,
    int reduce_opcode,
    int port_id,
    CMD_ID_NODE * pid_node);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_DMA_GEN_CMD_H */
