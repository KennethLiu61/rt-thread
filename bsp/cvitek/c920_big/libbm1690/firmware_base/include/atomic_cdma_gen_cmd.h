#ifndef ATOMIC_CDMA_GEN_CMD_H
#define ATOMIC_CDMA_GEN_CMD_H

#include "firmware_common.h"
#include "cdma_reg_value.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CDMA_MAX_N 65535
#define CDMA_MAX_C 65535
#define CDMA_MAX_H 65535
#define CDMA_MAX_W 65535
#define CDMA_MAX_LEN 65535

/// @brief config cdma route
/// @param route_type
void atomic_cdma_route_configuration(int dst_chipid,
                                     CDMA_ROUTE_TYPE route_type, int port);

void atomic_cdma_pio_reg_descriptor_update(int dst_chipid);

void atomic_cdma_config_tcredit_for_pld_test(int dst_chipid, u64 dst_addr,
                                             int dst_n, int dst_c, int dst_h,
                                             int dst_w, int dst_n_stride,
                                             int dst_c_stride, int dst_h_stride,
                                             int opcode, int port);

void atomic_cdma_pio_interrupt_poll(int dst_chipid);

void reset_cdma(int dst_chipid);

void sr_setup();

void cdma_send_cmodel_gen_cmd(
        int dst_chipid,
        u64 src_addr,
        u16 src_n,
        u16 src_c,
        u32 src_h,
        u32 src_w,
        u32 src_n_stride,
        u32 src_c_stride,
        u32 src_h_stride,
        int dtype,
        int stride_enable,
        int reduce_op,
        int nchw_copy,
        CMD_ID_NODE * pid_node
#if defined(SG_TV_GEN)
        , int is_first_loop,
        int is_last_loop
#endif
        );

void cdma_lossy_compress_gen_cmd(
        int dst_chipid,
        u64 src_addr,
        u16 src_n,
        u16 src_c,
        u32 src_h,
        u32 src_w,
        u32 src_n_stride,
        u32 src_c_stride,
        u32 src_h_stride,
        int dtype,
        int stride_enable,
        int reduce_op,
        int nchw_copy,
        CMD_ID_NODE * pid_node);

void cdma_lossy_decompress_gen_cmd(
        int dst_chipid,
        u64 src_addr,
        u16 src_n,
        u16 src_c,
        u32 src_h,
        u32 src_w,
        u32 src_n_stride,
        u32 src_c_stride,
        u32 src_h_stride,
        int dtype,
        int reduce_op,
        int stride_enable,
        int nchw_copy,
        CMD_ID_NODE * pid_node);

void cdma_recv_gen_cmd(
        int src_chipid,
        int dst_chipid,
        u64 dst_addr,
        u16 dst_n,
        u16 dst_c,
        u32 dst_h,
        u32 dst_w,
        u32 dst_n_stride,
        u32 dst_c_stride,
        u32 dst_h_stride,
        int opcode,
        int dtype,
        int stride_enable,
        CMD_ID_NODE * pid_node
#if defined(SG_TV_GEN)
        , int is_first_loop,
        int is_last_loop,
        int send_tsk_id
#endif
        );

void cdma_p2p_send_gen_cmd(
        int src_chipid,
        int dst_chipid,
        u32 cmd_length,
        u64 src_addr,
        u64 dst_addr,
        int dtype,
        CMD_ID_NODE * pid_node
#if defined(SG_TV_GEN)
        , int is_first_loop,
        int is_last_loop
#endif
        );

void cdma_write_gen_cmd(
        int src_chipid,
        int dst_chipid,
        u64 src_addr,
        u64 dst_addr,
        u16 src_n,
        u16 src_c,
        u32 src_h,
        u32 src_w,
        u32 src_n_stride,
        u32 src_c_stride,
        u32 src_h_stride,
        u16 dst_n,
        u16 dst_c,
        u32 dst_h,
        u32 dst_w,
        u32 dst_n_stride,
        u32 dst_c_stride,
        u32 dst_h_stride,
        int is_fill_const,
        int const_val,
        int dtype,
        int stride_enable,
        int nchw_copy,
        CMD_ID_NODE * pid_node);

void cdma_read_gen_cmd(
        int remote_chipid,
        int chipid,
        u64 src_addr,
        u64 dst_addr,
        u16 src_n,
        u16 src_c,
        u32 src_h,
        u32 src_w,
        u32 src_n_stride,
        u32 src_c_stride,
        u32 src_h_stride,
        u16 dst_n,
        u16 dst_c,
        u32 dst_h,
        u32 dst_w,
        u32 dst_n_stride,
        u32 dst_c_stride,
        u32 dst_h_stride,
        int dtype,
        int stride_enable,
        int nchw_copy,
        CMD_ID_NODE * pid_node);

void cdma_tcp_send_gen_cmd(
        int src_chipid,
        int dst_chipid,
        u16 buffer_length,
        u16 frame_length,
        u64 src_addr,
        int first_desc,
        int last_desc,
        CMD_ID_NODE * pid_node);

void cdma_tcp_recv_gen_cmd(
        int src_chipid,
        int dst_chipid,
        u16 buffer_length,
        u16 send_frame_length,
        u64 dst_addr,
        int first_desc,
        int last_desc,
        CMD_ID_NODE * pid_node);

#ifdef USING_CMODEL
void cdma_fake_all_reduce_gen_cmd(
        int dst_chipid,
        u64 src_addr,
        int src_n,
        int src_c,
        int src_h,
        int src_w,
        int src_n_stride,
        int src_c_stride,
        int src_h_stride,
        int opcode,
        int dtype,
        CMD_ID_NODE * pid_node);

void cdma_fake_p2p_gen_cmd(
        int dst_chipid,
        u64 src_addr,
        int src_n,
        int src_c,
        int src_h,
        int src_w,
        int src_n_stride,
        int src_c_stride,
        int src_h_stride,
        int opcode,
        int dtype,
        CMD_ID_NODE * pid_node);
#endif

#ifdef SG_TV_GEN
static int read_env_variable_(const char *key) {
  char *val = getenv(key);
  if (val == NULL) {
    return -1;
  }
  int int_val = atoi(val);
  if (int_val < 0 || int_val > 1)
    int_val = 0;
  return int_val;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // ATOMIC_CDMA_GEN_CMD_H
