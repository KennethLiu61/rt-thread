#include "firmware_common.h"
#include "atomic_cdma_gen_cmd.h"
#include "atomic_dma_utils.h"
#include "cdma_reg_def.h"
#include "cdma_reg_value.h"
#include "gen_cmd.h"
#include "tpu_defs.h"
#include "tpu_kernel.h"
#include "bringup_cdma.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif
#define ASSERT_SYS_STRIDE_TV_GEN_EARLY_DEBUG(sn, sc, sh, n, c, h, w) \
    ASSERT_FS_INFO(sh > w + 64, "sh=%d, w=%d", sh, w); \
    ASSERT_FS_INFO(sc > h * sh + 64, "sc=%d, h * sh=%d", sc, h * sh); \
    ASSERT_FS_INFO(sn > c * sc + 64, "sn=%d, c*sc=%d", sn, c * sc);

#define ASSERT_CDMA_TENSOR_NSIZE(n) \
    ASSERT_FS_INFO(n>0 && n<=CDMA_MAX_N, #n "=%d", n)

#define ASSERT_CDMA_TENSOR_CSIZE(c) \
    ASSERT_FS_INFO(c>0 && c<=CDMA_MAX_C, #c "=%d", c)

#define ASSERT_CDMA_TENSOR_HSIZE(h) \
    ASSERT_FS_INFO(h>0 && h<=CDMA_MAX_H, #h "=%d", h)

#define ASSERT_CDMA_TENSOR_WSIZE(w) \
    ASSERT_FS_INFO(w>0 && w<=CDMA_MAX_W, #w "=%d", w)

#define ASSERT_CDMA_TENSOR_SIZE(n,c,h,w) \
    ASSERT_CDMA_TENSOR_NSIZE(n); \
    ASSERT_CDMA_TENSOR_CSIZE(c); \
    ASSERT_CDMA_TENSOR_HSIZE(h); \
    ASSERT_CDMA_TENSOR_WSIZE(w)

#define ASSERT_CDMA_TENSOR_LENGTH(length) \
    ASSERT_FS_INFO(length>0 && length<=CDMA_MAX_LEN, #length "=%d", length)

int lookup_send_port(int dst_chipid) {
#ifdef USE_DEBUG_CDMA_PORT
  return DEBUG_CDMA_PORT;
#else
  return get_c2c_port(tpu_chip_id(), dst_chipid, C2C_SEND);
#endif
}

int lookup_recv_port(int src_chipid) {
#ifdef USE_DEBUG_CDMA_PORT
  return DEBUG_CDMA_PORT;
#else
  return get_c2c_port(tpu_chip_id(), src_chipid, C2C_RECV);
#endif
}

static inline int get_constant_value(const void * p_val, int format) {
    int constant = 0;
    int type_len = get_cdma_format_type_len(format);
    if (format == CDMA_DTYPE_FP20) {
        type_len = 4;
    }
    memcpy(&constant, p_val, type_len);
    return constant;
}

void sr_setup(){
  {
    u32 tmp = 0;
    (void)tmp;
    tmp = (C2C_PCIE_CTRL << 28) |
        (CDMA_CMD_BASE_ADDR_PAIR >> 32);
    WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_H32, tmp);
    tmp = (CDMA_CMD_BASE_ADDR_PAIR & ((1ul << 32) - 1)) >> 16;
    WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_M16, tmp);
    tmp = READ_CSR_CDMA(CDMA_CSR_4) | (1 << CDMA_CSR_RCV_CMD_OS);
    WRITE_CSR_CDMA(CDMA_CSR_4, tmp);
  }
  {
    u32 tmp = 0;
    (void)tmp;
    tmp = (READ_CSR_CDMA(CDMA_CSR_INTER_DIE_RW_V2) &
          ~(0xff << CDMA_CSR_INTER_DIE_WRITE_ADDR_L4)) |
          (C2C_PCIE_CTRL << CDMA_CSR_INTER_DIE_WRITE_ADDR_H4) |
          (0b0000 << CDMA_CSR_INTER_DIE_WRITE_ADDR_L4);
    WRITE_CSR_CDMA(CDMA_CSR_INTER_DIE_RW_V2, tmp);
    tmp = (READ_CSR_CDMA(CDMA_CSR_INTRA_DIE_RW_V2) &
          ~(0xff << CDMA_CSR_INTRA_DIE_READ_ADDR_L4)) |
          (AXI_NOC << CDMA_CSR_INTRA_DIE_READ_ADDR_H4) |
          (0b0000 << CDMA_CSR_INTRA_DIE_READ_ADDR_L4);
    WRITE_CSR_CDMA(CDMA_CSR_INTRA_DIE_RW_V2, tmp);
  }
}

void cdma_send_cmodel_gen_cmd(int dst_chipid, u64 src_addr, u16 src_n, u16 src_c,
                       u32 src_h, u32 src_w, u32 src_n_stride, u32 src_c_stride,
                       u32 src_h_stride, int dtype, int stride_enable,
                       int reduce_op, int nchw_copy, CMD_ID_NODE *pid_node
#if defined(SG_TV_GEN)
                       ,int is_first, int is_last
#endif
) {
  FW_DBG("%s: dst_chipid = %d, src_addr = 0x%llx, "
         "src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
         "src_n_stride = %d, src_c_stride = %d, src_h_stride = %d, "
         "dtype = %d, stride_enable = %d, nchw_copy = %d\n",
         __func__, dst_chipid, src_addr, src_n, src_c, src_h, src_w,
         src_n_stride, src_c_stride, src_h_stride, dtype, stride_enable,
         nchw_copy);
#if defined(SG_TV_GEN)
  // set_first_last_flag_for_cdma_tv_gen_batch_send_recv_test(is_first, is_last);
#endif
#ifdef USING_CMODEL
  // ASSERT_CDMA_TENSOR_SIZE(src_n, src_c, src_h, src_w);
  if (dtype == CDMA_DTYPE_FP20) {
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(!is_lmem(src_addr), "can't be local memory, src_addr:0x%llx",
                   src_addr);
  }
  ASSERT_FS_INFO(!is_smem(src_addr), "can't be static memory, src_addr:0x%llx",
                 src_addr);
  if (reduce_op == CDMA_OPCODE_ADD || reduce_op == CDMA_OPCODE_MAX ||
      reduce_op == CDMA_OPCODE_MIN)
    ASSERT((dtype == CDMA_DTYPE_FP20) || (dtype == CDMA_DTYPE_FP32) ||
           (dtype == CDMA_DTYPE_FP16) || (dtype == CDMA_DTYPE_BF16) ||
           (dtype == CDMA_DTYPE_INT32));
  if (reduce_op == CDMA_OPCODE_MUL)
    ASSERT((dtype == CDMA_DTYPE_FP20) || (dtype == CDMA_DTYPE_FP32) ||
           (dtype == CDMA_DTYPE_FP16) || (dtype == CDMA_DTYPE_BF16));
#endif

  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  int psum_op = (reduce_op == CDMA_OPCODE_NONE ? PSUM_OP_WO : PSUM_OP_WR);
  //WRITE_REG(0x6c007d0024 , 0x0, NODECHIP_REG);
  //WRITE_REG(0x6c007d0028 , 0x6c, NODECHIP_REG);
  //WRITE_REG(0x6c007d002c , 0x01ffffff, NODECHIP_REG);
  //WRITE_REG(0x6c007d0030 , 0x6c, NODECHIP_REG);
#ifdef DEBUG_REG
  {
    u32 tmp = 0;
    (void)tmp;
    tmp = (READ_CSR_CDMA(CDMA_CSR_INTER_DIE_RW_V2) &
          ~(0xff << CDMA_CSR_INTER_DIE_WRITE_ADDR_L4)) |
          (C2C_PCIE_CTRL << CDMA_CSR_INTER_DIE_WRITE_ADDR_H4) |
          (0b0000 << CDMA_CSR_INTER_DIE_WRITE_ADDR_L4);
    WRITE_CSR_CDMA(CDMA_CSR_INTER_DIE_RW_V2, tmp);
    tmp = (READ_CSR_CDMA(CDMA_CSR_INTRA_DIE_RW_V2) &
          ~(0xff << CDMA_CSR_INTRA_DIE_READ_ADDR_L4)) |
          (AXI_NOC << CDMA_CSR_INTRA_DIE_READ_ADDR_H4) |
          (0b0000 << CDMA_CSR_INTRA_DIE_READ_ADDR_L4);
    WRITE_CSR_CDMA(CDMA_CSR_INTRA_DIE_RW_V2, tmp);//
  }
#endif
  // tmp = (C2C_PCIE_CTRL << 28) |
  //   (CDMA_CMD_BASE_ADDR_PAIR >> 32);
  // WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_H32, tmp); //0x4
  // tmp = (CDMA_CMD_BASE_ADDR_PAIR & ((1ul << 32) - 1)) >> 16;
  // WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_M16, tmp);
  // tmp = READ_CSR_CDMA(CDMA_CSR_4) | (1 << CDMA_CSR_RCV_CMD_OS);
  // (CDMA_CSR_4, tmp);
  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[2] = {0}, high[2] = {0};
  u64 cmd_type = CDMA_SEND;
#if defined(SG_TV_GEN)
  int is_batch_send_recv_mode = read_env_variable_("TV_GEN_CDMA_BATCH_SEND_RECV_MODE");
  printf("cdma_send_cmodel_gen_cmd is_batch_send_recv_mode = %d\n", is_batch_send_recv_mode);
  if(1 == is_batch_send_recv_mode)
  low[0] = ((u64)stride_enable << 1) |           // stride_enable
        ((u64)nchw_copy << 2) |               // nchw_copy
        ((u64)cmd_type << 4) |                // cmd_type
        ((u64)is_first << 8) |                // cmd_type
        ((u64)is_last << 9) |                 // cmd_type
        ((u64)dtype << 12) |                  // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) | // src_start_addr_h13
        ((u64)psum_op << 29) |                // psum_op
        ((u64)src_n_stride << 32)             // src_nstride
      ;
  else
#endif
  low[0] = (1 << 0) |                            // intr_en
        ((u64)stride_enable << 1) |           // stride_enable
        ((u64)nchw_copy << 2) |               // nchw_copy
        ((u64)cmd_type << 4) |                // cmd_type
        ((u64)dtype << 12) |                  // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) | // src_start_addr_h13
        ((u64)psum_op << 29) |                // psum_op
        ((u64)src_n_stride << 32)             // src_nstride
      ;
  high[0] = (u64)src_c_stride |       // src_cstride
         ((u64)src_h_stride << 32) // src_nstride
      ;
  low[1] = (u64)src_n |         // src_nsize
        ((u64)src_c << 16) | // src_csize
        ((u64)src_h << 32)   // src_hsize
      ;
  high[1] = (u64)src_w |                    // src_wsize
         ((src_addr & 0xffffffff) << 32) // src_start_addr_l32
      ;
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}

void cdma_p2p_send_gen_cmd(int src_chipid, int dst_chipid, u32 cmd_length, u64 src_addr,
                           u64 dst_addr, int dtype, CMD_ID_NODE* pid_node
#if defined(SG_TV_GEN)
                       ,int is_first, int is_last
#endif
)
{
  FW_DBG(
      "%s: src_chipid = %d, dst_chipid = %d, src_addr = 0x%llx, "
      "dst_addr = 0x%llx, cmd_length = %u, dtype = %d\n",
      __func__, src_chipid, dst_chipid, src_addr, dst_addr, cmd_length, dtype);

  #ifdef USING_CMODEL
    int port = 0;
    int rank = cmodel_get_device_id();
    if (rank == src_chipid)
      port = lookup_send_port(dst_chipid);
    else
      port = lookup_recv_port(src_chipid);
    ASSERT(port != -1);
    // ASSERT_CDMA_TENSOR_LENGTH(cmd_length);
    if (dtype == CDMA_DTYPE_FP20) {
      ASSERT(src_addr % 128 == 0);
      ASSERT(dst_addr % 128 == 0);
      // ASSERT_FS_INFO(is_gmem(src_addr) && is_gmem(dst_addr),
      //             "can't be global memory, src_addr:0x%llx, dst_addr:0x%llx",
      //             src_addr, dst_addr);
      }
  #else
    int port = lookup_send_port(dst_chipid);
  #endif

  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[2] = {0}, high[2] = {0};
  u64 cmd_type = CDMA_GENERAL;
#if defined(SG_TV_GEN)
  int is_batch_send_recv_mode = read_env_variable_("TV_GEN_CDMA_BATCH_SEND_RECV_MODE");
  printf("cdma_p2p_send_gen_cmd is_batch_send_recv_mode = %d\n", is_batch_send_recv_mode);
  if(1 == is_batch_send_recv_mode)
  low[0] = ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)dtype << 12) |                   // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) |  // src_start_addr_h13
        (((dst_addr >> 32) & 0x1fff) << 32) |  // dst_start_addr_h13
        ((u64)src_chipid << 45) |              // src_chipid
        ((u64)dst_chipid << 49) |              // dst_chipid
        ((u64)is_first << 53) |                // cmd_type
        ((u64)is_last << 54)                   // cmd_type
      ;
  else
#endif
  low[0] = ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)dtype << 12) |                   // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) |  // src_start_addr_h13
        (((dst_addr >> 32) & 0x1fff) << 32) |  // dst_start_addr_h13
        ((u64)src_chipid << 45) |              // src_chipid
        ((u64)dst_chipid << 49)                // dst_chipid
      ;
  high[0] = ((u64)cmd_length) |              // cmd_length
         ((src_addr & 0xffffffff) << 32)  // src_start_addr_l32
      ;
  low[1] = (dst_addr & 0xffffffff);  // dst_start_addr_l32
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}

void cdma_read_gen_cmd(int remote_chipid, int chipid, u64 src_addr, u64 dst_addr,  u16 src_n,
                        u16 src_c,  u32 src_h,  u32 src_w, u32 src_n_stride, u32 src_c_stride,
                       u32 src_h_stride,  u16 dst_n,  u16 dst_c,  u32 dst_h,  u32 dst_w,
                       u32 dst_n_stride, u32 dst_c_stride, u32 dst_h_stride, int dtype,
                       int stride_enable, int nchw_copy, CMD_ID_NODE* pid_node)
{
  FW_DBG(
      "%s: remote_chipid = %d, chipid = %d, src_addr = 0x%llx, "
      "dst_addr = 0x%llx, src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
      "dst_n = %u, dst_c = %u, dst_h = %u, dst_w = %u, "
      "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
      "dst_n_stride = %u, dst_c_stride = %u, dst_h_stride = %u, "
      "dtype = %d, stride_enable = %d\n",
      __func__, remote_chipid, chipid, src_addr, dst_addr, src_n, src_c, src_h, src_w, dst_n,
      dst_c, dst_h, dst_w, src_n_stride, src_c_stride, src_h_stride, dst_n_stride, dst_c_stride,
      dst_h_stride, dtype, stride_enable);
  #ifdef USING_CMODEL
    int port = 0;
    int rank = cmodel_get_device_id();
    if (rank == remote_chipid)
      port = lookup_send_port(chipid);
    else
      port = lookup_recv_port(remote_chipid);
    ASSERT(port != -1);
    // ASSERT_CDMA_TENSOR_SIZE(src_n, src_c, src_h, src_w);
    // ASSERT_CDMA_TENSOR_SIZE(dst_n, dst_c, dst_h, dst_w);
    ASSERT_FS_INFO(dst_n * dst_c * dst_h * dst_w ==
                      src_n * src_c * src_h * src_w,
                  "dst_count=%d, src_count=%d", dst_n * dst_c * dst_h * dst_w,
                  src_n * src_c * src_h * src_w);
    if (dtype == CDMA_DTYPE_FP20) {
      ASSERT(src_addr % 128 == 0);
      ASSERT(dst_addr % 128 == 0);
      }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                    "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                    src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                    "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                    src_addr, dst_addr);
#if defined(SG_TV_GEN)
    char* early_debug_Evn = getenv("TV_GEN_CDMA_EARLY_DEBUG");
    int early_debug = 0;
    if (early_debug_Evn != NULL) {
      early_debug = atoi(early_debug_Evn);
      if (early_debug < 0 || early_debug > 1)
        early_debug = 0;
    }
    if (dtype != CDMA_DTYPE_FP20 && 1 == early_debug) {
      ASSERT_SYS_STRIDE_TV_GEN_EARLY_DEBUG(src_n_stride, src_c_stride, src_h_stride, src_n, src_c,
                                           src_h, src_w);
      ASSERT_SYS_STRIDE_TV_GEN_EARLY_DEBUG(dst_n_stride, dst_c_stride, dst_h_stride, dst_n, dst_c,
                                           dst_h, dst_w);
    }
#endif
  #else
    int port = lookup_recv_port(chipid);
  #endif

  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[4] = {0}, high[4] = {0};
  u64 cmd_type = CDMA_READ;
  low[0] = ((u64)stride_enable << 1) |            // stride_enable
        ((u64)nchw_copy << 2) |                // nchw_copy
        ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)dtype << 12) |                   // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) |  // src_start_addr_h13
        (((dst_addr >> 32) & 0x1fff) << 32) |  // dst_start_addr_h13
        ((u64)remote_chipid << 45) |           // remote_chipid
        ((u64)chipid << 49)                    // chipid
      ;
  high[0] = (u64)src_n_stride |        // src_nstride
         ((u64)src_c_stride << 32)  // src_cstride
      ;
  low[1] = (u64)src_h_stride |        // src_hstride
        ((u64)dst_n_stride << 32)  // dst_nstride
      ;
  high[1] = (u64)dst_c_stride |        // dst_cstride
         ((u64)dst_h_stride << 32)  // dst_hstride
      ;
  low[2] = (u64)src_n |          // src_nsize
        ((u64)src_c << 16) |  // src_csize
        ((u64)src_h << 32)    // src_hsize
      ;
  high[2] = (u64)src_w |          // src_wsize
         ((u64)dst_n << 32) |  // dst_nsize
         ((u64)dst_c << 48)    // dst_csize
      ;
  low[3] = (u64)dst_h |        // dst_wsize
        ((u64)dst_w << 32)  // dst_hsize
      ;

  high[3] = ((u64)src_addr & 0xffffffff) |        // src_start_addr_l32
         ((u64)(dst_addr & 0xffffffff) << 32)  // dst_start_addr_l32
      ;
  for (int i = 0; i < 4; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 4);
}

void cdma_write_gen_cmd(int src_chipid, int dst_chipid, u64 src_addr, u64 dst_addr, u16 src_n,
                        u16 src_c, u32 src_h, u32 src_w, u32 src_n_stride, u32 src_c_stride,
                        u32 src_h_stride, u16 dst_n, u16 dst_c, u32 dst_h, u32 dst_w,
                        u32 dst_n_stride, u32 dst_c_stride, u32 dst_h_stride, int is_fill_const,
                        int const_val, int dtype, int stride_enable, int nchw_copy, CMD_ID_NODE* pid_node)
{
  FW_DBG(
      "%s: src_chipid = %d, dst_chipid = %d, src_addr = 0x%llx, "
      "dst_addr = 0x%llx, src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
      "dst_n = %u, dst_c = %u, dst_h = %u, dst_w = %u, "
      "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
      "dst_n_stride = %u, dst_c_stride = %u, dst_h_stride = %u, "
      "%s const_val = %d, dtype = %d, stride_enable = %d\n",
      __func__, src_chipid, dst_chipid, src_addr, dst_addr, src_n, src_c, src_h, src_w, dst_n,
      dst_c, dst_h, dst_w, src_n_stride, src_c_stride, src_h_stride, dst_n_stride, dst_c_stride,
      dst_h_stride, is_fill_const == 1 ? "constant fill" : "memory copy", const_val, dtype, stride_enable);
  #ifdef USING_CMODEL
    int port = 0;
    int rank = cmodel_get_device_id();
    if (rank == src_chipid)
      port = lookup_send_port(dst_chipid);
    else
      port = lookup_recv_port(src_chipid);
    ASSERT(port != -1);
    // ASSERT_CDMA_TENSOR_SIZE(src_n, src_c, src_h, src_w);
    // ASSERT_CDMA_TENSOR_SIZE(dst_n, dst_c, dst_h, dst_w);
    ASSERT_FS_INFO(dst_n * dst_c * dst_h * dst_w ==
                      src_n * src_c * src_h * src_w,
                  "dst_count=%d, src_count=%d", dst_n * dst_c * dst_h * dst_w,
                  src_n * src_c * src_h * src_w);
    if (dtype == CDMA_DTYPE_FP20) {
      ASSERT(src_addr % 128 == 0);
      ASSERT(dst_addr % 128 == 0);
      }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                    "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                    src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
            "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
            src_addr, dst_addr);
#if defined(SG_TV_GEN)
    char* early_debug_Evn = getenv("TV_GEN_CDMA_EARLY_DEBUG");
    int early_debug = 0;
    if (early_debug_Evn != NULL) {
      early_debug = atoi(early_debug_Evn);
      if (early_debug < 0 || early_debug > 1)
        early_debug = 0;
    }
    if (dtype != CDMA_DTYPE_FP20 && 1 == early_debug) {
      ASSERT_SYS_STRIDE_TV_GEN_EARLY_DEBUG(src_n_stride, src_c_stride, src_h_stride, src_n, src_c,
                                           src_h, src_w);
      ASSERT_SYS_STRIDE_TV_GEN_EARLY_DEBUG(dst_n_stride, dst_c_stride, dst_h_stride, dst_n, dst_c,
                                           dst_h, dst_w);
    }
#endif
  #else
    int port = lookup_send_port(dst_chipid);
  #endif

  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[4] = {0}, high[4] = {0};
  u64 cmd_type = CDMA_WRITE;
  low[0] = ((u64)stride_enable << 1) |            // stride_enable
        ((u64)nchw_copy << 2) |                // nchw_copy
        ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)is_fill_const << 11) |           //
        ((u64)dtype << 12) |                   // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) |  // src_start_addr_h13
        (((dst_addr >> 32) & 0x1fff) << 32) |  // dst_start_addr_h13
        ((u64)src_chipid << 45) |              // src_chipid
        ((u64)dst_chipid << 49)                // dst_chipid
      ;
  if (1 == is_fill_const) {
    int constant = 0;
    constant = get_constant_value((void*)(&const_val), dtype);
    high[0] = (u64)constant |            // const_val instead of src_nstride if fill_constant_en
           ((u64)src_c_stride << 32)  // src_cstride
        ;
  } else {
    high[0] = (u64)src_n_stride |        // src_nstride
           ((u64)src_c_stride << 32)  // src_cstride
        ;
  }
  low[1] = (u64)src_h_stride |        // src_hstride
        ((u64)dst_n_stride << 32)  // dst_nstride
      ;

  high[1] = (u64)dst_c_stride |        // dst_cstride
         ((u64)dst_h_stride << 32)  // dst_hstride
      ;
  low[2] = (u64)src_n |          // src_nsize
        ((u64)src_c << 16) |  // src_csize
        ((u64)src_h << 32)    // src_hsize
      ;

  high[2] = (u64)src_w |          // src_wsize
         ((u64)dst_n << 32) |  // dst_nsize
         ((u64)dst_c << 48)    // dst_csize
      ;
  low[3] = (u64)dst_h |        // dst_wsize
        ((u64)dst_w << 32)  // dst_hsize
      ;

  high[3] = ((u64)src_addr & 0xffffffff) |        // src_start_addr_l32
         ((u64)(dst_addr & 0xffffffff) << 32)  // dst_start_addr_l32
      ;
  for (int i = 0; i < 4; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }
  END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 4);
}

void cdma_lossy_compress_gen_cmd(int dst_chipid, u64 src_addr, u16 src_n, u16 src_c, u32 src_h, u32 src_w,
                       u32 src_n_stride, u32 src_c_stride, u32 src_h_stride, int dtype,
                       int stride_enable, int reduce_op, int nchw_copy, CMD_ID_NODE* pid_node) {
  FW_DBG(
      "%s: dst_chipid = %d, src_addr = 0x%llx, "
      "src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
      "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
      "dtype = %d, stride_enable = %d, nchw_copy = %d\n",
      __func__, dst_chipid, src_addr, src_n, src_c, src_h, src_w,
      src_n_stride, src_c_stride, src_h_stride, dtype, stride_enable, nchw_copy);
  #ifdef USING_CMODEL
    // ASSERT_CDMA_TENSOR_SIZE(src_n, src_c, src_h, src_w);
    ASSERT(dtype == CDMA_DTYPE_FP32);
    ASSERT_FS_INFO(!is_smem(src_addr),
                  "can't be static memory, src_addr:0x%llx", src_addr);
  #endif

  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  int psum_op = (reduce_op == CDMA_OPCODE_NONE ? PSUM_OP_WO : PSUM_OP_WR);
  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[2] = {0}, high[2] = {0};
  u64 cmd_type = CDMA_LOSSY_COMPRESS;
  low[0] = (1 << 0) |                            // intr_en
        ((u64)stride_enable << 1) | // stride_enable
        ((u64)nchw_copy << 2) | // nchw_copy
        ((u64) cmd_type << 4) | // cmd_type
        ((u64)dtype << 12) | // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) | // src_start_addr_h13
        ((u64)psum_op << 29) | // psum_op
        ((u64)src_n_stride << 32) // src_nstride
        ;
  high[0] = (u64)src_c_stride | // src_cstride
          ((u64)src_h_stride << 32) // src_nstride
        ;
  low[1] = (u64)src_n | // src_nsize
      ((u64)src_c << 16) | // src_csize
      ((u64)src_h << 32) // src_hsize
      ;
  high[1] = (u64)src_w |  // src_wsize
          ((src_addr & 0xffffffff) << 32) // src_start_addr_l32
          ;
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}

void cdma_lossy_decompress_gen_cmd(int dst_chipid, u64 src_addr, u16 src_n, u16 src_c, u32 src_h,
                                   u32 src_w, u32 src_n_stride, u32 src_c_stride, u32 src_h_stride,
                                   int dtype, int reduce_op, int stride_enable, int nchw_copy,
                                   CMD_ID_NODE* pid_node)
{
  FW_DBG(
      "%s: dst_chipid = %d, src_addr = 0x%llx, "
      "src_n = %u, src_c = %u, src_h = %u, src_w = %u, "
      "src_n_stride = %u, src_c_stride = %u, src_h_stride = %u, "
      "dtype = %d, reduce_op = %d, nchw_copy = %d\n",
      __func__, dst_chipid, src_addr, src_n, src_c, src_h, src_w, src_n_stride, src_c_stride,
      src_h_stride, dtype, reduce_op, nchw_copy);
#ifdef USING_CMODEL
  // ASSERT_CDMA_TENSOR_SIZE(src_n, src_c, src_h, src_w);
  ASSERT(dtype == CDMA_DTYPE_FP20);
  if (dtype == CDMA_DTYPE_FP20) {
    ASSERT(src_addr % 128 == 0);
  }
  ASSERT_FS_INFO(!is_smem(src_addr) && !is_lmem(src_addr),
                 "can't be static memory and local memory, src_addr:0x%llx", src_addr);
#endif
  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  int psum_op = (reduce_op == CDMA_OPCODE_NONE ? PSUM_OP_WO : PSUM_OP_WR);
  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low[2] = {0}, high[2] = {0};
  u64 cmd_type = CDMA_LOSSY_DECOMPRESS;
  low[0] = (1 << 0) |                            // intr_en
        ((u64)stride_enable << 1) |            // stride_enable
        ((u64)nchw_copy << 2) | // nchw_copy
        ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)dtype << 12) |                   // src_data_format
        (((src_addr >> 32) & 0x1fff) << 16) |  // src_start_addr_h13
        ((u64)psum_op << 29) |                 // psum_op
        ((u64)src_n_stride << 32)              // src_nstride
        ;
  high[0] = (u64)src_c_stride | // src_cstride
         ((u64)src_h_stride << 32) // src_nstride
         ;
  low[1] = (u64)src_n |          // src_nsize
        ((u64)src_c << 16) |  // src_csize
        ((u64)src_h << 32)    // src_hsize
      ;
  high[1] = (u64)src_w |                     // src_wsize
         ((src_addr & 0xffffffff) << 32)  // src_start_addr_l32
      ;
  for (int i = 0; i < 2; ++i) {
    WRITE_CMD_CDMA(port, i, high[i], low[i]);
  }

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}

void cdma_recv_gen_cmd(int src_chipid, int dst_chipid, u64 dst_addr, u16 dst_n, u16 dst_c,
                       u32 dst_h, u32 dst_w, u32 dst_n_stride, u32 dst_c_stride, u32 dst_h_stride,
                       int opcode, int dtype, int stride_enable, CMD_ID_NODE* pid_node
#if defined(SG_TV_GEN)
                       ,int is_first, int is_last, int send_tsk_id
#endif
) {
    FW_DBG(
        "%s: src_chipid = %d, dst_chipid = %d, dst_addr = 0x%llx, "
        "dst_n = %u, dst_c = %u, dst_h = %u, dst_w = %u, "
        "dst_n_stride = %u, dst_c_stride = %u, dst_h_stride = %u, "
        "opcode = %d, dtype = %d, stride_enable = %d\n",
        __func__, src_chipid, dst_chipid, dst_addr, dst_n, dst_c, dst_h, dst_w,
        dst_n_stride, dst_c_stride, dst_h_stride, opcode, dtype, stride_enable);
#if defined(SG_TV_GEN)
  // set_first_last_flag_for_cdma_tv_gen_batch_send_recv_test(is_first, is_last);
#endif
  #ifdef USING_CMODEL
    // ASSERT_CDMA_TENSOR_SIZE(dst_n, dst_c, dst_h, dst_w);
    if (dtype == CDMA_DTYPE_FP20) {
        ASSERT(dst_addr % 128 == 0);
        ASSERT_FS_INFO(!is_lmem(dst_addr),
                   "can't be local memory, dst_addr:0x%llx", dst_addr);
    }
    // if (opcode != CDMA_OPCODE_NONE){
    //   ASSERT_FS_INFO(is_l2mem(dst_addr),
    //             "must be  l2memory,dst_addr:0x%llx", dst_addr);
    // }
    if (opcode == CDMA_OPCODE_ADD || opcode == CDMA_OPCODE_MAX || opcode == CDMA_OPCODE_MIN)
      ASSERT((dtype == CDMA_DTYPE_FP20) || (dtype == CDMA_DTYPE_FP32) || (dtype == CDMA_DTYPE_FP16) || (dtype == CDMA_DTYPE_BF16) || (dtype == CDMA_DTYPE_INT32));
    if (opcode == CDMA_OPCODE_MUL)
      ASSERT((dtype == CDMA_DTYPE_FP20) || (dtype == CDMA_DTYPE_FP32) || (dtype == CDMA_DTYPE_FP16) || (dtype == CDMA_DTYPE_BF16) );
    ASSERT_FS_INFO(!is_smem(dst_addr),
                   "can't be static memory,dst_addr:0x%llx", dst_addr);
  #endif
    int port = lookup_recv_port(src_chipid);
    ASSERT(port != -1);
    // int src_chipid_h2 = ((src_chipid & 0b1100) >> 2);
    // int src_chipid_l2 = (src_chipid & 0b11);

    #if defined(SG_TV_GEN)
    // TCREDIT invalid for cmodel_write_reg, only used dor sg_wr_tv_dump_reg_pointer
    //TCREDIT_NSTRIDE {0, 32}
    WRITE_REG(CDMA_TCREDICT(port), (u32)dst_n_stride, NODECHIP_REG);
    // TCREDIT_NSTRIDE {32, 32}
    WRITE_REG((CDMA_TCREDICT(port) + 32 / 8), (u32)dst_c_stride, NODECHIP_REG);
    // TCREDIT_HSTRIDE {64, 32}
    WRITE_REG((CDMA_TCREDICT(port) + 64 / 8), (u32)dst_h_stride, NODECHIP_REG);
    // TCREDIT_NSIZE {96, 16} TCREDIT_CSIZE {112, 16}
    WRITE_REG((CDMA_TCREDICT(port) + 96 / 8), (u32)(dst_n | (dst_c << 16)), NODECHIP_REG);
    // TCREDIT_HSIZE {128, 32}
    WRITE_REG((CDMA_TCREDICT(port) + 128 / 8), (u32)dst_h, NODECHIP_REG);
    // TCREDIT_WSIZE {160, 32}
    WRITE_REG((CDMA_TCREDICT(port) + 160 / 8), (u32)dst_w, NODECHIP_REG);
    // TCREDIT_START_ADDR_L32 {192, 32}
    WRITE_REG((CDMA_TCREDICT(port) + 192 / 8), (u32)(dst_addr & 0xffffffff), NODECHIP_REG);
    // TCREDIT_START_ADDR_H13 {224, 13} TCREDIT_REDUCE_OP {237, 3}
    WRITE_REG((CDMA_TCREDICT(port) + 224 / 8), (u32)(((dst_addr >> 32) & 0x1fff) | (opcode << 13)), NODECHIP_REG);
    #endif
    // WRITE_TOP_CDMA(0x24, 0x0);
    // WRITE_TOP_CDMA(0x28, 0x6c);
    // WRITE_TOP_CDMA(0x2c, 0x01ffffff);
    // WRITE_TOP_CDMA(0x30, 0x6c);
#ifdef DEBUG_REG
    {
      u32 tmp = 0;
      (void)tmp;
      tmp = (C2C_PCIE_CTRL << 28) |
          (CDMA_CMD_BASE_ADDR_PAIR >> 32);
      WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_H32, tmp); //0x4
      tmp = (CDMA_CMD_BASE_ADDR_PAIR & ((1ul << 32) - 1)) >> 16;
      WRITE_CSR_CDMA(CDMA_CSR_RCV_ADDR_M16, tmp);
      tmp = READ_CSR_CDMA(CDMA_CSR_4) | (1 << CDMA_CSR_RCV_CMD_OS);
      WRITE_CSR_CDMA(CDMA_CSR_4, tmp);
    }
#endif
    BEGIN_FAST_GEN_CMD_CDMA(port);
    u64 low[2] = {0}, high[2] = {0};
    u64 cmd_type = CDMA_RECV;
#if defined(SG_TV_GEN)
  int is_batch_send_recv_mode = read_env_variable_("TV_GEN_CDMA_BATCH_SEND_RECV_MODE");
  printf("cdma_recv_gen_cmd is_batch_send_recv_mode = %d\n", is_batch_send_recv_mode);
  int send_tsk_index = -1;
  if (send_tsk_id == CDMA_SEND)
    send_tsk_index = 0;
  else if (send_tsk_id == CDMA_LOSSY_COMPRESS)
    send_tsk_index = 1;
  else if (send_tsk_id == CDMA_LOSSY_DECOMPRESS)
    send_tsk_index = 2;
  if(1 == is_batch_send_recv_mode)
    low[0] = ((u64)stride_enable << 1) | // stride_enable
          // ((u64)src_chipid_h2 << 2) | // src chipid 3rd\4th bit
          (cmd_type << 4) | // cmd_type
          (is_first << 8) | // cmd_type
          (is_last << 9) | // cmd_type
          ((u64)dtype << 11) | // recv data format
          // ((u64)src_chipid_l2 << 14) | // src chipid 1st\2ed bit
          (((dst_addr >> 32) & 0x1fff) << 16) | // dst_start_addr_h13
          ((u64)opcode << 29) | // reduce_op
          ((u64)dst_n_stride << 32) // dst_nstride
          ;
  else if(-1 != send_tsk_index)
    low[0] = ((u64)stride_enable << 1) | // stride_enable
          // ((u64)src_chipid_h2 << 2) | // src chipid 3rd\4th bit
          (cmd_type << 4) | // cmd_type
          ((u64)dtype << 11) | // recv data format
          ((u64)send_tsk_index << 14) | // send_tsk_type for model tool
          // ((u64)src_chipid_l2 << 14) | // src chipid 1st\2ed bit
          (((dst_addr >> 32) & 0x1fff) << 16) | // dst_start_addr_h13
          ((u64)opcode << 29) | // reduce_op
          ((u64)dst_n_stride << 32) // dst_nstride
          ;
  else
#endif
    low[0] = ((u64)stride_enable << 1) | // stride_enable
          // ((u64)src_chipid_h2 << 2) | // src chipid 3rd\4th bit
          (cmd_type << 4) | // cmd_type
          ((u64)dtype << 11) | // recv data format
          // ((u64)src_chipid_l2 << 14) | // src chipid 1st\2ed bit
          (((dst_addr >> 32) & 0x1fff) << 16) | // dst_start_addr_h13
          ((u64)opcode << 29) | // reduce_op
          ((u64)dst_n_stride << 32) // dst_nstride
          ;
    high[0] = (u64)dst_c_stride | // dst_cstride
           ((u64)dst_h_stride << 32) // dst_nstride
          ;
    low[1] = (u64)dst_n | // dst_nsize
        ((u64)dst_c << 16) | // dst_csize
        ((u64)dst_h << 32) // dst_hsize
        ;
    high[1] = (u64)dst_w |  // src_wsize
           ((dst_addr & 0xffffffff) << 32) // dst_start_addr_l32
           ;
    for (int i = 0; i < 2; ++i) {
      WRITE_CMD_CDMA(port, i, high[i], low[i]);
    }

    END_FAST_GEN_CMD_CDMA(port, pid_node);
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(port)), 1, NODECHIP_REG);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, port << 7, pid_node, high, low, 2);
}

void cdma_tcp_send_gen_cmd(int src_chipid, int dst_chipid, u16 buffer_length, u16 frame_length, u64 src_addr,
                           int first_desc, int last_desc, CMD_ID_NODE* pid_node)
{
  FW_DBG(
      "%s: dev<%d> send to dev<%d>, src_addr = 0x%llx, frame_length = %hu, "
      "buffer_length = %hu, first_desc = %d, last_desc = %d\n",
      __func__, src_chipid, dst_chipid, src_addr, frame_length, buffer_length, first_desc, last_desc);

  #ifdef USING_CMODEL
    // ASSERT_CDMA_TENSOR_LENGTH(buffer_length);
    ASSERT_FS_INFO(!is_smem(src_addr),
                   "can't be static memory, src_addr:0x%llx", src_addr);
  #endif
  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  int intr_en = 0;
  if(1 == last_desc)
    intr_en = 1;
  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low = 0, high = 0;
  u64 cmd_type = CDMA_TCP_SEND;
  low = (u64)intr_en |                         // intr_en
        ((u64)first_desc << 2) |               // FD
        ((u64)last_desc << 3) |                // LD
        ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)buffer_length << 8) |            // buffer_length
        ((u64)frame_length << 32) |             // frame_length
        ((u64)src_chipid << 48) |              // src_chipid
        ((u64)dst_chipid << 52)                // dst_chipid
      ;
  high = (src_addr & 0xffffffff) |             // buffer_addr_l32
         (((src_addr >> 32) & 0x1fff) << 32)   // buffer_addr_h13
      ;
  WRITE_CMD_CDMA(port, 0, high, low);

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, port << 7, pid_node, &high, &low, 1);  // TODO
}

void cdma_tcp_recv_gen_cmd(int src_chipid, int dst_chipid, u16 buffer_length, u16 send_frame_length,
                           u64 dst_addr, int first_desc, int last_desc, CMD_ID_NODE* pid_node)
{
  FW_DBG(
      "%s: dev<%d> recv from dev<%d>, dst_addr = 0x%llx, "
      "buffer_length = %hd\n",
      __func__, dst_chipid, src_chipid, dst_addr, buffer_length);

  #ifdef USING_CMODEL
    // ASSERT_CDMA_TENSOR_LENGTH(buffer_length);
    ASSERT_FS_INFO(!is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx", dst_addr);
  #endif
  int port = lookup_recv_port(src_chipid);
  ASSERT(port != -1);
  int intr_en = 0;
  if(1 == last_desc)
    intr_en = 1;
  BEGIN_FAST_GEN_CMD_CDMA(port);
  u64 low = 0, high = 0;
  u64 cmd_type = CDMA_TCP_RECV;
  low = (u64)intr_en |                         // intr_en
        ((u64)first_desc << 2) |               // FD, used only for tvgen
        ((u64)last_desc << 3) |                // LD, used only for tvgen
        ((u64)cmd_type << 4) |                 // cmd_type
        ((u64)buffer_length << 8) |            // buffer_length
        ((u64)send_frame_length << 32) |       // send_frame_length, only used for cmodel
        ((u64)src_chipid << 48) |              // src_chipid
        ((u64)dst_chipid << 52)                // dst_chipid
      ;
  high = (dst_addr & 0xffffffff) |             // buffer_addr_l32
         (((dst_addr >> 32) & 0x1fff) << 32)   // buffer_addr_h13
      ;
  WRITE_CMD_CDMA(port, 0, high, low);

  END_FAST_GEN_CMD_CDMA(port, pid_node);
  profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, port << 7, pid_node, &high, &low, 1);
}

void atomic_cdma_route_configuration(int dst_chipid, CDMA_ROUTE_TYPE route_type,
                                     int port) {
  if (port == -1) {
    port = lookup_send_port(dst_chipid);
  }
  ASSERT(port != -1);
  FW_DBG("%s: dst_chipid=%d, port=%d, CDMA_ROUTE_TYPE=%d\n", __func__,
         dst_chipid, port, route_type);
  // D2D: AXI_RN
  if (CDMA_ROUTE_AXI_RN == route_type) {
    // reg_inter_die_write_addr : write to AXI_RN
    u32 value = READ_REG(CDMA_CSR_INTER_DIE_RW(port));
    value = (value & 0xffff00ff) | (0x80 << 8);
    WRITE_REG(CDMA_CSR_INTER_DIE_RW(port), value, NODECHIP_REG);
    // reg_intra_die_read_addr : read from AXI_RN
    value = READ_REG(CDMA_CSR_INTRA_DIE_RW(port));
    value = (value & 0xffffff00) | (0x80);
    WRITE_REG(CDMA_CSR_INTRA_DIE_RW(port), value, NODECHIP_REG);
  }
}

void atomic_cdma_pio_reg_descriptor_update(int dst_chipid) {
  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  FW_DBG("%s: dst_chipid=%d, port=%d\n", __func__, dst_chipid, port);
  WRITE_REG(CDMA_DESCRIPTOR_UPDATE(port), 1, NODECHIP_REG);
}

void atomic_cdma_config_tcredit_for_pld_test(int dst_chipid, u64 dst_addr,
                                             int dst_n, int dst_c, int dst_h,
                                             int dst_w, int dst_n_stride,
                                             int dst_c_stride, int dst_h_stride,
                                             int opcode, int port) {
  if (port == -1) {
    port = lookup_send_port(dst_chipid);
  }
  ASSERT(port != -1);
  FW_DBG("%s: dst_chipid=%d, port=%d, dst_addr=0x%llx, shape=[%d,%d,%d,%d], "
         "stride=[%d,%d,%d,1], opcode=%d\n",
         __func__, dst_chipid, port, dst_addr, dst_n, dst_c, dst_h, dst_w,
         dst_n_stride, dst_c_stride, dst_h_stride, opcode);
  // TCREDIT_NSTRIDE {0, 32}
  WRITE_REG(CDMA_TCREDICT(port), (u32)dst_n_stride, NODECHIP_REG);
  // TCREDIT_NSTRIDE {32, 32}
  WRITE_REG((CDMA_TCREDICT(port) + 32 / 8), (u32)dst_c_stride, NODECHIP_REG);
  // TCREDIT_HSTRIDE {64, 32}
  WRITE_REG((CDMA_TCREDICT(port) + 64 / 8), (u32)dst_h_stride, NODECHIP_REG);
  // TCREDIT_NSIZE {96, 16} TCREDIT_CSIZE {112, 16}
  WRITE_REG((CDMA_TCREDICT(port) + 96 / 8), (u32)(dst_n | (dst_c << 16)),
            NODECHIP_REG);
  // TCREDIT_HSIZE {128, 32}
  WRITE_REG((CDMA_TCREDICT(port) + 128 / 8), (u32)dst_h, NODECHIP_REG);
  // TCREDIT_WSIZE {160, 32}
  WRITE_REG((CDMA_TCREDICT(port) + 160 / 8), (u32)dst_w, NODECHIP_REG);
  // TCREDIT_START_ADDR_L32 {192, 32}
  WRITE_REG((CDMA_TCREDICT(port) + 192 / 8), (u32)(dst_addr & 0xffffffff),
            NODECHIP_REG);
  // TCREDIT_START_ADDR_H13 {224, 13} TCREDIT_REDUCE_OP {237, 3}
  WRITE_REG((CDMA_TCREDICT(port) + 224 / 8),
            (u32)(((dst_addr >> 32) & 0x1fff) | (opcode << 13)), NODECHIP_REG);
}

void atomic_cdma_pio_interrupt_poll(int dst_chipid) {
#ifdef CDMA_DES_PLD_TEST
  return;
#endif
  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  FW_DBG("%s: dst_chipid=%d, port=%d\n", __func__, dst_chipid, port);
#ifndef USING_CMODEL
  int timeout = 10000;
  while ((READ_REG(CDMA_CSR_CMD_DONE_STATUS(port)) & 0x1) != 0x1) {
#if !defined(DISABLE_CDMA) && !defined(USING_CMODEL)
    //udelay(1);
#endif
    if (--timeout == 0) {
      FW_DBG("%s: cdma polling wait timeout\n", __func__);
      reset_cdma(dst_chipid);
      return;
    }
  }
  // write 1 to clear
  WRITE_REG(CDMA_CSR_CMD_DONE_STATUS(port),
            (READ_REG(CDMA_CSR_CMD_DONE_STATUS(port)) | (1 << 0)),
            NODECHIP_REG);
#endif
}

void reset_cdma(int dst_chipid) {
  int port = lookup_send_port(dst_chipid);
  ASSERT(port != -1);
  CORE_PRINT("%s: dst_chipid=%d, port=%d\n", __func__, dst_chipid, port);
  u32 csr65_value = READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x10c);
  csr65_value |= 0x1;
  WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x10c, csr65_value, NODECHIP_REG);
  int timeout = 10000;
  while (true) {
    csr65_value = READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x10c);
    u32 reg_cfg_stopper_done = ((csr65_value << 6) & 0x1f);
    if (reg_cfg_stopper_done == 0x1f)
      break;
#if !defined(DISABLE_CDMA) && !defined(USING_CMODEL)
    //udelay(1);
#endif
    if (--timeout == 0) {
      CORE_PRINT("%s: reset_cdma wait timeout reg_cfg_stopper_done = %x\n", __func__, reg_cfg_stopper_done);
      return;
    }
  }
  u32 top_soft_reset_ctrl = READ_REG(C2C_TOP_BASE_ADDR(0) + 0X90);
  top_soft_reset_ctrl &= 0xfffffffe;
  CORE_PRINT("%s: reset_cdma sucessfully, write %x to %llx\n", __func__, top_soft_reset_ctrl, (u64)(C2C_TOP_BASE_ADDR(0) + 0X90));
  WRITE_REG(C2C_TOP_BASE_ADDR(0) + 0X90, top_soft_reset_ctrl, NODECHIP_REG);
}

#ifdef USING_CMODEL
void cdma_fake_all_reduce_gen_cmd(int dst_chipid, u64 src_addr, int src_n, int src_c, int src_h, int src_w,
                       int src_n_stride, int src_c_stride, int src_h_stride, int opcode, int dtype,
                       CMD_ID_NODE* pid_node) {
    FW_DBG(
        "%s: dst_chipid = %d, src_addr = 0x%llx, "
        "src_n = %d, src_c = %d, src_h = %d, src_w = %d, "
        "src_n_stride = %d, src_c_stride = %d, src_h_stride = %d, "
        "opcode = %d, dtype = %d\n",
        __func__, dst_chipid, src_addr, src_n, src_c, src_h, src_w,
        src_n_stride, src_c_stride, src_h_stride, opcode, dtype);

    int port = dst_chipid; // tpu core -> cdma port
    ASSERT(port != -1);
    int dst_chipid_h2 = ((dst_chipid & 0b1100) >> 2);
    int dst_chipid_m1 = ((dst_chipid & 0b10) >> 1);
    int dst_chipid_l1 = (dst_chipid & 0b1);
    BEGIN_FAST_GEN_CMD_CDMA(port);
    u64 low[2] = {0}, high[2] = {0};
    u64 cmd_type = CDMA_FAKE_ALL_REDUCE;
    low[0] = ((u64)dst_chipid_l1 << 3) | // dst chip id 1st bit
          ((u64) cmd_type << 4) | // cmd_type
          ((u64)dtype << 12) | // src_data_format
          ((u64)dst_chipid_m1 << 15) | // dst chip id 2ed bit
          (((src_addr >> 32) & 0x1fff) << 16) | // src_start_addr_h13
          ((u64) dst_chipid_h2 << 30) | // dst chip id 3rd\4th bit
          ((u64)src_n_stride << 32) // src_nstride
          ;
    high[0] = (u64)src_c_stride | // src_cstride
           ((u64)src_h_stride << 32) // src_nstride
          ;
    low[1] = (u64)src_n | // src_nsize
        ((u64)src_c << 16) | // src_csize
        ((u64)src_h << 32) // src_hsize
        ;
    high[1] = (u64)src_w |  // src_wsize
           ((src_addr & 0xffffffff) << 32) // src_start_addr_l32
           ;
    for (int i = 0; i < 2; ++i) {
      WRITE_CMD_CDMA(port, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_CDMA(port, pid_node);
    profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}

void cdma_fake_p2p_gen_cmd(int dst_chipid, u64 src_addr, int src_n, int src_c, int src_h, int src_w,
                       int src_n_stride, int src_c_stride, int src_h_stride, int opcode, int dtype,
                       CMD_ID_NODE* pid_node) {

    FW_DBG(
        "%s: dst_chipid = %d, src_addr = 0x%llx, "
        "src_n = %d, src_c = %d, src_h = %d, src_w = %d, "
        "src_n_stride = %d, src_c_stride = %d, src_h_stride = %d, "
        "opcode = %d, dtype = %d\n",
        __func__, dst_chipid, src_addr, src_n, src_c, src_h, src_w,
        src_n_stride, src_c_stride, src_h_stride, opcode, dtype);

    int port = 8; // cdma port for p2p
    ASSERT(port != -1);
    int dst_chipid_h2 = ((dst_chipid & 0b1100) >> 2);
    int dst_chipid_m1 = ((dst_chipid & 0b10) >> 1);
    int dst_chipid_l1 = (dst_chipid & 0b1);
    BEGIN_FAST_GEN_CMD_CDMA(port);
    u64 low[2] = {0}, high[2] = {0};
    u64 cmd_type = CDMA_FAKE_P2P;
    low[0] = ((u64)dst_chipid_l1 << 3) | // dst chip id 1st bit
          ((u64) cmd_type << 4) | // cmd_type
          ((u64)dtype << 12) | // src_data_format
          ((u64)dst_chipid_m1 << 15) | // dst chip id 2ed bit
          (((src_addr >> 32) & 0x1fff) << 16) | // src_start_addr_h13
          ((u64) dst_chipid_h2 << 30) | // dst chip id 3rd\4th bit
          ((u64)src_n_stride << 32) // src_nstride
          ;
    high[0] = (u64)src_c_stride | // src_cstride
           ((u64)src_h_stride << 32) // src_nstride
          ;
    low[1] = (u64)src_n | // src_nsize
        ((u64)src_c << 16) | // src_csize
        ((u64)src_h << 32) // src_hsize
        ;
    high[1] = (u64)src_w |  // src_wsize
           ((src_addr & 0xffffffff) << 32) // src_start_addr_l32
           ;
    for (int i = 0; i < 2; ++i) {
      WRITE_CMD_CDMA(port, i, high[i], low[i]);
    }
    END_FAST_GEN_CMD_CDMA(port, pid_node);
    profile_time_set_node(ENGINE_CDMA, cmd_type,
      0, dtype | port << 7, pid_node, high, low, 2);
}
#endif
