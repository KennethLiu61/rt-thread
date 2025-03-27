#include "nodechip_pld_test.h"
#include "tpu_kernel.h"
#include "tpu_defs.h"

#define N 1
#define C 1024
#define H 1
#define W 1024

#define H_STRIDE (W)
#define C_STRIDE (H_STRIDE * H)
#define N_STRIDE (C_STRIDE * C)

//                         chip 0                                                                    chip 1
// ---------------------------------------------------------------------------------------       ---------------
// gdma                       sdma                        cdma(port 0)                           cdma(port 0)
//                            dma_tensor
// sys_send msg_id:1, wcnt:3  sys_send msg_id:1, wcnt:3   sys_msg_rx_send msg_id:1, wcnt:3
// sys_wait msg_id:1, scnt:3  sys_wait msg_id:1, scnt:3   sys_msg_rx_wait msg_id:1, scnt:3
//                                                                                               DMA_send + PSUM
// DMA_tensor + ARE                                       DMA_receive_tensor + ARE
//                                                                                                               
// sys_send msg_id:3, wcnt:3  sys_send msg_id:3, wcnt:3   sys_msg_rx_send msg_id:3, wcnt:3
// sys_wait msg_id:3, scnt:3  sys_wait msg_id:3, scnt:3   sys_msg_rx_wait msg_id:3, scnt:3
//                            dma_tensor

void nodechip_cdma_are_snd_test(
    unsigned long long input_addr, unsigned long long output_addr)
{
  int dst_chipid = 0;
  int src_chipid = 1;
  int reduce_opcode = ALL_REDUCE_ADD;
  data_type_t dtype = DT_FP32;

  tpu_initialize();
  tpu_cdma_send(dst_chipid,
                src_chipid,
                input_addr,
                N, H, H, W,
                N_STRIDE, C_STRIDE, H_STRIDE,
                reduce_opcode,
                dtype);
  tpu_poll();
}

static void dma_barrier() {
  int cdma_port = 0;
  int msg_cnt = 3; // GDMA, SDMA and CDMA receive
  int msg_id = tpu_get_local_msg_id();

  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);

  // cdma sys_msg_rx_send, sys_msg_rx_wait
  tpu_cdma_send_msg(cdma_port, msg_id, msg_cnt);
  tpu_cdma_wait_msg(cdma_port, msg_id, msg_cnt);
}

void nodechip_cdma_are_rcv_test(
    unsigned long long input_addr, unsigned long long output_addr)
{
  int dst_chipid = 1;
  int src_chipid = 0;
  int reduce_psum = ALL_REDUCE_PSUM_WR;
  int reduce_opcode = ALL_REDUCE_ADD;
  data_type_t dtype = DT_FP32;

  local_addr_t l_addr = 0;
  system_addr_t g_l2_addr = tpu_l2_sram_get_start_addr();

  tpu_initialize();

  // Clear L2M
  {
    scalar_t scaler_val = {.u32 = 0};
    dim4 shape = {N, C, H, W};
    tpu_sdma_set_C_system(
        g_l2_addr,
        scaler_val,
        &shape,
        NULL,
        DT_UINT32);
  }

  dim4 shape = {N, C, H, W};
  tpu_gdma_cpy_S2L(l_addr,
                   input_addr,
                   &shape,
                   NULL,
                   NULL,
                   dtype);

  dma_barrier();

  tpu_cdma_recv(src_chipid,
                dst_chipid,
                g_l2_addr,
                N, H, H, W,
                N_STRIDE, C_STRIDE, H_STRIDE,
                g_l2_addr,
                reduce_opcode,
                dtype);
  tpu_gdma_cpy_reduce_L12L2(g_l2_addr,
                            l_addr,
                            &shape,
                            NULL,
                            NULL,
                            dtype,
                            reduce_psum,
                            reduce_opcode);

  dma_barrier();

  tpu_sdma_system_cpy(output_addr,
                      g_l2_addr,
                      N * N_STRIDE,
                      dtype);

  tpu_poll();
}
