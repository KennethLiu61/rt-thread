#include "firmware_common_inline.h"
#include "firmware_common_macro.h"
#include "nodechip_pld_test.h"
#include "tpu_defs.h"
#include "tpu_kernel.h"

#define N 1
#define C 1
#define H 1
#define W 1024

#define H_STRIDE (W)
#define C_STRIDE (H_STRIDE * H)
#define N_STRIDE (C_STRIDE * C)

void nodechip_cdma_write_test(unsigned long long input_addr,
                              unsigned long long output_addr) {
  int dst_chipid = 0;
  int src_chipid = 0;
  data_type_t dtype = DT_INT8;

  tpu_initialize();
  // WARNING: reg config need port info but we get port_id in gen_cmd function
  // walkaround: we use only cdma port 0 here, need fixed later

  // A. switch config for k2k
  // write32(0x6c00791000,0) set pio mode
  u32 write_reg = READ_REG(CDMA_ENGINE_MAIN_CTRL(0));
  write_reg = write_reg & 0xfffffffe;
  WRITE_REG((CDMA_ENGINE_MAIN_CTRL(0)), write_reg, NODECHIP_REG);
  // write32(0x6c0079100c,0x8000)
  write_reg = READ_REG(CDMA_CSR_INTER_DIE_RW(0));
  write_reg = write_reg & 0xffff00ff;
  write_reg = write_reg | 0x8000;
  WRITE_REG((CDMA_CSR_INTER_DIE_RW(0)), write_reg, NODECHIP_REG);
  // write32(0x6c0079123c,0x80)
  write_reg = READ_REG(CDMA_CSR_INTRA_DIE_RW(0));
  write_reg = write_reg & 0xffffff00;
  write_reg = write_reg | 0x80;
  WRITE_REG((CDMA_CSR_INTRA_DIE_RW(0)), write_reg, NODECHIP_REG);

  // B. des mode config(jump this step since it's pio mode)
  // C. config cdma instruction
  tpu_cdma_write(src_chipid, dst_chipid, input_addr, output_addr, N, C, H, W,
                 N_STRIDE, C_STRIDE, H_STRIDE, N, C, H, W, N_STRIDE, C_STRIDE,
                 H_STRIDE, 0 /*is_fill_const*/, 0 /*const_val*/, dtype,
                 0 /*stride_enable*/, 0 /*nchw_copy*/
  );

  // D. sys end config for des mode(jump this step since it's pio mode)
  // E. pio run cmd config
  // write32(0x6c00791000,0x1)
  WRITE_REG((CDMA_DESCRIPTOR_UPDATE(0)), 1, NODECHIP_REG);
  tpu_poll();
}