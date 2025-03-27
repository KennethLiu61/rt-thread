#ifdef USING_CMODEL
#include "cmodel_multi_thread.h"
#endif

#include "firmware_common.h"
#include "firmware_runtime.h"
#include "atomic_gdma_gen_cmd.h"
#include "firmware_timer.h"
#include "gdma_reg_value.h"
#include "firmware_common_inline.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include "sg_api_struct.h"

#ifdef USING_CMODEL
#include "cmodel_common.h"
#else
#include "firmware_top.h"
#endif
#include "atomic_sys_gen_cmd.h"
#ifdef __cplusplus
extern "C"
{
#endif

  void nodechip_multi_msg_sync_multi_core(
      u64 bdc_cmd_offset,
      u64 gdma_cmd_offset,
      u64 hau_cmd_offset,
      u64 sdma_cmd_offset,
      u64 imm_buf_offset,
      int bdc_cmd_num,
      int gdma_cmd_num,
      int hau_cmd_num,
      int sdma_cmd_num,
      u32 bdc_cmd_byte_size,
      u32 gdma_cmd_byte_size,
      u32 hau_cmd_byte_size,
      u32 sdma_cmd_byte_size,
      u32 total_io_size,
      u64 placeholder)
  {

#ifdef USING_FAKE_DDR_MODE
    const int core_offset = PLD_BASE_ADDR;
#else
    const int core_offset = 0;
#endif
    const u64 CMODEL_GMEM_START_ADDR = 0x0;
    int base_idx[] = {0};
    // always reserve 0xc000000 for firmware
    u64 _base_addr = placeholder;
    _base_addr += (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR + core_offset);
    u64 base_addr[] = {_base_addr};
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);
    u32 write_gdma_reg, write_sdma_reg;

    if (bdc_cmd_num != 0)
    {
      ASSERT((bdc_cmd_offset & 0x7f) == 0x0);
      u64 bdc_cmd_offset_shift = bdc_cmd_offset >> 5;
      u32 write_bdc_reg = (bdc_cmd_offset_shift & 0xffffffff);
      write_bdc_reg = write_bdc_reg | 0x1;
      WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
      CORE_PRINT(" bdc des config done, cmd_num=%d, core_id=%d\n", bdc_cmd_num, CORE_ID);
    }
    if (gdma_cmd_num != 0)
    {
      ASSERT((gdma_cmd_offset & 0x7f) == 0x0);
      u64 gdma_cmd_offset_shift = gdma_cmd_offset >> 7;
      write_gdma_reg = (gdma_cmd_offset_shift & 0x0fffffff);
      WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_GDMA);

      write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
      write_gdma_reg = write_gdma_reg | 0x1;
      WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
      CORE_PRINT(" gdma des config done, cmd_num=%d, core_id=%d\n", gdma_cmd_num, CORE_ID);
    }

    if (hau_cmd_num != 0)
    {
      ASSERT((hau_cmd_offset & 0x3) == 0);
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_HAU);
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2, ((hau_cmd_offset & 0xffffffff) >> 2));
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32, (hau_cmd_offset >> 32));
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
      CORE_PRINT(" hau des config done, cmd_num=%d, core_id=%d\n", hau_cmd_num, CORE_ID);
    }
    if (sdma_cmd_num != 0)
    {
      ASSERT((sdma_cmd_offset & 0x7f) == 0x0);
      u64 sdma_cmd_offset_shift = sdma_cmd_offset >> 7;
      write_sdma_reg = (sdma_cmd_offset_shift & 0x0fffffff);
      WRITE_REG((SDMA_ENGINE_MAIN_CTRL + 0x4), write_sdma_reg, NODECHIP_REG);
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_SDMA);

      write_sdma_reg = READ_REG(SDMA_ENGINE_MAIN_CTRL);
      write_sdma_reg = write_sdma_reg | 0x1;
      WRITE_REG(SDMA_ENGINE_MAIN_CTRL, write_sdma_reg, NODECHIP_REG);
      CORE_PRINT(" sdma des config done, cmd_num=%d, core_id=%d\n", sdma_cmd_num, CORE_ID);
    }

#ifdef USING_CMODEL
    bool using_cmd_arr[4];
    using_cmd_arr[ENGINE_BD] = (bdc_cmd_num != 0);
    using_cmd_arr[ENGINE_GDMA] = (gdma_cmd_num != 0);
    using_cmd_arr[ENGINE_HAU] = (hau_cmd_num != 0);
    using_cmd_arr[ENGINE_SDMA] = (sdma_cmd_num != 0);

    u64 engine_address_arr[4];
    engine_address_arr[ENGINE_BD] = bdc_cmd_offset;
    engine_address_arr[ENGINE_GDMA] = gdma_cmd_offset;
    engine_address_arr[ENGINE_HAU] = hau_cmd_offset;
    engine_address_arr[ENGINE_SDMA] = sdma_cmd_offset;

    cmodel_multi_engine(engine_address_arr, using_cmd_arr, CORE_ID);
#endif

    id_node.bd_cmd_id = bdc_cmd_num;
    id_node.gdma_cmd_id = gdma_cmd_num;
    id_node.hau_cmd_id = hau_cmd_num;
    id_node.sdma_cmd_id = sdma_cmd_num;
    poll_all_engine_done(&id_node);
    CORE_PRINT("poll done.  bdc gdma hau cmd num( %d, %d, %d)\n", bdc_cmd_num, gdma_cmd_num, hau_cmd_num);
  }

void msg_sync_multi_core(const void *api_buf)
{
    sg_api_msg_sync_multi_core_t *msg_sync_multi_core = (sg_api_msg_sync_multi_core_t *)api_buf;
    int cur_core_id = CORE_ID;
    u64 CMODEL_GMEM_START_ADDR = 0x0;
    u64 bdc_cmd_offset = msg_sync_multi_core->param[cur_core_id].tpu_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    u64 gdma_cmd_offset = msg_sync_multi_core->param[cur_core_id].gdma_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    u64 hau_cmd_offset = msg_sync_multi_core->param[cur_core_id].hau_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    u64 sdma_cmd_offset = msg_sync_multi_core->param[cur_core_id].sdma_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    u64 imm_buf_offset = msg_sync_multi_core->param[cur_core_id].imm_buf_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    int bdc_cmd_num = msg_sync_multi_core->param[cur_core_id].tpu_cmd_nums;
    int gdma_cmd_num = msg_sync_multi_core->param[cur_core_id].gdma_cmd_nums;
    int hau_cmd_num = msg_sync_multi_core->param[cur_core_id].hau_cmd_nums;
    int sdma_cmd_num = msg_sync_multi_core->param[cur_core_id].sdma_cmd_nums;
    u32 bdc_cmd_byte_size = msg_sync_multi_core->param[cur_core_id].tpu_cmd_byte_size;
    u32 gdma_cmd_byte_size = msg_sync_multi_core->param[cur_core_id].gdma_cmd_byte_size;
    u32 hau_cmd_byte_size = msg_sync_multi_core->param[cur_core_id].hau_cmd_byte_size;
    u32 sdma_cmd_byte_size = msg_sync_multi_core->param[cur_core_id].sdma_cmd_byte_size;
    u32 imm_buf_byte_size = msg_sync_multi_core->param[cur_core_id].imm_buf_byte_size;
    int loop = msg_sync_multi_core->loop;
    int enable_pio_des_interleave = msg_sync_multi_core->enable_pio_des_interleave;
    int pio_addr = msg_sync_multi_core->param[cur_core_id].pio_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    UNUSED(imm_buf_byte_size);
    int input_num = msg_sync_multi_core->input_num;
    int output_num = msg_sync_multi_core->output_num;
    unsigned long long placeholder = msg_sync_multi_core->placeholder_addr;
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);
    for(int i = 0; i < input_num; i++)
    {
      unsigned long long fixed_addr = placeholder + msg_sync_multi_core->input[i].origin_addr;
      general_gdma_gen_cmd(
        msg_sync_multi_core->input[i].addr,
        fixed_addr,
        GDMA_INT8,
        msg_sync_multi_core->input[i].size,
        false,
        MASTER_THREAD,
        &id_node);
    }
    poll_all_engine_done(&id_node);

    for (int i = 0; i < loop; i++)
    {
      // set base msg id
      int base_msg_id = tpu_get_base_msg_id();
      tpu_set_base_msg_id(base_msg_id);

      FW_DBG("loop: %d\n", i);
      nodechip_multi_msg_sync_multi_core(
          bdc_cmd_offset,
          gdma_cmd_offset,
          hau_cmd_offset,
          sdma_cmd_offset,
          imm_buf_offset,
          bdc_cmd_num,
          gdma_cmd_num,
          hau_cmd_num,
          sdma_cmd_num,
          bdc_cmd_byte_size,
          gdma_cmd_byte_size,
          hau_cmd_byte_size,
          sdma_cmd_byte_size,
          msg_sync_multi_core->total_io_size,
          placeholder);
      // reset base message id
      tpu_set_base_msg_id(0);

      int base_idx[] = {0};
#ifdef USING_FAKE_DDR_MODE
      u64 base_addr[] = {PLD_BASE_ADDR + (GLOBAL_MEM_START_ADDR - 0x0)};
#else
      u64 base_addr[] = {0};
#endif
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_GDMA);
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_SDMA);
      atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_HAU);
      resync_cmd_id(&id_node);
      for(int j = 0; j < output_num; j++)
      {
        unsigned long long fixed_addr = placeholder + msg_sync_multi_core->output[j].origin_addr;
        general_gdma_gen_cmd(
          fixed_addr,
          msg_sync_multi_core->output[j].addr,
          GDMA_INT8,
          msg_sync_multi_core->output[j].size,
          false,
          MASTER_THREAD,
          &id_node);
      }
      poll_all_engine_done(&id_node);
      if (enable_pio_des_interleave)
      {
        FW_DBG("enter pio\n");
        tpu_initialize();
        int shape[4] = {2, 3, 50, 4};
        int len = 2 * 3 * 4 * 50;
        u64 input_addr = pio_addr;
        u64 output_addr = ALIGN(pio_addr + len * sizeof(float), 128);
        extern void nodechip_active(
            u64, u64, int *, int, data_type_t, sg_active_type_t, float *);
        nodechip_active(
            input_addr,
            output_addr,
            shape,
            4,
            DT_FP32,
            ACTIVE_EXP,
            NULL);

        tpu_poll();
      }
    }
  }
TPUKERNEL_FUNC_REGISTER(msg_sync_multi_core)
#ifdef __cplusplus
}
#endif
