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
#ifdef USING_FAKE_DDR_MODE
#include "atomic_sys_gen_cmd.h"
#endif
#ifdef __cplusplus
extern "C"
{
#endif

void nodechip_multi_msg_sync(const sg_api_msg_sync_t *api)
{
#ifdef USING_FAKE_DDR_MODE
    const int core_offset = PLD_BASE_ADDR;
    const u64 CMODEL_GMEM_START_ADDR = 0x0;
    int base_idx = 0;
    //实际中基地址可能为负数
    // 0xc000000 is 192M for jumping multicore firmware packages
    u64 base_addr = ALIGN(0xc000000, 4096) +
        ALIGN((api->tpu_cmd_byte_size), 4096) + ALIGN((api->gdma_cmd_byte_size), 4096) +
        ALIGN((api->hau_cmd_byte_size), 4096) + ALIGN((api->sdma_cmd_byte_size), 4096) +
        ALIGN((api->imm_buf_byte_size), 4096);
    base_addr += (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR + core_offset);
    CORE_PRINT(" bdc gdma hau sdma cmd num( %d, %d, %d, %d) ,offset( %llx, %llx, %llx, %llx), base_addr=0x%llx, core_id=%d\n",
     api->tpu_cmd_nums, api->gdma_cmd_nums, api->hau_cmd_nums, api->sdma_cmd_nums, api->tpu_cmd_addr,
     api->gdma_cmd_addr, api->hau_cmd_addr, api->sdma_cmd_addr, base_addr, CORE_ID);
#endif

    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);
    u32 write_gdma_reg, write_sdma_reg;

    if (api->tpu_cmd_nums != 0)
    {
      ASSERT((api->tpu_cmd_addr & 0x7f) == 0x0);
      u64 bdc_cmd_offset_shift = api->tpu_cmd_addr >> 5;
      u32 write_bdc_reg = (bdc_cmd_offset_shift & 0xffffffff);
      write_bdc_reg = write_bdc_reg | 0x1;
      WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
      CORE_PRINT(" bdc des config done, cmd_num=%d, core_id=%d\n", api->tpu_cmd_nums, CORE_ID);
    }
    if (api->gdma_cmd_nums != 0)
    {
      ASSERT((api->gdma_cmd_addr & 0x7f) == 0x0);
      u64 gdma_cmd_offset_shift = api->gdma_cmd_addr >> 7;
      write_gdma_reg = (gdma_cmd_offset_shift & 0x0fffffff);
      WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
#ifdef USING_FAKE_DDR_MODE
      atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_GDMA);
#endif

      write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
      write_gdma_reg = write_gdma_reg | 0x1;
      WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
      CORE_PRINT(" gdma des config done, cmd_num=%d, core_id=%d\n", api->gdma_cmd_nums, CORE_ID);
    }

    if (api->hau_cmd_nums != 0)
    {
      ASSERT((api->hau_cmd_addr & 0x3) == 0);
#ifdef USING_FAKE_DDR_MODE
      atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_HAU);
#endif
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2, ((api->hau_cmd_addr & 0xffffffff) >> 2));
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32, (api->hau_cmd_addr >> 32));
      FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
      CORE_PRINT(" hau des config done, cmd_num=%d, core_id=%d\n", api->hau_cmd_nums, CORE_ID);
    }
    if (api->sdma_cmd_nums != 0)
    {
      ASSERT((api->sdma_cmd_addr & 0x7f) == 0x0);
      u64 sdma_cmd_offset_shift = api->sdma_cmd_addr >> 7;
      write_sdma_reg = (sdma_cmd_offset_shift & 0x0fffffff);
      WRITE_REG((SDMA_ENGINE_MAIN_CTRL + 0x4), write_sdma_reg, NODECHIP_REG);
#ifdef USING_FAKE_DDR_MODE
      atomic_set_base_ddr(&base_idx, &base_addr, 1, ENGINE_SDMA);
#endif

      write_sdma_reg = READ_REG(SDMA_ENGINE_MAIN_CTRL);
      write_sdma_reg = write_sdma_reg | 0x1;
      WRITE_REG(SDMA_ENGINE_MAIN_CTRL, write_sdma_reg, NODECHIP_REG);
      CORE_PRINT(" sdma des config done, cmd_num=%d, core_id=%d\n", api->sdma_cmd_nums, CORE_ID);
    }

#ifdef USING_CMODEL
    bool using_cmd_arr[4];
    using_cmd_arr[ENGINE_BD] = (api->tpu_cmd_nums != 0);
    using_cmd_arr[ENGINE_GDMA] = (api->gdma_cmd_nums != 0);
    using_cmd_arr[ENGINE_HAU] = (api->hau_cmd_nums != 0);
    using_cmd_arr[ENGINE_SDMA] = (api->sdma_cmd_nums != 0);

    u64 engine_address_arr[4];
    engine_address_arr[ENGINE_BD] = api->tpu_cmd_addr;
    engine_address_arr[ENGINE_GDMA] = api->gdma_cmd_addr;
    engine_address_arr[ENGINE_HAU] = api->hau_cmd_addr;
    engine_address_arr[ENGINE_SDMA] = api->sdma_cmd_addr;

    cmodel_multi_engine(engine_address_arr, using_cmd_arr, get_cur_nodechip_idx());
#endif

    id_node.bd_cmd_id = api->tpu_cmd_nums;
    id_node.gdma_cmd_id = api->gdma_cmd_nums;
    id_node.hau_cmd_id = api->hau_cmd_nums;
    id_node.sdma_cmd_id = api->sdma_cmd_nums;
    poll_all_engine_done(&id_node);
    CORE_PRINT("poll done.  bdc gdma hau cmd num( %d, %d, %d)\n", api->tpu_cmd_nums, api->gdma_cmd_nums, api->hau_cmd_nums);
  }

  sg_fw_status_t sg_api_msg_sync(
      unsigned char *api_buf,
      int size)
  {
// #ifdef USING_CMODEL
//     return SG_FW_SUCCESS;
// #endif
    const sg_api_msg_sync_t *msg_sync_ptr = (sg_api_msg_sync_t *)api_buf;
    ASSERT(api_buf && size == sizeof(sg_api_msg_sync_t));
    sg_api_msg_sync_t api = {0};
    // PLD_CMD_ADDR = PLD_GLOBAL_MEM_START_ADDR - CMODEL_GLOBAL_MEM_START_ADDR
    u64 CMODEL_GMEM_START_ADDR = 0x0;
    api.tpu_cmd_addr = msg_sync_ptr->tpu_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.gdma_cmd_addr = msg_sync_ptr->gdma_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.hau_cmd_addr = msg_sync_ptr->hau_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.sdma_cmd_addr = msg_sync_ptr->sdma_cmd_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.imm_buf_addr = msg_sync_ptr->imm_buf_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.pio_addr = msg_sync_ptr->pio_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.tpu_cmd_nums = msg_sync_ptr->tpu_cmd_nums;
    api.gdma_cmd_nums = msg_sync_ptr->gdma_cmd_nums;
    api.sdma_cmd_nums = msg_sync_ptr->sdma_cmd_nums;
    api.hau_cmd_nums = msg_sync_ptr->hau_cmd_nums;
    api.tpu_cmd_byte_size = msg_sync_ptr->tpu_cmd_byte_size;
    api.gdma_cmd_byte_size = msg_sync_ptr->gdma_cmd_byte_size;
    api.sdma_cmd_byte_size = msg_sync_ptr->sdma_cmd_byte_size;
    api.hau_cmd_byte_size = msg_sync_ptr->hau_cmd_byte_size;
    api.imm_buf_byte_size = msg_sync_ptr->imm_buf_byte_size;
    api.loop = msg_sync_ptr->loop;
    api.enable_pio_des_interleave = msg_sync_ptr->enable_pio_des_interleave;
    api.input_addr = msg_sync_ptr->input_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    api.output_addr = msg_sync_ptr->output_addr + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    for (int i = 0; i < api.loop; i++)
    {
      FW_DBG("loop: %d\n", i);
      nodechip_multi_msg_sync(&api);

      if (api.enable_pio_des_interleave)
      {
        FW_DBG("enter pio\n");
        tpu_initialize();
        int shape[4] = {2, 3, 50, 4};
        int len = 2 * 3 * 4 * 50;
        u64 input_addr = api.pio_addr;
        u64 output_addr = ALIGN(api.pio_addr + len * sizeof(float), 128);
        // u64 output_idx_addr = ALIGN(output_addr + len * 4, 128);
        // u64 buffer_addr = ALIGN(output_idx_addr + len * 4, 128);
        // extern void nodechip_sort_per_dim(
        //     u64, u64, u64, u64, int*, int, int, int, int, int, data_type_t);
        // nodechip_sort_per_dim(
        //     input_addr,
        //     output_addr,
        //     output_idx_addr,
        //     buffer_addr,
        //     shape,
        //     4,
        //     3,
        //     0,
        //     1,
        //     1,
        //     DT_FP32);

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

    return SG_FW_SUCCESS;
  }
#ifdef __cplusplus
}
#endif
