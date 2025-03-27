#ifdef USING_CMODEL
#include "cmodel_multi_thread.h"
#endif
#ifdef PIO_DESC_INTERLEAVE
#include "test_pio.h"
#endif

#include "firmware_common.h"
#include "firmware_runtime.h"
#include "atomic_gdma_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"
#include "firmware_timer.h"
#include "gdma_reg_value.h"
#include "firmware_common_inline.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include "sg_api_struct.h"

#define REMOVE_MSG_SYNC

#ifdef USING_CMODEL
#include "cmodel_common.h"
#else
#include "firmware_top.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// align with compiler
enum TagType {
  TAG_USERS = 0,
  TAG_WEIGHT = 0x1,
  TAG_ACTIVATION = 0x2,
  TAG_L2M = 0x1e,
};

#pragma pack(1)
typedef struct {
  uint64_t user_ddr;
  uint64_t cmd_ddr;
  uint32_t byte_size;
} io_info_t;
typedef struct {
  uint32_t bdc_cmd_num;
  uint32_t gdma_cmd_num;
  uint32_t bdc_cmd_byte_size;
  uint32_t gdma_cmd_byte_size;
} cmd_group_t;
typedef struct {
  uint32_t input_num;
  io_info_t *input_info;
  uint32_t output_num;
  io_info_t *output_info;
  uint64_t bdc_cmd_addr;
  uint64_t gdma_cmd_addr;
  uint32_t cmd_group_num;
  cmd_group_t *cmd_group;
  uint64_t coeff_start_addr;
  uint64_t activation_start_addr;
  uint32_t core_list;
  uint64_t hau_cmd_addr;
  uint64_t sdma_cmd_addr;
} multi_fullnet_info_t;
#pragma pack()

static void parse_api_info(const void *api_info,
                           multi_fullnet_info_t *net_info) {
#define IR_UNPACK(dst_ptr, ir_ptr, size)                                       \
  memcpy(dst_ptr, ir_ptr, size);                                               \
  ir_ptr = (char *)ir_ptr + size

  const char *s8_p = (const char *)api_info;
  memset(net_info, 0x0, sizeof(net_info));
  IR_UNPACK(&net_info->input_num, s8_p, sizeof(net_info->input_num));
  if(net_info->input_num == 0){
    u64 command_addr=0, command_size=0;
    IR_UNPACK(&command_addr, s8_p, sizeof(u64));
    IR_UNPACK(&command_size, s8_p, sizeof(u64));
    invalidate_cache(command_addr, ALIGN(command_size,64));
    s8_p = (const char*)GET_GLOBAL_ADDR(command_addr);
    IR_UNPACK(&net_info->input_num, s8_p, sizeof(net_info->input_num));
  }
  int buffer_size = sizeof(io_info_t) * net_info->input_num;
  net_info->input_info = (io_info_t *)malloc(buffer_size);
  IR_UNPACK(net_info->input_info, s8_p, buffer_size);

  IR_UNPACK(&net_info->output_num, s8_p, sizeof(net_info->output_num));
  buffer_size = net_info->output_num * sizeof(io_info_t);
  net_info->output_info = (io_info_t *)malloc(buffer_size);
  IR_UNPACK(net_info->output_info, s8_p, buffer_size);

  IR_UNPACK(&net_info->bdc_cmd_addr, s8_p, sizeof(net_info->bdc_cmd_addr));
  IR_UNPACK(&net_info->gdma_cmd_addr, s8_p, sizeof(net_info->gdma_cmd_addr));
  IR_UNPACK(&net_info->cmd_group_num, s8_p, sizeof(net_info->cmd_group_num));

  buffer_size = net_info->cmd_group_num * sizeof(cmd_group_t);
  net_info->cmd_group = (cmd_group_t *)malloc(buffer_size);
  IR_UNPACK(net_info->cmd_group, s8_p, buffer_size);

  IR_UNPACK(&net_info->coeff_start_addr, s8_p,
            sizeof(net_info->coeff_start_addr));
  IR_UNPACK(&net_info->activation_start_addr, s8_p,
            sizeof(net_info->activation_start_addr));
  IR_UNPACK(&net_info->core_list, s8_p, sizeof(net_info->core_list));
  IR_UNPACK(&net_info->hau_cmd_addr, s8_p, sizeof(net_info->hau_cmd_addr));
  IR_UNPACK(&net_info->sdma_cmd_addr, s8_p, sizeof(net_info->sdma_cmd_addr));

#undef IR_UNPACK
}
void nodechip_multi_fullnet(
    u64 bdc_cmd_offset,
    u64 gdma_cmd_offset,
    int bdc_cmd_num,
    int gdma_cmd_num,
    u64 hau_cmd_addr,
    u64 sdma_cmd_addr);

sg_fw_status_t sg_api_multi_fullnet(unsigned char *api_buf, int size) {
  UNUSED(size);
  uint32_t idx = 0;
  multi_fullnet_info_t net_info;
  int offset = 0;
  if (tpu_tpuv7_env())
    offset = (size / tpu_workitem_num()) * tpu_workitem_index();
  parse_api_info(api_buf + offset, &net_info);
  FW_INFO("input_num = %d\n", net_info.input_num);
  FW_INFO("output_num = %d\n", net_info.output_num);
#ifdef USING_FAKE_DDR_MODE
  for (idx = 0; idx < net_info.cmd_group_num; ++idx) {
    CORE_PRINT(
        " bdc_cmd_byte_size[%d] = %d, gdma_cmd_byte_size[%d] = %d    core %d\n",
        idx, net_info.cmd_group[idx].bdc_cmd_byte_size, idx,
        net_info.cmd_group[idx].gdma_cmd_byte_size, CORE_ID);
  }
#endif

  // To be compatible with the old runtime
  if (!tpu_tpuv7_env())
    tpu_core_context_setup(CORE_ID, MAX_TPU_CORE_NUM, net_info.core_list);
  tpu_initialize();
  for (idx = 0; idx < net_info.input_num; ++idx) {
    FW_INFO("user_ddr 0x%lx, cmd_ddr 0x%lx, data_size %u\n",
            net_info.input_info[idx].user_ddr, net_info.input_info[idx].cmd_ddr,
            net_info.input_info[idx].byte_size);
    if (net_info.input_info[idx].byte_size == 0 ||
        net_info.input_info[idx].user_ddr == net_info.input_info[idx].cmd_ddr)
      continue;

    tpu_gdma_system_cpy(net_info.input_info[idx].cmd_ddr,
                        net_info.input_info[idx].user_ddr,
                        net_info.input_info[idx].byte_size, DT_INT8);
  }

#ifdef USING_PERF_MODE
  tpu_sync_all();
#endif

  // wait for the ready of all input data
  tpu_poll();

  {
    u64 core_offset = 0;
#ifdef USING_FAKE_DDR_MODE
    core_offset = PLD_BASE_ADDR + (GLOBAL_MEM_START_ADDR - 0x0);
#endif
    // set base addr
    int use_core_id = 0;
#ifdef PLD_MULTI_MESSION
    use_core_id = CORE_ID;
#else
    for (int i = 0; i < MAX_TPU_CORE_NUM; ++i) {
      if (net_info.core_list & (1 << i)) {
        use_core_id = i;
        break;
      }
    }
#endif
    // To be compatible with the old runtime
    if (tpu_tpuv7_env()) {
      if (net_info.core_list > 1) { //this is parallel mode, that 1 core bmodel run on multi cores
        use_core_id = tpu_start_physical_core_id()  + tpu_workitem_index();
        FW_INFO("use core id %d\n", use_core_id);
      }
      else
        use_core_id = tpu_start_physical_core_id();
    }
    // L2M base_addr = 16M * core_id
    int base_idx[] = {TAG_WEIGHT, TAG_ACTIVATION, TAG_L2M};
    u64 base_addr[] = {net_info.coeff_start_addr + core_offset,
                       net_info.activation_start_addr + core_offset,
                       0x1000000 * use_core_id};
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_GDMA);
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_SDMA);
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_HAU);

    // set base message id
    int base_id = tpu_get_base_msg_id();
    tpu_set_base_msg_id(base_id);
  }

  FW_INFO("[core=%d] hau_cmd_addr %lx, sdma_cmd_addr %lx\n", CORE_ID,
          net_info.hau_cmd_addr, net_info.sdma_cmd_addr);

  u64 bdc_cmd_offset = net_info.bdc_cmd_addr;
  u64 gdma_cmd_offset = net_info.gdma_cmd_addr;
  for (idx = 0; idx < net_info.cmd_group_num; idx++) {
    const cmd_group_t *cmd_group = &net_info.cmd_group[idx];
    FW_INFO("bdc_cmd_num %u, gdma_cmd_num %u, group_idx %u\n",
            cmd_group->bdc_cmd_num, cmd_group->gdma_cmd_num, idx);
    nodechip_multi_fullnet(bdc_cmd_offset, gdma_cmd_offset,
                           cmd_group->bdc_cmd_num, cmd_group->gdma_cmd_num,
                           idx == 0 ? net_info.hau_cmd_addr : 0,
                           idx == 0 ? net_info.sdma_cmd_addr : 0);
    bdc_cmd_offset += cmd_group->bdc_cmd_byte_size;
    gdma_cmd_offset += cmd_group->gdma_cmd_byte_size;
  }

  {
    // reset base addr
    int base_idx[] = {TAG_WEIGHT, TAG_ACTIVATION, TAG_L2M};
    u64 base_addr[] = {0, 0, 0};
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_GDMA);
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_SDMA);
    atomic_set_base_ddr(base_idx, base_addr, 3, ENGINE_HAU);

    // reset base message id
    tpu_set_base_msg_id(0);
  }
  // move data to user defined address
  tpu_initialize();
  for (idx = 0; idx < net_info.output_num; ++idx) {
    if (net_info.output_info[idx].byte_size == 0 ||
        net_info.output_info[idx].user_ddr == net_info.output_info[idx].cmd_ddr)
      continue;
    tpu_gdma_system_cpy(net_info.output_info[idx].user_ddr,
                        net_info.output_info[idx].cmd_ddr,
                        net_info.output_info[idx].byte_size, DT_INT8);
  }

  // wait for the ready of output data
  tpu_poll();

  free(net_info.input_info);
  free(net_info.output_info);
  free(net_info.cmd_group);

#if defined(ENABLE_TIMER_FOR_FULLNET)
  firmware_timer_print();
#endif

  return SG_FW_SUCCESS;
}

THREAD CMD_ID_NODE des_id_node;

#ifdef DES_POLL_DELAY
THREAD u64 des_start_timestamp;
#endif


void tpu_poll_descriptor_internal()
{
#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "tpu_poll_descriptor_internal GDMA %d, TIU %d, SDMA %d...\n",
        des_id_node.gdma_cmd_id,
        des_id_node.bd_cmd_id,
        des_id_node.sdma_cmd_id);
#endif

#ifdef DES_POLL_DELAY
    // Magical ~20us delay before poll is required for short-duration
    // descriptor like binary
    const u64 magical_delay = 200000llu;
    while (firmware_timer_get_cycle() - des_start_timestamp < magical_delay);
#endif

#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "poll_all_engine_done, GDMA %d, TIU %d, SDMA %d.\n",
        des_id_node.gdma_cmd_id,
        des_id_node.bd_cmd_id,
        des_id_node.sdma_cmd_id);
#endif

    poll_bd_engine_des_done();
    poll_gdma_engine_des_done();
    poll_sdma_engine_des_done();


    poll_all_engine_done(&des_id_node);
#ifdef DUMP_CACHED_CMD
    TP_DEBUG("Poll desc regs done.\n");
#endif

    resync_cmd_id(&des_id_node);

#ifdef DUMP_CACHED_CMD
    TP_DEBUG("CMD reset.\n");
#endif
}

bool tpu_is_descriptor_mode()
{
#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "tpu_is_descriptor_mode GDMA %d, TIU %d, CDMA %d, SDMA %d.\n",
        des_id_node.gdma_cmd_id, des_id_node.bd_cmd_id, des_id_node.cdma_cmd_id[2], des_id_node.sdma_cmd_id);
#endif
    return des_id_node.gdma_cmd_id || des_id_node.sdma_cmd_id;
}

void nodechip_multi_fullnet_ext(
    u64 bdc_cmd_offset,
    u64 gdma_cmd_offset,
    int bdc_cmd_num,
    int gdma_cmd_num,
    int sdma_cmd_num,
    u64 hau_cmd_addr,
    u64 sdma_cmd_addr,
    int do_poll)
{
#ifdef USING_FAKE_DDR_MODE
    u64 CMODEL_GMEM_START_ADDR = 0x0;
    const u64 core_offset = PLD_BASE_ADDR + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    bdc_cmd_offset += core_offset;
    gdma_cmd_offset += core_offset;
    if (hau_cmd_addr) {
      hau_cmd_addr += core_offset;
    }
    if (sdma_cmd_addr) {
      sdma_cmd_addr += core_offset;
    }

    CORE_PRINT(" bdc gdma  cmd num( %d, %d) ,offset( %llx, %llx), core_id=%d\n",
              bdc_cmd_num, gdma_cmd_num, bdc_cmd_offset, gdma_cmd_offset,  CORE_ID);
#endif
  resync_cmd_id(&des_id_node);
  if(bdc_cmd_num == 1)
    bdc_cmd_num = 0;
  if(gdma_cmd_num == 1)
    gdma_cmd_num = 0;
  if(sdma_cmd_num == 1)
  {
    sdma_cmd_num = 0;
    sdma_cmd_addr = 0;
  }
  u32 write_gdma_reg;
  if (bdc_cmd_num != 0) {
    ASSERT((bdc_cmd_offset & 0x7f) == 0x0);
    u64 bdc_cmd_offset_shift = bdc_cmd_offset >> 5;
    u32 write_bdc_reg = (bdc_cmd_offset_shift & 0x3fffffffcLLU);
    write_bdc_reg = write_bdc_reg | 0x1;
    WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
    (void)(write_bdc_reg);
  }
  if (gdma_cmd_num != 0) {
    ASSERT((gdma_cmd_offset & 0x7f) == 0x0);
    u64 gdma_cmd_offset_shift = gdma_cmd_offset >> 7;
    write_gdma_reg = (gdma_cmd_offset_shift & 0x3fffffff);
    WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
    write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
    write_gdma_reg = write_gdma_reg | 0x1;
    WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
    (void)(write_gdma_reg);
  }

  if (hau_cmd_addr)
  {
    ASSERT((hau_cmd_addr & 0x3) == 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2, ((hau_cmd_addr & 0xffffffff) >> 2));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32, (hau_cmd_addr >> 32));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
  }

  if (sdma_cmd_addr) {
    ASSERT((sdma_cmd_addr & 0x7f) == 0x0);
    u64 sdma_cmd_shift = sdma_cmd_addr >> 7;
    u32 write_sdma_reg = (sdma_cmd_shift & 0x3fffffff);
    WRITE_REG((SDMA_ENGINE_MAIN_CTRL + 0x4), write_sdma_reg, NODECHIP_REG);
    write_sdma_reg = READ_REG(SDMA_ENGINE_MAIN_CTRL);
    write_sdma_reg = write_sdma_reg | 0x1;
    WRITE_REG(SDMA_ENGINE_MAIN_CTRL, write_sdma_reg, NODECHIP_REG);
    (void)(write_sdma_reg);
  }

#ifdef USING_FAKE_DDR_MODE
  CORE_PRINT("des config done    core %d\n", CORE_ID);
#endif
#ifdef USING_CMODEL
  bool using_cmd_arr[ENGINE_END] = {0};
  using_cmd_arr[ENGINE_BD] = (bdc_cmd_num != 0);
  using_cmd_arr[ENGINE_GDMA] = (gdma_cmd_num != 0);
  using_cmd_arr[ENGINE_HAU] = (hau_cmd_addr != 0);
  using_cmd_arr[ENGINE_SDMA] = (sdma_cmd_addr != 0);

  u64 engine_address_arr[ENGINE_END] = {0};
  engine_address_arr[ENGINE_BD] = bdc_cmd_offset;
  engine_address_arr[ENGINE_GDMA] = gdma_cmd_offset;
  engine_address_arr[ENGINE_HAU] = hau_cmd_addr;
  engine_address_arr[ENGINE_SDMA] = sdma_cmd_addr;

  cmodel_multi_engine(engine_address_arr, using_cmd_arr, get_cur_nodechip_idx());
#endif
  des_id_node.bd_cmd_id = bdc_cmd_num;
  des_id_node.gdma_cmd_id = gdma_cmd_num;
  des_id_node.sdma_cmd_id = sdma_cmd_num;

#ifdef DES_POLL_DELAY
  des_start_timestamp = firmware_timer_get_cycle();
#endif

  if (!do_poll) return;

  tpu_poll_descriptor_internal();

#if defined(PIO_DESC_INTERLEAVE) && (!(defined(USING_CMODEL)))
  test_pio_inst();
#endif
}

void nodechip_multi_fullnet(
    u64 bdc_cmd_offset,
    u64 gdma_cmd_offset,
    int bdc_cmd_num,
    int gdma_cmd_num,
    u64 hau_cmd_addr,
    u64 sdma_cmd_addr)
{
    nodechip_multi_fullnet_ext(
        bdc_cmd_offset, gdma_cmd_offset,
        bdc_cmd_num, gdma_cmd_num, 0,
        hau_cmd_addr, sdma_cmd_addr,
        1);
}

#define CACHE_ENGINE_NUM 4

typedef struct
{
    sg_api_launch_cache_multicore_t m;
    sg_api_cmd_descriptor cmds[CACHE_ENGINE_NUM + CDMA_API_NUM];
} fixed_sized_cache_param;

typedef struct
{
    int batch;
    fixed_sized_cache_param data[16];
} async_descs_remainder_t;

THREAD async_descs_remainder_t async_descs_remainder;

void nodechip_multi_fullnet_for_c2c(
  u64 *vsdma_cmd_addr,
  int vsdma_engine_num,
  int *vsdma_cmd_num,
  u64 *cdma_cmd_addr,
  int cdma_engine_num,
  int *cdma_cmd_num,
  u64 gdma_cmd_addr,
  int gdma_cmd_num,
  int nranks,
  int cur_rank,
  int *chip_map
) {
  CMD_ID_NODE id_node;
  memset(&id_node, 0, sizeof(id_node));
  // resync_cmd_id(&id_node);
  tpu_sccl_init(nranks, cur_rank, chip_map, 1); // using_ring
  int* cdma_ports = tpu_cdma_get_used_ports();
  if(gdma_cmd_num > 0) {
    FW_REG_ID_WRITE(GDMA_ENGINE_MAIN_CTRL, GDMA_ID_CFG_CMD_ID_RESET, 1);
    id_node.gdma_cmd_id = 0;
    ASSERT((gdma_cmd_addr & 0x7f) == 0x0);
    u64 gdma_cmd_offset_shift = gdma_cmd_addr >> 7;
    u32 write_gdma_reg = (gdma_cmd_offset_shift & 0x3fffffff);
    WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
    write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
    write_gdma_reg = write_gdma_reg | 0x1;
    WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
    (void)(write_gdma_reg);
  }
  if(vsdma_engine_num > 0) {
    for(int i = 0; i < vsdma_engine_num; ++i) 
    {
      // config cmd id
      resync_vsdma_port_subsys_cmd_id(&id_node, i);
      id_node.vsdma_cmd_id[i] = vsdma_cmd_num[i];
      // config cmd addr
      ASSERT((vsdma_cmd_addr[i] & 0x7f) == 0x0);
      u64 vsdma_cmd_shift = vsdma_cmd_addr[i] >> 7;
      u32 write_vsdma_reg = (vsdma_cmd_shift & 0x3fffffff);
      WRITE_REG((VSDMA_ENGINE_MAIN_CTRL(i) + 0xc), write_vsdma_reg, NODECHIP_REG);
      // enable descriptor
      write_vsdma_reg = READ_REG(VSDMA_ENGINE_MAIN_CTRL(i));
      write_vsdma_reg = write_vsdma_reg | 0x1 << 2;
      WRITE_REG(VSDMA_ENGINE_MAIN_CTRL(i), write_vsdma_reg, NODECHIP_REG);
      (void)(write_vsdma_reg);
    }
  }
  if(cdma_engine_num > 0) {
    for(int i = 0; i < cdma_engine_num; ++i) {
      int port = cdma_ports[i];
      // assume cdma_ports are sorted as right, left, peer
      resync_cdma_port_subsys_cmd_id(&id_node, port);
      id_node.cdma_cmd_id[port] = cdma_cmd_num[i];
      // config cmd addr
      u64 write_cdma_reg = cdma_cmd_addr[i] >> 7;
      WRITE_REG(CDMA_CSR_REG_DES_ADDR_L32(port), write_cdma_reg, NODECHIP_REG);
      write_cdma_reg = cdma_cmd_addr[i] >> 39;
      WRITE_REG(CDMA_CSR_REG_DES_ADDR_H1(port), write_cdma_reg, NODECHIP_REG);
      // write des rw addr
      write_cdma_reg = READ_REG(CDMA_CSR_REG_DES_RW_ADDR(port));
      write_cdma_reg &= (~(0xff));
      write_cdma_reg |= 0x80;
      WRITE_REG(CDMA_CSR_REG_DES_RW_ADDR(port), write_cdma_reg, NODECHIP_REG);
      // enable descriptor
      write_cdma_reg = READ_REG(CDMA_ENGINE_MAIN_CTRL(port));
      write_cdma_reg = write_cdma_reg | 0x1;
      WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port), write_cdma_reg, NODECHIP_REG);
      (void)(write_cdma_reg);
    }
  }
#ifdef USING_CMODEL
  bool using_cmd_arr[ENGINE_END] = {0};
  using_cmd_arr[ENGINE_BD] = 0;
  using_cmd_arr[ENGINE_GDMA] = (gdma_cmd_num != 0);
  using_cmd_arr[ENGINE_HAU] = 0;
  using_cmd_arr[ENGINE_SDMA] = 0;
  u64 engine_address_arr[ENGINE_END] = {0};
  engine_address_arr[ENGINE_BD] = 0;
  engine_address_arr[ENGINE_GDMA] = gdma_cmd_addr;
  engine_address_arr[ENGINE_HAU] = 0;
  engine_address_arr[ENGINE_SDMA] = 0;
  cmodel_multi_engine_all_engine(
    engine_address_arr,
    using_cmd_arr,
    get_cur_nodechip_idx(),
    cdma_cmd_addr,
    vsdma_cmd_addr,
    cdma_engine_num,
    vsdma_engine_num,
    cdma_ports
  );
#endif
  poll_gdma_engine_done(&id_node);
  if(vsdma_engine_num > 0) {
    for(int i = 0; i < vsdma_engine_num; ++i) {
      poll_vsdma_engine_done(i, &id_node);
    }
  }
  if(cdma_engine_num > 0) {
    for(int i = 0; i < cdma_engine_num; ++i) {
      int port = cdma_ports[i];
      poll_cdma_engine_done(port, &id_node);
    }
  }
}

int c2c_descriptor(const void *api_buf) {
  sg_api_c2c_descriptor_t *api = (sg_api_c2c_descriptor_t *)api_buf;
  int loop = api->loop;
  for(int i = 0; i < loop; ++i) {
    nodechip_multi_fullnet_for_c2c(
    api->vsdma_cmd_addr,
    api->vsdma_engine_num,
    api->vsdma_cmd_num,
    api->cdma_cmd_addr,
    api->cdma_engine_num,
    api->cdma_cmd_num,
    api->gdma_cmd_addr,
    api->gdma_cmd_num,
    api->sccl_args.nranks,
    api->sccl_args.rank,
    api->sccl_args.chip_map
    );
  }
  return 0;
}

// ATTENTION: all engine start!

inline static void resync_cmd_id_all_engine(CMD_ID_NODE *id_node)
{
  resync_cmd_id(id_node);
  if(tpu_is_last_workitem()) {
    int used_cdma_num = tpu_cdma_get_used_port_num();
    if(used_cdma_num > 0) {
      int* cdma_ports = tpu_cdma_get_used_ports();
      for(int i = 0; i < used_cdma_num; ++i) {
        if(id_node->cdma_cmd_id[cdma_ports[i]] == 0) continue;
        poll_cdma_engine_des_done(cdma_ports[i]);
        resync_cdma_port_subsys_cmd_id(id_node, cdma_ports[i]);
      }
    }
  }
}

void tpu_poll_descriptor_internal_all_engine()
{
#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "tpu_poll_descriptor_internal_all_engine GDMA %d, TIU %d, SDMA %d, CDMA %d...\n",
        des_id_node.gdma_cmd_id,
        des_id_node.bd_cmd_id,
        des_id_node.sdma_cmd_id,
        des_id_node.cdma_cmd_id[2]);
#endif

    poll_bd_engine_des_done();
    poll_gdma_engine_des_done();
    poll_sdma_engine_des_done();

#ifdef DES_POLL_DELAY
    // Magical ~20us delay before poll is required for short-duration
    // descriptor like binary
    const u64 magical_delay = 200000llu;
    while (firmware_timer_get_cycle() - des_start_timestamp < magical_delay);
#endif
    poll_all_engine_done(&des_id_node);

#ifdef DUMP_CACHED_CMD
    TP_DEBUG(
        "poll_all_engine_done, GDMA %d, TIU %d, SDMA %d.\n",
        des_id_node.gdma_cmd_id,
        des_id_node.bd_cmd_id,
        des_id_node.sdma_cmd_id);
#endif

    resync_cmd_id_all_engine(&des_id_node);
}

void firmware_kernel_tick();
void firmware_kernel_tock(int);

void nodechip_multi_fullnet_ext_all_engine(
    u64 bdc_cmd_offset,
    u64 gdma_cmd_offset,
    int bdc_cmd_num,
    int gdma_cmd_num,
    int sdma_cmd_num,
    u64 hau_cmd_addr,
    u64 sdma_cmd_addr,
    u64 *cdma_cmd_addr,
    int *cdma_cmd_num,
    u64* base_addr,
    int base_addr_num,
    int do_poll)
{
#ifdef USING_FAKE_DDR_MODE
    u64 CMODEL_GMEM_START_ADDR = 0x0;
    const u64 core_offset = PLD_BASE_ADDR + (GLOBAL_MEM_START_ADDR - CMODEL_GMEM_START_ADDR);
    bdc_cmd_offset += core_offset;
    gdma_cmd_offset += core_offset;
    if (hau_cmd_addr) {
      hau_cmd_addr += core_offset;
    }
    if (sdma_cmd_addr) {
      sdma_cmd_addr += core_offset;
    }

    CORE_PRINT(" bdc gdma  cmd num( %d, %d) ,offset( %llx, %llx), core_id=%d\n",
              bdc_cmd_num, gdma_cmd_num, bdc_cmd_offset, gdma_cmd_offset,  CORE_ID);
#endif
  int base_idx[base_addr_num];
  for (int i = 0; i < base_addr_num; ++i)
    base_idx[i] = i + 1;
  resync_cmd_id_all_engine(&des_id_node);
  // for empty descriptor cmd group
  if(bdc_cmd_num == 1)
    bdc_cmd_num = 0;
  if(gdma_cmd_num == 1)
    gdma_cmd_num = 0;
  if(sdma_cmd_num == 1)
  {
    sdma_cmd_num = 0;
    sdma_cmd_addr = 0;
  }
  u32 write_gdma_reg;
  if (gdma_cmd_num) {
    atomic_set_base_ddr(base_idx, base_addr, base_addr_num, ENGINE_GDMA);
    ASSERT((gdma_cmd_offset & 0x7f) == 0x0);
    u64 gdma_cmd_offset_shift = gdma_cmd_offset >> 7;
    write_gdma_reg = (gdma_cmd_offset_shift & 0x3fffffff);
    WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);
    write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
    write_gdma_reg = write_gdma_reg | 0x1;
    WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
    (void)(write_gdma_reg);
  }
  if (bdc_cmd_num) {
    ASSERT((bdc_cmd_offset & 0x7f) == 0x0);
    u64 bdc_cmd_offset_shift = bdc_cmd_offset >> 5;
    u32 write_bdc_reg = (bdc_cmd_offset_shift & 0x3fffffffcLLU);
    write_bdc_reg = write_bdc_reg | 0x1;
    WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
    (void)(write_bdc_reg);
  }

  if (sdma_cmd_addr) {
    atomic_set_base_ddr(base_idx, base_addr, base_addr_num, ENGINE_SDMA);
    ASSERT((sdma_cmd_addr & 0x7f) == 0x0);
    u64 sdma_cmd_shift = sdma_cmd_addr >> 7;
    u32 write_sdma_reg = (sdma_cmd_shift & 0x3fffffff);
    WRITE_REG((SDMA_ENGINE_MAIN_CTRL + 0x4), write_sdma_reg, NODECHIP_REG);
    write_sdma_reg = READ_REG(SDMA_ENGINE_MAIN_CTRL);
    write_sdma_reg = write_sdma_reg | 0x1;
    WRITE_REG(SDMA_ENGINE_MAIN_CTRL, write_sdma_reg, NODECHIP_REG);
    (void)(write_sdma_reg);
  }
  if (hau_cmd_addr)
  {
    atomic_set_base_ddr(base_idx, base_addr, base_addr_num, ENGINE_HAU);
    ASSERT((hau_cmd_addr & 0x3) == 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2, ((hau_cmd_addr & 0xffffffff) >> 2));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32, (hau_cmd_addr >> 32));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
  }
#ifdef USING_CMODEL
  int cmodel_used_cdma_num = 0;
#endif
  int used_cdma_num = tpu_cdma_get_used_port_num();
  if(used_cdma_num > 0) {
    int* cdma_ports = tpu_cdma_get_used_ports();
    for(int i = 0; i < used_cdma_num; ++i) {
      if (cdma_cmd_num[i] > 1) {
#ifdef USING_CMODEL
        cmodel_used_cdma_num++;
#endif
        int port = cdma_ports[i];
        atomic_cdma_port_set_base_ddr(base_idx, base_addr, base_addr_num, port);
        des_id_node.cdma_cmd_id[port] = cdma_cmd_num[i];
        ASSERT((cdma_cmd_addr[i] & 0x7f) == 0x0);
        u64 write_cdma_reg = cdma_cmd_addr[i] >> 7;
        WRITE_REG(CDMA_CSR_REG_DES_ADDR_L32(port), write_cdma_reg, NODECHIP_REG);
        write_cdma_reg = cdma_cmd_addr[i] >> 39;
        WRITE_REG(CDMA_CSR_REG_DES_ADDR_H1(port), write_cdma_reg, NODECHIP_REG);
        write_cdma_reg = READ_REG(CDMA_CSR_REG_DES_RW_ADDR(port));
        write_cdma_reg &= (~(0xff));
        write_cdma_reg |= 0x80;
        WRITE_REG(CDMA_CSR_REG_DES_RW_ADDR(port), write_cdma_reg, NODECHIP_REG);
        write_cdma_reg = READ_REG(CDMA_ENGINE_MAIN_CTRL(port));
        write_cdma_reg = write_cdma_reg | 0x1;
        WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port), write_cdma_reg, NODECHIP_REG);
        (void)(write_cdma_reg);
      }
    }
  }

#ifdef USING_FAKE_DDR_MODE
  CORE_PRINT("des config done    core %d\n", CORE_ID);
#endif
#ifdef USING_CMODEL
  bool using_cmd_arr[ENGINE_END] = {0};
  using_cmd_arr[ENGINE_BD] = (bdc_cmd_num != 0);
  using_cmd_arr[ENGINE_GDMA] = (gdma_cmd_num != 0);
  using_cmd_arr[ENGINE_HAU] = (hau_cmd_addr != 0);
  using_cmd_arr[ENGINE_SDMA] = (sdma_cmd_addr != 0);

  u64 engine_address_arr[ENGINE_END] = {0};
  engine_address_arr[ENGINE_BD] = bdc_cmd_offset;
  engine_address_arr[ENGINE_GDMA] = gdma_cmd_offset;
  engine_address_arr[ENGINE_HAU] = hau_cmd_addr;
  engine_address_arr[ENGINE_SDMA] = sdma_cmd_addr;
  int used_vsdma_num = 0;
  u64 vsdma_address_arr[1] = {0};
  u64 cdma_address_arr[CDMA_API_NUM] = {0};
  for (int i = 0; i < CDMA_API_NUM; ++i) {
    cdma_address_arr[i] = cdma_cmd_addr[i];
  }
  int* cdma_ports = tpu_cdma_get_used_ports();
  cmodel_multi_engine_all_engine(
    engine_address_arr,
    using_cmd_arr,
    get_cur_nodechip_idx(),
    cdma_address_arr,
    vsdma_address_arr,
    cmodel_used_cdma_num,
    used_vsdma_num,
    cdma_ports);
#endif
  des_id_node.bd_cmd_id = bdc_cmd_num;
  des_id_node.gdma_cmd_id = gdma_cmd_num;
  des_id_node.sdma_cmd_id = sdma_cmd_num;

#ifdef DES_POLL_DELAY
  des_start_timestamp = firmware_timer_get_cycle();
#endif

  if (!do_poll) return;

  tpu_poll_descriptor_internal_all_engine();
}

sg_fw_status_t sg_api_launch_cache_multicore_ext_all_engine(
    const void *args,
    unsigned size,
    int work_idx,
    int do_poll)
{
    // Please query sg2260/spec/include/engine_type.h to get the right engine index
    // ENGINE_BD   = 0,
    // ENGINE_GDMA = 1,
    // ENGINE_HAU = 2,
    // ENGINE_SDMA = 3,

    sg_api_launch_cache_multicore_t *api = (sg_api_launch_cache_multicore_t *)args;

#define sprintfcat(var, ...) snprintf(var + strlen(var), sizeof(var) - strlen(var), __VA_ARGS__)
#ifdef USING_LLM_TICK_TOCK_PROFILE
    char id_str[128] = "";
    unsigned id = api->id;
    for (int i = 0; i < 4; ++i)
    {
        unsigned v = (id >> ((3 - i) * 8)) & 0xff;
        sprintfcat(id_str, "%d ", v);
    }
    TP_DEBUG("Execute cached kernel id %s\n", id_str);
    firmware_kernel_tick();
#endif

    tpu_poll();

#ifdef USING_LLM_TICK_TOCK_PROFILE
    TP_DEBUG("Cached kernel %d tpu_poll done.\n", api->id);
#elif defined(DUMP_CACHED_CMD)
    TP_DEBUG("Cached kernel tpu_poll done.\n");
#endif

    sg_api_cmd_descriptor *cmds = api->cmds + work_idx * CACHE_ENGINE_NUM;

#ifdef DUMP_CACHED_CMD
    char line[1024] = "";
    TP_DEBUG("========Core %d cached cmd dump=======\n", work_idx);
    for (int i = 0; i < CACHE_ENGINE_NUM; ++i)
    {
        TP_DEBUG("Engine %d, %d instructions @0x%llx:\n", i, cmds[i].num, cmds[i].addr);
        char *data = (char *)tpu_global_mem_addr(cmds[i].addr);
        unsigned word_num = cmds[i].bytes / 2;
        if (word_num > 100) word_num = 100;
        for (unsigned j = 0; j < word_num; ++j)
        {
            if (j % 16 != 0) sprintfcat(line, " ");
            else if (j != 0)
            {
                TP_DEBUG("%s\n", line);
                line[0] = 0;
            }
            sprintfcat(line, "%04x", ((uint16_t *)data)[j]);
        }
        if (word_num == 0) continue;
        if (word_num * 2 < cmds[i].bytes)
            TP_DEBUG("%s...\n", line);
        else
            TP_DEBUG("%s\n", line);
        line[0] = 0;
    }
#endif

    u64 cdma_cmd_addr[CDMA_API_NUM] = {0};
    int cdma_cmd_num[CDMA_API_NUM] = {0};
    if(tpu_is_last_workitem()) {
      int start_offset = (work_idx == 0) ? 1 : tpu_workitem_num();
      sg_api_cmd_descriptor *start_cmds = api->cmds + start_offset * CACHE_ENGINE_NUM;
      int used_cdma_num = tpu_cdma_get_used_port_num();
      if(used_cdma_num > 0) {
        for(int i = 0; i < used_cdma_num; ++i) {
          cdma_cmd_addr[i] = start_cmds[i].addr;
          cdma_cmd_num[i] = start_cmds[i].num;
        }
      }
    }

    nodechip_multi_fullnet_ext_all_engine(
        cmds[ENGINE_BD].addr,
        cmds[ENGINE_GDMA].addr,
        cmds[ENGINE_BD].num,
        cmds[ENGINE_GDMA].num,
        cmds[ENGINE_SDMA].num,
        cmds[ENGINE_HAU].addr,
        cmds[ENGINE_SDMA].addr,
        cdma_cmd_addr,
        cdma_cmd_num,
        api->base_addr,
        api->addr_num,
        do_poll);

    return SG_FW_SUCCESS;
}

void tpu_poll_descriptor()
{
#ifndef REMOVE_MSG_SYNC
    int msg_id;
#endif

    if (!tpu_is_descriptor_mode())
        return;

#ifdef DUMP_CACHED_CMD
    TP_DEBUG("tpu_poll_descriptor\n");
#endif
    tpu_poll_descriptor_internal_all_engine();
#ifdef DUMP_CACHED_CMD
    TP_DEBUG("tpu_poll_descriptor_internal_all_engine done\n");
    TP_DEBUG("async_descs_remainder.batch: %d\n", async_descs_remainder.batch);
#endif
    if (async_descs_remainder.batch == 0)
        goto done;
    for (int i = 0; i < async_descs_remainder.batch; ++i)
    {
#ifndef REMOVE_MSG_SYNC
        msg_id = tpu_get_schedule_msg_id();

#ifdef NOP_SYNC
        UNUSED(msg_id); tpu_nop();
#else
        tpu_sync_all_with_msg_id(msg_id);
#endif // NOP_SYNC

#endif // REMOVE_MSG_SYNC

        sg_api_launch_cache_multicore_ext_all_engine(
            async_descs_remainder.data + i,
            sizeof(*async_descs_remainder.data),
            0, 1);
    }

done:

#ifndef REMOVE_MSG_SYNC
    msg_id = tpu_get_schedule_msg_id();
#ifdef DUMP_CACHED_CMD
    TP_DEBUG("Barrier after remainder cache tasks, msg id %d.\n", msg_id);
#endif

#ifdef NOP_SYNC
    tpu_nop();
#else
    tpu_sync_all_with_msg_id(msg_id);
#endif // NOP_SYNC

    tpu_poll();
#endif // REMOVE_MSG_SYNC

    async_descs_remainder.batch = 0;
}

sg_fw_status_t sg_api_launch_cache_batch_all_engine(const void *args, unsigned size)
{
    sg_api_cache_launch_batch_t *api = (sg_api_cache_launch_batch_t*)args;
    tpu_sccl_init(api->sccl_args.nranks, api->sccl_args.rank, api->sccl_args.chip_map, api->sccl_args.use_ring);

#ifdef DUMP_CACHED_CMD
    TP_DEBUG("latest batch num: %d\n", api->num);
#endif
    ASSERT(api->num > 0);
    sg_api_launch_cache_multicore_t *launch_param;
    int work_idx = tpu_group_index() * tpu_workitem_num() + tpu_workitem_index();
    int work_offset = work_idx * CACHE_ENGINE_NUM;
    int work_num = tpu_group_num() * tpu_workitem_num();
    int param_stride = sizeof(*api->param) + sizeof(*launch_param->cmds) * CACHE_ENGINE_NUM * work_num
                    + CDMA_API_NUM * sizeof(sg_api_cmd_descriptor);
    (void)param_stride;
    int async_hero_idx = api->num - 1;
    for (int i = api->num - 1; i >= 0; --i)
    {
        launch_param = (sg_api_launch_cache_multicore_t *)((char *)api->param + param_stride * i);

        if (launch_param->cmds[ENGINE_GDMA].num <= 16)
            continue;
        async_hero_idx = i;
        break;
    }

    tpu_poll_descriptor();

    int i;
    char *param;

    param = (char *)(api->param);
    for (i = 0; i <= async_hero_idx; ++i)
    {
        sg_api_launch_cache_multicore_ext_all_engine(param, api->length[i], work_idx, i != async_hero_idx);
        param += api->length[i];

#ifndef REMOVE_MSG_SYNC
        if (i != async_hero_idx)
        {
            int msg_id = tpu_get_schedule_msg_id();

#ifdef NOP_SYNC
            UNUSED(msg_id); tpu_nop();
#else
            tpu_sync_all_with_msg_id(msg_id);
#endif // NOP_SYNC
        }

#endif // REMOVE_MSG_SYNC
    }

    if (async_hero_idx + 1 >= api->num)
        return SG_FW_SUCCESS;

    sg_api_cmd_descriptor *cmds;
    async_descs_remainder.batch = 0;
    for (; i < api->num; ++i)
    {
        launch_param = (sg_api_launch_cache_multicore_t *)param;
        memcpy(
            &async_descs_remainder.data[async_descs_remainder.batch].m,
            launch_param, sizeof(*launch_param));
        cmds = launch_param->cmds + work_offset;
        memcpy(
            async_descs_remainder.data[async_descs_remainder.batch].cmds,
            cmds,
            sizeof(*cmds) * CACHE_ENGINE_NUM);
        if(tpu_is_last_workitem()) {
          memcpy(
              async_descs_remainder.data[async_descs_remainder.batch].cmds + CACHE_ENGINE_NUM,
              cmds + CACHE_ENGINE_NUM,
              (CDMA_API_NUM) * sizeof(sg_api_cmd_descriptor));
        }

        ++async_descs_remainder.batch;
        param += api->length[i];
    }

    return SG_FW_SUCCESS;
}

#ifdef __cplusplus
}
#endif
