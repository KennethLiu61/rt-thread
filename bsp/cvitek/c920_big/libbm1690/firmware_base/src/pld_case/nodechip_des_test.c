#include "firmware_common_inline.h"
#include "firmware_common_macro.h"
#include "nodechip_pld_test.h"
#ifdef USING_CMODEL
#include "cmodel_multi_thread.h"
#endif

static void mem_check(unsigned long long output_value_addr,
                      unsigned long long output_index_addr) {
  // cmd is int8 sort_per_dim, shape = {2, 7, 1, 3}, sort_dim = 3
  // 1. cast to int32
  // 2. sort per dim, ascending order, outputs are value and index
  // 3. cast to int8
  // input value in sort_dim is {-3, 100, 7}
  int shape[4] = {2, 7, 1, 3};
  const int size = shape[0] * shape[1] * shape[2] * shape[3];
  int8_t ref_value[] = {-3, 7, 100};
  int32_t ref_index[] = {0, 2, 1};

  invalidate_cache(output_value_addr, ALIGN(size * sizeof(int8_t), CACHE_LINE_SIZE));
  invalidate_cache(output_index_addr, ALIGN(size * sizeof(int32_t), CACHE_LINE_SIZE));
  int8_t *value_ptr = (int8_t *)GET_GLOBAL_ADDR(output_value_addr);
  int32_t *index_ptr = (int32_t *)GET_GLOBAL_ADDR(output_index_addr);
  bool cmp_err = 0;
  for (int i = 0; i < size / shape[3]; i++) {
    for (int j = 0; j < shape[3]; j++) {
      if (value_ptr[i * shape[3] + j] != ref_value[j] ||
          index_ptr[i * shape[3] + j] != ref_index[j]) {
        CORE_PRINT("cmp error at idx:(%d, %d), got: (%d, %d), ref: (%d, %d)\n",
                  i, j, value_ptr[i * shape[3] + j],
                  index_ptr[i * shape[3] + j], ref_value[i * shape[3] + j],
                  ref_index[i * shape[3] + j]);
        cmp_err = 1;
        break;
      }
    }
  }
  if (cmp_err == 1) {
    CORE_PRINT("cmp error\n");
  } else {
    CORE_PRINT("cmp sucess\n");
  }
}
static void reset_output(unsigned long long output_value_addr,
                         unsigned long long output_index_addr) {
  // cmd is int8 sort_per_dim, shape = {2, 7, 1, 3}, sort_dim = 3
  // 1. cast to int32
  // 2. sort per dim, ascending order, outputs are value and index
  // 3. cast to int8
  // input value in sort_dim is {-3, 100, 7}
  int shape[4] = {2, 7, 1, 3};
  const int size = shape[0] * shape[1] * shape[2] * shape[3];

  int8_t *value_ptr = (int8_t *)GET_GLOBAL_ADDR(output_value_addr);
  int32_t *index_ptr = (int32_t *)GET_GLOBAL_ADDR(output_index_addr);
  for (int i = 0; i < size; i++) {
    value_ptr[i] = 0xff;
    index_ptr[i] = 0xdeadbeef;
  }
  flush_cache(output_value_addr, ALIGN(size, CACHE_LINE_SIZE));
  flush_cache(output_index_addr, ALIGN(size*sizeof(int32_t), CACHE_LINE_SIZE));
}
static void move_cmd(unsigned long long src_addr, unsigned long long dst_addr,
                     int cmd_size) {
  if (src_addr == dst_addr)
    return;
  invalidate_cache(src_addr, ALIGN(cmd_size, CACHE_LINE_SIZE));
  void *src_ptr = GET_GLOBAL_ADDR(src_addr);
  void *dst_ptr = GET_GLOBAL_ADDR(dst_addr);
  memcpy(dst_ptr, src_ptr, cmd_size);
  flush_cache(dst_addr, ALIGN(cmd_size, CACHE_LINE_SIZE));
}
static void des_test_once(unsigned long long tpu_cmd_addr,
                          unsigned long long gdma_cmd_addr,
                          unsigned long long hau_cmd_addr, int tpu_cmd_num,
                          int gdma_cmd_num, int hau_cmd_num) {
  CMD_ID_NODE id_node;
  resync_cmd_id(&id_node);
  if (tpu_cmd_num != 0) {
    ASSERT((tpu_cmd_addr & 0x7f) == 0x0);
    u64 bdc_cmd_offset_shift = tpu_cmd_addr >> 5;
    u32 write_bdc_reg = (bdc_cmd_offset_shift & 0xffffffff);
    write_bdc_reg = write_bdc_reg | 0x1;
    WRITE_REG((BD_ENGINE_MAIN_CTRL + 0x8), write_bdc_reg, NODECHIP_REG);
    CORE_PRINT(" bdc des config done, cmd_num=%d, core_id=%d\n", tpu_cmd_num,
              CORE_ID);
  }
  if (gdma_cmd_num != 0) {
    ASSERT((gdma_cmd_addr & 0x7f) == 0x0);
    u64 gdma_cmd_offset_shift = gdma_cmd_addr >> 7;
    u32 write_gdma_reg = (gdma_cmd_offset_shift & 0x0fffffff);
    WRITE_REG((GDMA_ENGINE_MAIN_CTRL + 0x4), write_gdma_reg, NODECHIP_REG);

    write_gdma_reg = READ_REG(GDMA_ENGINE_MAIN_CTRL);
    write_gdma_reg = write_gdma_reg | 0x1;
    WRITE_REG(GDMA_ENGINE_MAIN_CTRL, write_gdma_reg, NODECHIP_REG);
    CORE_PRINT(" gdma des config done, cmd_num=%d, core_id=%d\n", gdma_cmd_num,
              CORE_ID);
  }

  if (hau_cmd_num != 0) {
    ASSERT((hau_cmd_addr & 0x3) == 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_PIO_ENABLE, 0);
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_31_2,
                    ((hau_cmd_addr & 0xffffffff) >> 2));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START_ADDR_63_32,
                    (hau_cmd_addr >> 32));
    FW_REG_ID_WRITE(HAU_ENGINE_MAIN_CTRL, SORT_ID_DSCRP_START, 1);
    CORE_PRINT(" hau des config done, cmd_num=%d, core_id=%d\n", hau_cmd_num,
              CORE_ID);
  }

#ifdef USING_CMODEL
  bool using_cmd_arr[3];
  using_cmd_arr[ENGINE_BD] = (tpu_cmd_num != 0);
  using_cmd_arr[ENGINE_GDMA] = (gdma_cmd_num != 0);
  using_cmd_arr[ENGINE_HAU] = (hau_cmd_num != 0);

  u64 engine_address_arr[3];
  engine_address_arr[ENGINE_BD] = tpu_cmd_addr;
  engine_address_arr[ENGINE_GDMA] = gdma_cmd_addr;
  engine_address_arr[ENGINE_HAU] = hau_cmd_addr;

  cmodel_multi_engine(engine_address_arr, using_cmd_arr,
                      get_cur_nodechip_idx());
#endif
  id_node.bd_cmd_id = tpu_cmd_num;
  id_node.gdma_cmd_id = gdma_cmd_num;
  id_node.hau_cmd_id = hau_cmd_num;
  poll_all_engine_done(&id_node);
  CORE_PRINT("poll done.  bdc gdma hau cmd num( %d, %d, %d)\n", tpu_cmd_num,
            gdma_cmd_num, hau_cmd_num);
}
void nodechip_des_test(uint8_t *api_buf) {
  sg_api_pld_des_test_t *api = (sg_api_pld_des_test_t *)api_buf;

  int i = 0;
  while (api->hau_cmd_addr + i * api->addr_stride + api->hau_cmd_size <
         api->end_addr) {
    reset_output(api->output_data_addr, api->output_index_addr);
    move_cmd(api->tpu_cmd_addr, api->tpu_cmd_addr + i * api->addr_stride,
             api->tpu_cmd_size);
    move_cmd(api->gdma_cmd_addr, api->gdma_cmd_addr + i * api->addr_stride,
             api->gdma_cmd_size);
    move_cmd(api->hau_cmd_addr, api->hau_cmd_addr + i * api->addr_stride,
             api->hau_cmd_size);
    des_test_once(api->tpu_cmd_addr + i * api->addr_stride,
                  api->gdma_cmd_addr + i * api->addr_stride,
                  api->hau_cmd_addr + i * api->addr_stride, api->tpu_cmd_num,
                  api->gdma_cmd_num, api->hau_cmd_num);
    mem_check(api->output_data_addr, api->output_index_addr);
    i++;
  }
}