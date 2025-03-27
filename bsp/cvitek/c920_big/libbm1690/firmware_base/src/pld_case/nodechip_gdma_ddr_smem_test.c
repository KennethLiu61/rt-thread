#include "nodechip_pld_test.h"
#include "atomic_gdma_gen_cmd.h"
#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_ddr_smem_test(
    unsigned long long input_addr,
    unsigned long long output_addr) {

  const int size = (STATIC_MEM_SIZE - STATIC_MEM_SHARE_SIZE) / 2;
  u64 st, et;
  CMD_ID_NODE id_node;
  resync_cmd_id(&id_node);
  // ------------------------------------------------------
  // step 1: gdma from ddr to smem
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(input_addr, STATIC_MEM_START_ADDR, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma ddr to smem using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 2: gdma from smem to smem
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(STATIC_MEM_START_ADDR, STATIC_MEM_START_ADDR + size, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma smem to smem using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 3: gdma from smem to ddr
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(STATIC_MEM_START_ADDR + size, output_addr, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma smem to ddr using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 4: gdma from ddr to l2
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(input_addr, tpu_l2_sram_get_start_addr(), 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma ddr to l2 using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 5: gdma from l2 to l2
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(tpu_l2_sram_get_start_addr(), tpu_l2_sram_get_start_addr() + size, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  // ------------------------------------------------------
  // step 6: gdma from l2 to smem
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(tpu_l2_sram_get_start_addr() + size, STATIC_MEM_START_ADDR, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma l2 to smem using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 7: gdma from smem to l2
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(STATIC_MEM_START_ADDR, tpu_l2_sram_get_start_addr(), 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma smem to l2 using time %lldus\n", et - st);
  // ------------------------------------------------------
  // step 8: gdma from l2 to ddr
  // ------------------------------------------------------
  st = firmware_timer_get_time_us();
  general_gdma_gen_cmd(tpu_l2_sram_get_start_addr(), output_addr, 0, size, false, MASTER_THREAD, &id_node);
  poll_all_engine_done(&id_node);
  et = firmware_timer_get_time_us();
  printf("gdma l2 to ddr using time %lldus\n", et - st);
}
