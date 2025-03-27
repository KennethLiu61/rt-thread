#ifndef USING_FW_DEBUG
#define USING_FW_DEBUG
#endif
#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "firmware_runtime.h"
#include "memmap.h"
#include "firmware_common_inline.h"
#include "tpu_kernel.h"

void nodechip_send_instru_test(unsigned char *api_buf) {
  sg_api_pld_send_instruction_t *api = (sg_api_pld_send_instruction_t *)api_buf;
  TPUKERNEL_ASSERT(api_buf != NULL);
  tpu_initialize();
  int loops = api->loops;
  TPUKERNEL_ASSERT(loops > 0);
  unsigned long long input_addr = api->input_global_addr;
  int N = api->N;
  int C = api->C;
  int H = api->H;
  int W = api->W;
  dim4 shape = {N, C, H, W};
  u64 api_start_time  = 0ULL;
  u64 api_end_time = 0ULL;
  unsigned long long output_addr = api->output_global_addr;
/*
  dim4 in_stride = {
     shape.h * shape.w,
    shape.h * shape.w,
    shape.w,
    1};
  dim4 out_stride = {
    shape.h * shape.w,
    shape.h * shape.w,
    shape.w,
    1};
*/
  int c_per_npu = DIV_UP(C, NPU_NUM);
  int hw_sz = ALIGN(H * W, 16) * sizeof(float);
  int tensor_sz = N * c_per_npu * hw_sz;

  unsigned int A_local_addr = 0;
  unsigned int B_local_addr = tensor_sz;
  unsigned int R_local_addr = tensor_sz * 2;
  api_start_time = firmware_timer_get_time_us();
  for(int j = 0; j < loops; j++)
  {
   // CORE_PRINT("send gdma S2L 1  j=%d\n",j);
    tpu_gdma_cpy_S2L(
        A_local_addr,
        input_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  tpu_gdma_cpy_S2L(
        B_local_addr,
        input_addr+tensor_sz,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  api_end_time = firmware_timer_get_time_us();
  CORE_PRINT("test API ID %d\n", api->test_id);
  CORE_PRINT("Send instructions(tpu_gdma_cpy_S2L) loops: %d \n", loops * 2);
  printf("Total send time: %lldus\n", (api_end_time-api_start_time));
  CORE_PRINT("Total send time: %lldus\n", (api_end_time-api_start_time));
  CORE_PRINT("Average gdma 1 instruction sending time : %8.3fus/\n", ((float)(api_end_time-api_start_time))/((float)loops*2.0f));

  tpu_poll();
  CORE_PRINT("end S2L 2\n");

  TPUKERNEL_ASSERT((unsigned int)tensor_sz * 3 <= (unsigned int)LOCAL_MEM_SIZE);

    // u32 value = READ_REG(BD_ENGINE_BASE_ADDR+80);
    //WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB, (value | 0x1), NODECHIP_REG);
    // CORE_PRINT("open tpu 0x%x\n", value);

  api_start_time = firmware_timer_get_time_us();
  for(int i = 0; i < loops; i++)
  {
    tpu_bdc_fp_add(
        R_local_addr,
        A_local_addr,
        B_local_addr,
        &shape,
        NULL,
        NULL,
        NULL,
        DT_FP32);
  }

  api_end_time = firmware_timer_get_time_us();
  CORE_PRINT("test API ID %d\n", api->test_id);
  CORE_PRINT("Send instructions(tpu_bdc_fp_add) loops: %d \n", loops);
  CORE_PRINT("Total send time: %lldus\n", (api_end_time-api_start_time));
  CORE_PRINT("Average bdc 1 instruction sending time : %8.8fus/\n", ((float)(api_end_time-api_start_time))/((float)loops));

  tpu_gdma_cpy_L2S(
        output_addr,
        R_local_addr,
        &shape,
        NULL,  //&out_stride,
        NULL, //&in_stride,
        DT_FP32);

  tpu_poll();
  CORE_PRINT("poll done\n");
}
