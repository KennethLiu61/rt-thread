#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

void nodechip_gdma_perf_test(unsigned char *api_buf)
{

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
  // fp20 must be 128 bytes aligned
  system_addr_t l2_addr = tpu_l2_sram_get_start_addr();
  system_addr_t l2_addr_2 = ALIGN(tpu_l2_sram_get_start_addr() + N * C * H * W * sizeof(float), 128);
  global_addr_t fp20_test_addr = ALIGN(input_addr, 128);
  global_addr_t fp20_test_addr_2 = ALIGN(fp20_test_addr + N * C * H * W * sizeof(float), 128);

  // gdma  s2s  performance test
  u64 s2s_start_time = 0ULL;
  u64 s2s_end_time = 0ULL;
  float bw = 0;
  int tensor_size = N * C * H * W;
  int fp20_tensor_size = ALIGN(tensor_size, 51);
  int fp20_byte_size = fp20_tensor_size * 128;
  printf("================= S2S ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  unsigned long long dst_addr = ALIGN((input_addr + tensor_size * sizeof(float)), 64);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        dst_addr,
        input_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", input_addr, dst_addr);
  printf("gdma S2S transfer size: %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)tensor_size * sizeof(float) / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= S2L2 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        l2_addr,
        input_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", input_addr, l2_addr);
  printf("gdma S2L2 transfer size: %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)tensor_size * sizeof(float) / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= L22S ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        dst_addr,
        l2_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", l2_addr, dst_addr);
  printf("gdma L22S transfer size: %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)tensor_size * sizeof(float) / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= L22L2 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        l2_addr_2,
        l2_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", l2_addr, l2_addr_2);
  printf("gdma L22L2 transfer size: %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)tensor_size * sizeof(float) / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= S2S FP20 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        fp20_test_addr_2,
        fp20_test_addr,
        &shape,
        NULL,
        NULL,
        DT_FP20);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", fp20_test_addr, fp20_test_addr_2);
  printf("gdma S2S transfer size: %u(0x%x) bytes \n", loops * fp20_byte_size, fp20_byte_size);
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)fp20_byte_size / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= S2L2 FP20 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        l2_addr,
        fp20_test_addr,
        &shape,
        NULL,
        NULL,
        DT_FP20);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", fp20_test_addr, l2_addr);
  printf("gdma S2L2 transfer size: %u(0x%x) bytes \n", loops * fp20_byte_size, fp20_byte_size);
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)fp20_byte_size / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  printf("================= L22S FP20 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for(int j = 0; j < loops; j++) {
    tpu_gdma_cpy_S2S(
        fp20_test_addr_2,
        l2_addr,
        &shape,
        NULL,
        NULL,
        DT_FP20);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", l2_addr, fp20_test_addr_2);
  printf("gdma L22S transfer size: %u(0x%x) bytes \n", loops * fp20_byte_size, fp20_byte_size);
  printf("Total send time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = loops * (float)fp20_byte_size / (float)((s2s_end_time - s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  // gdma  s2l  performance test
  unsigned long long output_addr = api->output_global_addr;
  unsigned int A_local_addr = 0;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        input_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  unsigned long long total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_FP32\n", N, C, H, W);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2s  performance test
  printf("\n================= L2S ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr,
        A_local_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, output_addr);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= L2L2 ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr,
        A_local_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, l2_addr);
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 102.f);

  printf("\n================= L22L ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        A_local_addr,
        l2_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr, A_local_addr);
  printf("gdma L22L transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2l  performance test
  printf("\n================= L2L ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_FP32\n", N, C, H, W);
  int size = DIV_UP(shape.c, NPU_NUM) * shape.h * shape.w * shape.n * sizeof(float);
  if (size <= (LOCAL_MEM_SIZE / 2))
  {
    s2s_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loops; i++)
    {
      tpu_gdma_cpy_L2L(
          0,
          65536,
          &shape,
          NULL, //&out_stride,
          NULL, //&in_stride,
          DT_FP32);
    }
    tpu_poll();
    s2s_end_time = firmware_timer_get_time_us();
    total_time = s2s_end_time - s2s_start_time;
    printf("gdma src addr = 0x%08x, dst addr = 0x%08x\n", 65536, 0);
    printf("gdma L2L transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
    printf("Total send time: %lldus\n", total_time);
    bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
    printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  }
  else
  {
    printf(" L2L :  %d = src_stride.n * src_shape.n * sizeof(float)  > LOCAL_MEM_SIZE/2\n", size);
  }

  printf("\n\n\n\n\n\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u32)16,
        input_addr + (u64)16,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L (16bytes align) ==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", input_addr + (u64)16, A_local_addr + (u32)16);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u64)16,
        l2_addr + (u64)16,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= L22L (16bytes align) ==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr + (u64)16, A_local_addr + (u32)16);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  printf("gdma src addr = 0x%08llx\n", l2_addr);
  printf("gdma L22L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2s  performance test
  printf("\n================= L2S  (16bytes align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr + (u64)16,
        A_local_addr + (u32)16,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)16, output_addr + (u64)16);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= L2L2  (16bytes align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_FP32\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr + (u32)16,
        A_local_addr + (u64)16,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)16, l2_addr + (u64)16);
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u64)4,
        input_addr + (u64)4,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L  (fp32 dtype align)==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", input_addr + (u64)4, A_local_addr + (u32)4);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u64)4,
        l2_addr + (u64)4,
        &shape,
        NULL,
        NULL,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= L22L  (fp32 dtype align)==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr + (u64)4, A_local_addr + (u32)4);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  printf("gdma src addr = 0x%08llx\n", l2_addr);
  printf("gdma L22L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2s  performance test
  printf("\n================= L2S  (fp32 dtype align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr + (u64)4,
        A_local_addr + (u32)4,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)4, output_addr + (u64)4);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= L2L2  (fp32 dtype align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_FP32\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr + (u64)4,
        A_local_addr + (u32)4,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)4, l2_addr + (u64)4);
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 4;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u32)1,
        input_addr + (u64)1,
        &shape,
        NULL,
        NULL,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L (int8 dtype align) a ==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", input_addr + (u64)1, A_local_addr + (u32)1);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H * 4, W);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 4;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u32)1,
        l2_addr + (u64)1,
        &shape,
        NULL,
        NULL,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= L22L (int8 dtype align) a ==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr + (u64)1, A_local_addr + (u32)1);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H * 4, W);
  printf("gdma src addr = 0x%08llx\n", l2_addr);
  printf("gdma L22L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 4;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        input_addr + (u64)1,
        &shape,
        NULL,
        NULL,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L (int8 dtype align) b==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", input_addr + (u64)1, A_local_addr);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H * 4, W);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 4;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        l2_addr + (u64)1,
        &shape,
        NULL,
        NULL,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= L22L (int8 dtype align) b==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr + (u64)1, A_local_addr);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H * 4, W);
  printf("gdma src addr = 0x%08llx\n", l2_addr);
  printf("gdma L22L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2s  performance test
  printf("\n================= L2S (int8 dtype align) ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr + (u64)1,
        A_local_addr + (u32)1,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)1, output_addr + (u64)1);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= L2L2 (int8 dtype align) ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT8\n", N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr + (u64)1,
        A_local_addr + (u32)1,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT8);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 2;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u32)2,
        input_addr + (u64)2,
        &shape,
        NULL,
        NULL,
        DT_INT16);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L  (int16 dtype align)==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", input_addr + (u64)2, A_local_addr + (u32)2);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx\n", input_addr);
  printf("gdma S2L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  shape.h = H * 2;
  s2s_start_time = firmware_timer_get_time_us();
  for (int j = 0; j < loops; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr + (u32)2,
        l2_addr + (u64)2,
        &shape,
        NULL,
        NULL,
        DT_INT16);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("\n================= L22L  (int16 dtype align)==================\n");
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08x\n", l2_addr + (u64)2, A_local_addr + (u32)2);
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx\n", l2_addr);
  printf("gdma L22L transfer:(loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);


  // gdma  l2s  performance test
  printf("\n================= L2S  (int16 dtype align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n", shape.n, shape.c, shape.h, shape.w);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr + (u64)2,
        A_local_addr + (u32)2,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT16);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr + (u32)2, output_addr + (u64)2);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= L2L2  (int16 dtype align)==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT16\n", shape.n, shape.c, shape.h, shape.w);
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr + (u64)2,
        A_local_addr + (u32)2,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT16);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  // gdma  l2s  performance test
  shape.n = N;
  shape.c = C;
  shape.h = H + (H % 16 == 0 ? 1 : 0);
  shape.w = W + (W % 16 == 0 ? 1 : 0);
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  printf("\n\n\n\n\n\n================= L2S + stride ==================\n");
  dim4 l_stride, g_stride;
  tpu_continuous_stride(&g_stride, &shape);
  tpu_aligned_stride(&l_stride, 0, &shape, DT_FP32);

  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr,
        A_local_addr,
        &shape,
        &g_stride, //&out_stride,
        &l_stride, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, output_addr);
  printf("gdma   shape(N,C,H,W)=(%d, %d, %d, %d), l_stride=(%d,%d,%d,%d), g_stride(%d,%d,%d,%d) DT_FP32\n",
         shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w,
         g_stride.n, g_stride.c, g_stride.h, g_stride.w);

  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  printf("\n================= L2L2 + stride ==================\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_L2S(
        l2_addr,
        A_local_addr,
        &shape,
        &g_stride, //&out_stride,
        &l_stride, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, l2_addr);
  printf("gdma   shape(N,C,H,W)=(%d, %d, %d, %d), l_stride=(%d,%d,%d,%d), g_stride(%d,%d,%d,%d) DT_FP32\n",
         shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w,
         g_stride.n, g_stride.c, g_stride.h, g_stride.w);
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= S2L + stride ==================\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        output_addr,
        &shape,
        &l_stride, //&out_stride,
        &g_stride, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, output_addr);
  printf("gdma   shape(N,C,H,W)=(%d, %d, %d, %d), l_stride=(%d,%d,%d,%d), g_stride(%d,%d,%d,%d) DT_FP32\n",
         shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w,
         g_stride.n, g_stride.c, g_stride.h, g_stride.w);

  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n\n\n\n\n\n");
  printf("\n================= L22L + stride ==================\n");
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        l2_addr,
        &shape,
        &l_stride, //&out_stride,
        &g_stride, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, l2_addr);
  printf("gdma   shape(N,C,H,W)=(%d, %d, %d, %d), l_stride=(%d,%d,%d,%d), g_stride(%d,%d,%d,%d) DT_FP32\n",
         shape.n, shape.c, shape.h, shape.w, l_stride.n, l_stride.c, l_stride.h, l_stride.w,
         g_stride.n, g_stride.c, g_stride.h, g_stride.w);
  printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n\n\n\n\n\n", bw / 1024 / 1024 / 1024.f);

  dim4 stride;
  // gdma  nc_trans  performance test
  printf("\n================= nc_trans   S2L ==================\n");
  shape.n = C;
  shape.c = N;
  shape.h = H;
  shape.w = W;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  tpu_aligned_stride(&stride, 0, &shape, DT_INT32);
  if (stride.n * shape.n * (int)sizeof(int) <= LOCAL_MEM_SIZE)
  {
    s2s_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loops; i++)
    {
      tpu_gdma_cpy_nc_trans_S2L(
          A_local_addr,
          output_addr,
          &shape,
          NULL, //&out_stride,
          NULL, //&in_stride,
          DT_INT32);
    }
    tpu_poll();
    s2s_end_time = firmware_timer_get_time_us();
    total_time = s2s_end_time - s2s_start_time;
    printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
           shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
    printf("gdma  src addr = 0x%08llx,dst addr = 0x%08x\n", output_addr, A_local_addr);
    printf("gdma S2L transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
    printf("Total send time: %lldus\n", total_time);
    bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
    printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  }
  else
  {
    printf("case skipped, nc_trans S2L :  %d = dst_stride.n * dst_shape.n *sizeof(int) > LOCAL_MEM_SIZE\n", stride.n * shape.n* (int)sizeof(int));
  }

  printf("\n================= nc_trans   S22L ==================\n");
  shape.n = C;
  shape.c = N;
  shape.h = H;
  shape.w = W;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  tpu_aligned_stride(&stride, 0, &shape, DT_INT32);
  if(stride.n * shape.n * (int)sizeof(int)<=LOCAL_MEM_SIZE)
  {
    s2s_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loops; i++)
    {
      tpu_gdma_cpy_nc_trans_S2L(
          A_local_addr,
          l2_addr,
          &shape,
          NULL, //&out_stride,
          NULL, //&in_stride,
          DT_INT32);
    }
    tpu_poll();
    s2s_end_time = firmware_timer_get_time_us();
    total_time = s2s_end_time - s2s_start_time;
    printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
           shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
    printf("gdma  src addr = 0x%08llx,dst addr = 0x%08x\n", l2_addr, A_local_addr);
    printf("gdma L22L transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
    printf("Total send time: %lldus\n", total_time);
    bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
    printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  } else {
    printf("case skipped, nc_trans S2L :  %d = dst_stride.n * dst_shape.n *sizeof(int) > LOCAL_MEM_SIZE\n", stride.n * shape.n* (int)sizeof(int));
  }

  // gdma  nc_trans  performance test
  printf("\n================= nc_trans   L2S ==================\n");
  shape.n = N;
  shape.c = C;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  dim4 trans_shape = shape;
  trans_shape.n = shape.c;
  trans_shape.c = shape.n;
  tpu_aligned_stride(&stride, 0, &trans_shape, DT_INT32);
  if (stride.n * trans_shape.n * (int)sizeof(int) <= LOCAL_MEM_SIZE)
  {
    s2s_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loops; i++)
    {
      tpu_gdma_cpy_nc_trans_L2S(
          output_addr,
          A_local_addr,
          &shape,
          NULL, //&out_stride,
          NULL, //&in_stride,
          DT_INT32);
    }
    tpu_poll();
    s2s_end_time = firmware_timer_get_time_us();
    total_time = s2s_end_time - s2s_start_time;
    printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
           shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
    printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, output_addr);
    printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
    printf("Total send time: %lldus\n", total_time);
    bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
    printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  }
  else
  {
    printf("case skipped, nc_trans L2S :  %d = src_stride.n * src_shape.n *sizeof(int)  > LOCAL_MEM_SIZE\n", stride.n * shape.n* (int)sizeof(int));
  }

  printf("\n================= nc_trans   L2L2 ==================\n");
  shape.n = N;
  shape.c = C;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  trans_shape.n = shape.c;
  trans_shape.c = shape.n;
  tpu_aligned_stride(&stride, 0, &trans_shape, DT_INT32);
  if (stride.n * trans_shape.n * (int)sizeof(int) <= LOCAL_MEM_SIZE)
  {
    s2s_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loops; i++)
    {
      tpu_gdma_cpy_nc_trans_L2S(
          l2_addr,
          A_local_addr,
          &shape,
          NULL, //&out_stride,
          NULL, //&in_stride,
          DT_INT32);
    }
    tpu_poll();
    s2s_end_time = firmware_timer_get_time_us();
    total_time = s2s_end_time - s2s_start_time;
    printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
           shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
    printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n", A_local_addr, l2_addr);
    printf("gdma L2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
    printf("Total send time: %lldus\n", total_time);
    bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
    printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
  }
  else
  {
    printf("case skipped, nc_trans L2L2 :  %d = src_stride.n * src_shape.n *sizeof(int)  > LOCAL_MEM_SIZE\n", stride.n * shape.n* (int)sizeof(int));
  }

  // gdma  nc_trans  performance test
  printf("\n================= nc_trans   L2L ==================\n");
  shape.n = N;
  shape.c = C;
  tpu_aligned_stride(&stride, 0, &shape, DT_INT32);
  if (stride.n * shape.n * (int)sizeof(int) <= LOCAL_MEM_SIZE / 2)
  {
    shape.n = C;
    shape.c = N;
    tensor_size = shape.n * shape.c * shape.h * shape.w;
    tpu_aligned_stride(&stride, 0, &shape, DT_INT32);
    if (stride.n * shape.n *(int)sizeof(int) <= (LOCAL_MEM_SIZE / 2))
    {

      s2s_start_time = firmware_timer_get_time_us();
      for (int i = 0; i < loops; i++)
      {
        tpu_gdma_cpy_nc_trans_L2L(
            0,
            65536,
            &shape,
            NULL, //&out_stride,
            NULL, //&in_stride,
            DT_INT32);
      }
      tpu_poll();
      s2s_end_time = firmware_timer_get_time_us();
      total_time = s2s_end_time - s2s_start_time;
      printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
            shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
      printf("gdma src addr = 0x%08x, dst addr = 0x%08x\n", 65536, 0);
      printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
      printf("Total send time: %lldus\n", total_time);
      bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
      printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);
    }
    else
    {
      printf("case skipped, nc_trans L2L :  %d = dst_stride.n * dst_shape.n *sizeof(int) > LOCAL_MEM_SIZE/2\n", stride.n * shape.n*(int)sizeof(int));
    }
  }
  else
  {
    printf("case skipped, nc_trans L2L :  %d = src_stride.n * src_shape.n *sizeof(int)  > LOCAL_MEM_SIZE/2\n", stride.n * shape.n);
  }

  // gdma  nc_trans  performance test
  printf("\n================= nc_trans   S2S ==================\n");
  shape.n = C;
  shape.c = N;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_nc_trans_S2S(
        output_addr,
        input_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
        shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", input_addr, output_addr);
  printf("gdma L2S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= nc_trans   S2L2 ==================\n");
  shape.n = C;
  shape.c = N;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  s2s_start_time = firmware_timer_get_time_us();
  for (int i = 0; i < loops; i++)
  {
    tpu_gdma_cpy_nc_trans_S2S(
        l2_addr,
        input_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
        shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", input_addr, l2_addr);
  printf("gdma S2L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= nc_trans   L22S ==================\n");
  shape.n = N;
  shape.c = C;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  s2s_start_time = firmware_timer_get_time_us();
  {
    tpu_gdma_cpy_nc_trans_S2S(
        output_addr,
        l2_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
        shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", l2_addr, output_addr);
  printf("gdma L22S transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("\n================= nc_trans   L22L2 ==================\n");
  shape.n = N;
  shape.c = C;
  tensor_size = shape.n * shape.c * shape.h * shape.w;
  s2s_start_time = firmware_time_get_ns();
  {
    tpu_gdma_cpy_nc_trans_S2S(
        l2_addr_2,
        l2_addr,
        &shape,
        NULL, //&out_stride,
        NULL, //&in_stride,
        DT_INT32);
  }
  tpu_poll();
  s2s_end_time = firmware_time_get_ns();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma transfer,src shape(N,C,H,W)=(%d, %d, %d, %d),dst shape(N,C,H,W)=(%d, %d, %d, %d), DT_INT32\n",
        shape.c, shape.n, shape.h, shape.w, shape.n, shape.c, shape.h, shape.w);
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n", l2_addr, l2_addr_2);
  printf("gdma L22L2 transfer: (loops * single_tensor_size) = %lu(0x%lx) bytes \n", loops * tensor_size * sizeof(float), tensor_size * sizeof(float));
  printf("Total send time: %lldns\n", total_time);
  bw = (float)tensor_size * sizeof(float) * loops / (float)(total_time * 1e-9);
  printf("Average bandwidth : %.3fGB/s\n", bw / 1024 / 1024 / 1024.f);

  printf("poll done\n");
}