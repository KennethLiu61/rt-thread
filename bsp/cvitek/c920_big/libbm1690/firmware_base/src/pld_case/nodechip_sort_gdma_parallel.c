#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "atomic_gen_cmd.h"
#include "tpu_kernel.h"

 // 16MB
#define  TOTAL_SIZE     0x1000000

void nodechip_sort_gdma_parallel_test(unsigned long long input_addr,
    unsigned long long output_addr) {
  tpu_initialize();

  int N = 1;
  int C = 16;
  int H = 128;
  int W = 256;
  dim4 shape = {N, C, H, W};

  int gdma_size = N * C * H * W * sizeof(float);    // 2MB
  // gdma s2s test
  u64 start_time  = 0ULL;
  u64 end_time = 0ULL;
  u64 sort_start_time  = 0ULL;
  u64 sort_end_time = 0ULL;
  float bw = 0;
  int tensor_size= N * C * H * W;

  u64 sort_in_addr = input_addr + gdma_size;
  u64 sort_out_addr = output_addr + gdma_size;

  // 2. HAU: sort
  int len = 0x40000;
  int sort_len = len;
  int topk = len;
  data_type_t dtpye = DT_FP32;
  sort_start_time = firmware_timer_get_time_us();
  tpu_hau_sort(
        sort_out_addr,
        sort_in_addr,
        sort_len,
        topk,
        0,
        dtpye   //FP32
        );

  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
  int loop = 800;
  start_time = firmware_timer_get_time_us();
  for(int i = 0; i < loop; i++){
    tpu_gdma_cpy_S2S(
        output_addr,
        input_addr,
        &shape,
        NULL,
        NULL,
        DT_FP32);
   }
  tpu_poll();
  end_time = firmware_timer_get_time_us();
  printf("gdma src addr = 0x%08llx, dst addr = 0x%08llx\n",input_addr, output_addr);
  printf("gdma S2S transfer size： %lu(0x%lx) bytes \n", loop*tensor_size*sizeof(float),tensor_size*sizeof(float));
  printf("Total gdma time: %lldus\n", (end_time - start_time));
  bw = (float)tensor_size*sizeof(float)*loop/(float)((end_time-start_time) * 1e-6);
  printf("Average bandwidth : %.3fMB/s\n", bw/1024/1024);
  tpu_hau_poll();
  sort_end_time = firmware_timer_get_time_us();
  printf("len=%d topk=%d fp32  \n", sort_len, topk);
  printf("Total sort time: %lldus\n", (sort_end_time - sort_start_time));
}

