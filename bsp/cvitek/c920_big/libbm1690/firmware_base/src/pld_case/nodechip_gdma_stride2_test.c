#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

void nodechip_gdma_stride2_test(unsigned char *api_buf) {

  sg_api_pld_send_instruction_t *api = (sg_api_pld_send_instruction_t *)api_buf;
  TPUKERNEL_ASSERT(api_buf != NULL);
  tpu_initialize();
  int order_type = api->loops;
  TPUKERNEL_ASSERT(order_type > 0);
  unsigned long long input_addr = api->input_global_addr;
  int N = api->N;
  int C = api->C;
  int H = api->H;
  int W = api->W;
  dim4 shape = {N, C, H, W};
  stride_type src_shape[4] = {N, C, H, W};


  int i = 0, loops = 1;
  // N:0, C:1, H:2, W:3
  int order[4];
  switch (order_type) {
    case 1:
        order[0]=0;order[1]=1;order[2]=3;order[3]=2;    // (n,c,w,h)  transpose
        break;
    case 2:
        order[0]=0;order[1]=3;order[2]=2;order[3]=1;    // (n,w,h,c)  transpose
        break;
    case 3:
        order[0]=1;order[1]=0;order[2]=2;order[3]=3;    // (c,n,h,w)  transpose
        break;
    case 4:
        order[0]=3;order[1]=2;order[2]=1;order[3]=0;    // (w,h,c,n)  transpose
        break;
        order[0]=3;order[1]=2;order[2]=1;order[3]=0;    // (h,c,w,n)  transpose
    default:
        order[0]=0;order[1]=1;order[2]=2;order[3]=3;    // (n,c,h,w) not transpose
        break;
  }

  u32 src_n_stride, src_c_stride, src_h_stride;
  src_n_stride = src_shape[1]*src_shape[2]*src_shape[3];
  src_c_stride = src_shape[2]*src_shape[3];
  src_h_stride = src_shape[3];  //w

  dim4 src_stride = {
        src_n_stride,
        src_c_stride,
        src_h_stride,
        1
    };
    stride_type  out_stride[4];
    stride_type tmp_dst_stride[4];  //转置后的shape的 stride
    tmp_dst_stride[3] = 1;
    tmp_dst_stride[2] = src_shape[order[3]];
    tmp_dst_stride[1] = src_shape[order[2]] * src_shape[order[3]];
    tmp_dst_stride[0] = src_shape[order[1]] * src_shape[order[2]] * src_shape[order[3]];

    //src shape 的新stride的值，按着这个shape值可以完成相应(order_type)的转置。
    for(i = 0; i < 4; i++) {
        out_stride[order[i]] = tmp_dst_stride[i];
    }

  dim4 dst_stride = {
        out_stride[0],       //dst_n_stride
        out_stride[1],       //dst_c_stride
        out_stride[2],       //dst_h_stride
        out_stride[3]        //dst_w_stride
    };

  // gdma  s2s  performance test
  u64 s2s_start_time  = 0ULL;
  u64 s2s_end_time = 0ULL;
  float bw = 0;
  int tensor_size= N * C * H * W;
  printf("================= S2S ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d), order(%d, %d, %d, %d)\n",N, C, H, W, order[0],order[1],order[2],order[3]);
   unsigned long long dst_addr = ALIGN((input_addr + tensor_size * sizeof(float)), 64);
  s2s_start_time = firmware_timer_get_time_us();
  for(i = 0; i < loops; i++) {
      tpu_gdma_cpy_S2S(
        dst_addr,
        input_addr,
        &shape,
        &dst_stride,
        &src_stride,
        DT_INT16);
     }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);

  printf("Total transpoer time: %lldus\n", (s2s_end_time - s2s_start_time));
  bw = (float)tensor_size*sizeof(float)*(float)loops/(float)((s2s_end_time-s2s_start_time) * 1e-6);
  printf("Average bandwidth : %.3fMB/s\n", bw/1024/1024);
#if 0
 // gdma  s2l  performance test
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

  unsigned int A_local_addr = 0;
  s2s_start_time = firmware_timer_get_time_us();
  for(int j = 0; j < 10; j++)
  {
    tpu_gdma_cpy_S2L(
        A_local_addr,
        input_addr,
        &src_shape,
        &dst_stride,
        &src_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  unsigned long long total_time = s2s_end_time - s2s_start_time;
  printf("\n================= S2L ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
  printf("gdma src addr = 0x%08llx\n",input_addr);
  printf("gdma S2L transfer size： (10 * single_tensor_size) = %lu(0x%lx) bytes \n", 10 * tensor_size*sizeof(float),tensor_size*sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size*sizeof(float)*10/(float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fMB/s\n", bw/1024/1024);

  // gdma  l2s  performance test
  printf("\n================= L2S ==================\n");
  printf("gdma transfer,shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
  s2s_start_time = firmware_timer_get_time_us();
  for(int i = 0; i < 10; i++)
  {
    tpu_gdma_cpy_L2S(
        output_addr,
        A_local_addr,
        &src_shape,
        NULL,  //&out_stride,
        NULL, //&in_stride,
        DT_FP32);
  }
  tpu_poll();
  s2s_end_time = firmware_timer_get_time_us();
  total_time = s2s_end_time - s2s_start_time;
  printf("gdma src addr = 0x%08x, dst addr = 0x%08llx\n",A_local_addr, output_addr);
  printf("gdma L2S transfer size： (10 * single_tensor_size) = %lu(0x%lx) bytes \n", 10 * tensor_size*sizeof(float),tensor_size*sizeof(float));
  printf("Total send time: %lldus\n", total_time);
  bw = (float)tensor_size*sizeof(float)*10/(float)(total_time * 1e-6);
  printf("Average bandwidth : %.3fMB/s\n", bw/1024/1024);
  printf("poll done\n");
#endif
}
