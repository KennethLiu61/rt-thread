#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "base_def.h"
#include "tpu_kernel.h"

#define N 4
#define C 32
#define H 32
#define W 64
#define loop 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

void nodechip_fused_linear_test() {
    tpu_initialize();

    dim4 shape = {N, C, H, W};
    float len = (float)N * C * H * W;
    int A_size, B_size, C_size, R_size;

    unsigned long long A_addr = 0;
    unsigned long long B_addr = 0;
    unsigned long long C_addr = 0;
    unsigned long long R_addr = 0;

    u64 lin_start_time = 0ull;
    u64 lin_end_time = 0ull;

    // fp32 lin_mac test
    printf("============ LIN_MAC_FP32 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP32);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP32);
    C_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    C_size = get_tensor_size_onelocal_mem(1, C, 1, 1, C_addr, false, FP32);
    R_addr = ALIGN(C_addr + C_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP32);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, C_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, C_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_scale_bias(R_addr,
                              A_addr,
                              B_addr,
                              C_addr,
                              &shape,
                              DT_FP32);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // fp16 lin_mac test
    printf("============ LIN_MAC_FP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP16);
    C_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    C_size = get_tensor_size_onelocal_mem(1, C, 1, 1, C_addr, false, FP16);
    R_addr = ALIGN(C_addr + C_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, C_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, C_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_scale_bias(R_addr,
                              A_addr,
                              B_addr,
                              C_addr,
                              &shape,
                              DT_FP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // bf16 lin_mac test
    printf("============ LIN_MAC_BFP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, BFP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, BFP16);
    C_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    C_size = get_tensor_size_onelocal_mem(1, C, 1, 1, C_addr, false, BFP16);
    R_addr = ALIGN(C_addr + C_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, BFP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, C_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, C_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_scale_bias(R_addr,
                              A_addr,
                              B_addr,
                              C_addr,
                              &shape,
                              DT_BFP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // fp32 lin_add_sqr test
    printf("============ LIN_ADD_SQR_FP32 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP32);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP32);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP32);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_add_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_FP32);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // fp16 lin_add_sqr test
    printf("============ LIN_ADD_SQR_FP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP16);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_add_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_FP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // bf16 lin_add_sqr test
    printf("============ LIN_ADD_SQR_BFP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, BFP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, BFP16);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, BFP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_add_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_BFP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // fp32 lin_sub_sqr test
    printf("============ LIN_SUB_SQR_FP32 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP32);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP32);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP32);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_sub_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_FP32);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // fp16 lin_sub_sqr test
    printf("============ LIN_SUB_SQR_FP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, FP16);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_add_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_FP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));

    // bf16 lin_sub_sqr test
    printf("============ LIN_SUB_SQR_BFP16 ============\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, BFP16);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(1, C, 1, 1, B_addr, false, BFP16);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, BFP16);
    TPUKERNEL_ASSERT(R_addr + R_size <= (unsigned int)LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    lin_start_time = firmware_timer_get_time_us();
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_add_bias_sqr(R_addr,
                                A_addr,
                                B_addr,
                                &shape,
                                DT_BFP16);
    }
    tpu_poll();
    lin_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops: %.2f\n", (lin_end_time - lin_start_time), len / ((lin_end_time - lin_start_time) / loop * 1e6));
}
