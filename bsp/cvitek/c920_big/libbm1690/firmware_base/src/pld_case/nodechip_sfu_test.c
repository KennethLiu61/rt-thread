#include "atomic_gen_cmd.h"
#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

#define N 4
#define C 32
#define H 32
#define W 64
#define loop 10
#define TAYLOR_COEFF_LEN 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

#define GEN_ADDR(prec0, prec1, is_taylor)                                               \
    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, prec0);             \
    if (is_taylor) {                                                                    \
        coeff_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);                           \
        coeff_size = get_tensor_size_onelocal_mem(1, NPU_NUM, 1, 32,                    \
                                                  coeff_addr, true, prec0);             \
        R_addr = ALIGN(coeff_addr + coeff_size, LOCAL_BANK_SIZE);                       \
    } else {                                                                            \
        R_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);                               \
    }                                                                                   \
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, prec1);             \
    TPUKERNEL_ASSERT(R_addr + R_size <= LOCAL_MEM_SIZE);                                \
    if (is_taylor)                                                                      \
        printf("A_addr = 0x%08llx, coeff_addr = 0x%08llx, R_addr = 0x%08llx\n",         \
                A_addr, coeff_addr, R_addr);                                            \
    else                                                                                \
        printf("A_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, R_addr);

#define TEST_BEGIN(prec0, prec1, is_taylor)                                             \
    printf("loop: %d\n", loop);                                                         \
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);                             \
    GEN_ADDR(prec0, prec1, is_taylor)                                                   \
    sfu_start_time = firmware_timer_get_time_us();

#define TEST_END                                                                        \
    tpu_poll();                                                                         \
    sfu_end_time = firmware_timer_get_time_us();                                        \
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),          \
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));

void nodechip_sfu_test() {
    tpu_initialize();

    dim4 shape = {N, C, H, W};
    int A_size, R_size, coeff_size;

    unsigned long long A_addr = 0;
    unsigned long long R_addr = 0;
    unsigned long long coeff_addr = 0;

    u64 sfu_start_time = 0ull;
    u64 sfu_end_time = 0ull;

    // test rsqrt
    data_type_t rsqrt_dtype[] = {DT_FP32, DT_FP16, DT_BFP16};
    const char *rsqrt_str[] = {"FP32", "FP16", "BF16"};
    for (size_t i = 0; i < sizeof(rsqrt_dtype) / sizeof(rsqrt_dtype[0]); i++) {
        printf("============ SFU_RSQRT_%s ============\n", rsqrt_str[i]);
        TEST_BEGIN(rsqrt_dtype[i] >> 1, rsqrt_dtype[i] >> 1, false)
        for (int iterx = 0; iterx < loop; iterx++) {
            tpu_bdc_fp_rsqrt(
                R_addr,
                A_addr,
                &shape,
                rsqrt_dtype[i]);
        }
        TEST_END
    }

    // test normal
    printf("============== SFU_NORMAL ==============\n");
    printf("src: FP32, dst: FP32\n");
    TEST_BEGIN(FP32, FP32, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_FP32,
                                 DT_FP32);
    }
    TEST_END
    printf("---------------------------\n");
    printf("src: FP32, dst: INT32\n");
    TEST_BEGIN(FP32, INT32, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_INT32,
                                 DT_FP32);
    }
    TEST_END
    printf("---------------------------\n");
    printf("src: FP16, dst: FP16\n");
    TEST_BEGIN(FP16, FP16, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_FP16,
                                 DT_FP16);
    }
    TEST_END
    printf("---------------------------\n");
    printf("src: FP16, dst: INT16\n");
    TEST_BEGIN(FP16, INT16, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_INT16,
                                 DT_FP16);
    }
    TEST_END
    printf("---------------------------\n");
    printf("src: BFP16, dst: BFP16\n");
    TEST_BEGIN(BFP16, BFP16, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_BFP16,
                                 DT_BFP16);
    }
    TEST_END
    printf("---------------------------\n");
    printf("src: BFP16, dst: INT16\n");
    TEST_BEGIN(BFP16, INT16, false)
    for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_exponent_part(R_addr,
                                 A_addr,
                                 &shape,
                                 DT_INT16,
                                 DT_BFP16);
    }
    TEST_END

    // test taylor
    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);
    printf("============== SFU_TAYLOR ==============\n");
    printf("dtype: FP32");
    TEST_BEGIN(FP32, FP32, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR,
                           coeff_addr,
                           FP32,
                           FP32,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));

    printf("---------------------------\n");
    printf("dtype: FP16");
    TEST_BEGIN(FP16, FP16, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR,
                           coeff_addr,
                           FP16,
                           FP16,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));

    printf("---------------------------\n");
    printf("dtype: BFP16");
    TEST_BEGIN(BFP16, BFP16, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR,
                           coeff_addr,
                           BFP16,
                           BFP16,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));

    // test taylor 4x
    printf("============= SFU_TAYLOR_4x ============\n");
    printf("dtype: FP32");
    TEST_BEGIN(FP32, FP32, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR_4X,
                           coeff_addr + tpu_npu_index(R_addr) * LOCAL_MEM_SIZE,
                           FP32,
                           FP32,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));
    printf("---------------------------\n");
    printf("dtype: FP16");
    TEST_BEGIN(FP16, FP16, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR_4X,
                           coeff_addr + tpu_npu_index(R_addr) * LOCAL_MEM_SIZE,
                           FP16,
                           FP16,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));
    printf("---------------------------\n");
    printf("dtype: BFP16");
    TEST_BEGIN(BFP16, BFP16, true)
    for (int i = 0; i < loop; i++) {
        atomic_sfu_gen_cmd(A_addr,
                           R_addr,
                           N, C, H, W,
                           TAYLOR_COEFF_LEN,
                           SFU_TAYLOR_4X,
                           coeff_addr + tpu_npu_index(R_addr) * LOCAL_MEM_SIZE,
                           BFP16,
                           BFP16,
                           MASTER_THREAD,
                           &id_node);
    }
    poll_all_engine_done(&id_node);
    sfu_end_time = firmware_timer_get_time_us();
    printf("Total time: %lldus, Tops:%.6f\n", (sfu_end_time - sfu_start_time),
           (float)(N * C * H * W) / ((sfu_end_time - sfu_start_time) / (float)loop * 1e6));
}
