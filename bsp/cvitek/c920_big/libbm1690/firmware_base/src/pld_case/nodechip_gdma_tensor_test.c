#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

#define N 4
#define C 64
#define H 4
#define W 4
#define loop 10

void nodechip_gdma_tensor_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < loop; ++t) {
#define END(info, direction, src_format, count, type, reduce_op) \
    }                   \
    poll_all_engine_done(&id_node);     \
    end_time = firmware_timer_get_time_us();    \
    printf("test %d times %s %s %s src_format %s\n", loop, #info, reduce_op, #direction, src_format);            \
    printf("Total %s time: %lldus, BW: %.4fGB/s\n\n", #info, (end_time - start_time), \
            (float)loop * count * get_gdma_format_type_len(type) * 1e6 / (end_time - start_time) / (powf(1024.f, 3)));

    int GDMA_FORMAT[8] = {GDMA_INT8, GDMA_INT16, GDMA_INT32, GDMA_FP8_E4M3, GDMA_FP8_E5M2, GDMA_FP16, GDMA_BF16, GDMA_FP32};
    int eu_num[8] = {64, 32, 16, 64, 64, 32, 32, 16};
    char * gdma_src_format[8] = {"GDMA_INT8",
                            "GDMA_INT16",
                            "GDMA_INT32",
                            "GDMA_FP8_E4M3",
                            "GDMA_FP8_E5M2",
                            "GDMA_FP16",
                            "GDMA_BF16",
                            "GDMA_FP32"};
    u64 start_time = 0ull;
    u64 end_time = 0ull;
    const int l_offset = 0;
    const int g1_offset = 0x1000000;
    const u64 g2_offset = 0x2000000;
    u64 src_addr = GLOBAL_MEM_START_ADDR + g1_offset;
    u64 dst_addr = GLOBAL_MEM_START_ADDR + g2_offset;

    int npu_idx = NPU_NUM >> 2;
    stride_type src_g_W_stride = 1;
    stride_type src_g_H_stride = W;
    stride_type src_g_C_stride = H * W;
    stride_type src_g_N_stride = C * H * W;
 
    stride_type dst_g_W_stride = 1;
    stride_type dst_g_H_stride = src_g_H_stride * 2;
    stride_type dst_g_C_stride = src_g_C_stride * 2;
    stride_type dst_g_N_stride = src_g_N_stride * 2;

    CMD_ID_NODE id_node;
    resync_cmd_id(&id_node);

    // test TensorStrideMove
    printf("========== GDMA_TensorStrideMove ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    stride_type dst_l_W_stride = 1, dst_l_H_stride = W;
    stride_type dst_l_C_stride, dst_l_N_stride;
    int count = N * C * H * W;
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    for (int i = 0; i < 8; i++) {
        dst_l_C_stride = ALIGN(W * H, eu_num[i]);
        dst_l_N_stride = dst_l_C_stride * ceiling_func(C + npu_idx, NPU_NUM);
        BEGIN()
        tensor_stride_move_gen_cmd(
            l_offset,
            npu_idx,
            src_addr,
            N,
            C,
            H,
            W,
            src_g_N_stride,
            src_g_C_stride,
            src_g_H_stride,
            src_g_W_stride,
            dst_l_N_stride,
            dst_l_C_stride,
            dst_l_H_stride,
            dst_l_W_stride,
            GDMA_FORMAT[i],
            GDMA_S2L,
            1,  // N/C transpose
            &id_node);
        END(TensorStrideMove, S2L, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }

    // test TensorCompactMove
    printf("========== GDMA_TensorCompactMove ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    for (int i = 0; i < 8; i++) {
        BEGIN()
        tensor_compact_move_gen_cmd(
            l_offset,
            npu_idx,
            src_addr,
            N,
            C,
            H,
            W,
            GDMA_FORMAT[i],
            GDMA_S2L,
            1,  // N/C transpose
            0, //thread_id,
            &id_node);
        END(TensorCompactMove, S2S, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }

    // test TensorGeneralMove
    printf("========== GDMA_TensorGeneralMove ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    for (int i = 0; i < 8; i++) {
        BEGIN()
        tensor_general_move_gen_cmd(
            src_addr, //local_addr or global_addr
            0, //use only from local_mem
            N,
            C,
            H,
            W,
            src_g_N_stride,
            src_g_C_stride,
            src_g_H_stride,
            src_g_W_stride,
            GDMA_FORMAT[i],
            dst_addr, //local_addr or global_addr
            0, //use only to local_mem
            N,
            C,
            H,
            W,
            dst_g_N_stride,
            dst_g_C_stride,
            dst_g_H_stride,
            dst_g_W_stride,
            GDMA_S2S,
            1,  // N/C transpose
            &id_node);
        END(TensorGeneralMove, S2S, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }

    // test TensorGeneralMove ARE
    printf("========== GDMA_TensorGeneralMove_ARE ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    char * all_reduce_opcode[5] = {"ALL_REDUCE_NOP",
                                   "ALL_REDUCE_MUL",
                                   "ALL_REDUCE_MAX",
                                   "ALL_REDUCE_MIN",
                                   "ALL_REDUCE_ADD"};
    dst_addr = tpu_l2_sram_get_start_addr();
    printf("src_addr = 0x%08llx, dst_addr = 0x%08llx\n",
            input_addr, output_addr);
    //can test for 8 types of gdma format
    for (int i = 0; i < 5; i++) {
        BEGIN()
        tensor_general_move_reduce_gen_cmd(
            src_addr, 0,
            N, C, H, W,
            src_g_N_stride, src_g_C_stride, src_g_H_stride, src_g_W_stride,
            GDMA_FP32,
            dst_addr, 0,
            N, C, H, W,
            dst_g_N_stride, dst_g_C_stride, dst_g_H_stride, dst_g_W_stride,
            GDMA_S2S,
            0,  // N/C transpose
            GDMA_ARE_PSUM_RW,
            i,
            MASTER_THREAD,
            &id_node);
        END(TensorGeneralMove_ARE, S2S, "GDMA_FP32", count, GDMA_FP32, all_reduce_opcode[i])
    }

    // test LocalFillConst
    printf("========== GDMA_LocalFillConst ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    int const_val = 0x12345678;
    const stride_type W_stride = 1;
    const stride_type H_stride = W * W_stride;
    const stride_type C_stride = H * H_stride;
    const stride_type l_N_stride = ceiling_func(C, NPU_NUM) * C_stride;
    for (int i = 0; i < 8; i++) {
        BEGIN()
            fill_constant_gen_local_cmd_stride(
            l_offset, npu_idx,
            &const_val,
            GDMA_FORMAT[i],
            N, C, H, W,
            l_N_stride, C_stride, H_stride, W_stride,
            1, 0, MASTER_THREAD,
            &id_node);
        END(LocalFillConst, 2L, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }

    // test GlobalFillConst
    printf("========== GDMA_GlobalFillConst ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    dst_addr = GLOBAL_MEM_START_ADDR + g1_offset;
    for (int i = 0; i < 8; i++) {
        BEGIN()
        fill_constant_gen_global_cmd_stride(
            dst_addr,
            &const_val,
            GDMA_FORMAT[i],
            N, C, H, W,
            dst_g_N_stride, dst_g_C_stride, dst_g_H_stride, dst_g_W_stride,
            1, MASTER_THREAD,
            &id_node);
        END(GlobalFillConst, S2S, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }

    // test TensorBroadcast
    printf("========== GDMA_TensorBroadcast ==========\n");
    printf("loop: %d\n", loop);
    printf("shape(N,C,H,W)=(%d, %d, %d, %d)\n",N, C, H, W);
    // to satisfy dst_C + dst_local_idx <= NPU_NUM
    npu_idx = NPU_NUM >> 2;
    int broadcast_C = NPU_NUM >> 1;
    for (int i = 0; i < 8; i++) {
        BEGIN()
        tensor_broadcast_move_gen_cmd(
        src_addr, 0,
        l_offset, npu_idx,
        N, H, W, broadcast_C,
        0, 0, 0, 0,
        GDMA_FORMAT[i],
        0,//stride_enable
        GDMA_S2L,
        &id_node);
        END(TensorBroadcast, S2L, gdma_src_format[i], count, GDMA_FORMAT[i], "")
    }
}

