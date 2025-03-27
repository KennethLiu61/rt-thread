#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

#define N 1
#define C 64
#define H 16
#define W 16

void nodechip_gdma_tensor_move_fp20_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
    tpu_initialize();

    int GDMA_FORMAT = DT_FP20;
    int num_block = ceiling_func(N * C * H * W, 51);
    u64 l2m1_offset = 0x0;
    u64 l2m2_offset = l2m1_offset + num_block * 128;

    dim4 shape = {N, C, H, W};


    printf("========== GDMA_TensorMoveFP20 ==========\n");
    // test global -> global
    tpu_gdma_cpy_S2S(output_addr, input_addr, &shape, NULL, NULL, GDMA_FORMAT);
    // test global -> l2 -> l2 -> global
    u64 dst_l2_addr = tpu_l2_sram_get_start_addr() + l2m1_offset;
    u64 src_l2_addr = tpu_l2_sram_get_start_addr() + l2m2_offset;
    u64 output_offset1 = num_block * 128;
    // global -> l2
    tpu_gdma_cpy_S2S(dst_l2_addr, input_addr, &shape, NULL, NULL, GDMA_FORMAT);
    // l2 -> l2
    tpu_gdma_cpy_S2S(src_l2_addr, dst_l2_addr, &shape, NULL, NULL, GDMA_FORMAT);
    // l2 -> global
    tpu_gdma_cpy_S2S(output_addr + output_offset1, src_l2_addr, &shape, NULL, NULL, GDMA_FORMAT);
    tpu_sync_all();


    printf("========== SDMA_TensorMoveFP20 ==========\n");
    // test global -> global
    u64 output_offset2 = num_block * 128 * 2;
    tpu_sdma_cpy_S2S(output_addr + output_offset2, input_addr, &shape,  NULL, NULL, GDMA_FORMAT);
    // test global -> l2 -> l2 -> global
    u64 l2m3_offset = l2m2_offset + num_block * 128;
    u64 l2m4_offset = l2m3_offset + num_block * 128;
    u64 dst_l2_addr_1 = tpu_l2_sram_get_start_addr() + l2m3_offset;
    u64 src_l2_addr_1 = tpu_l2_sram_get_start_addr() + l2m4_offset;
    u64 output_offset3 = num_block * 128 * 3;
    // global -> l2
    tpu_sdma_cpy_S2S(dst_l2_addr_1, input_addr, &shape, NULL, NULL, GDMA_FORMAT);
    // l2 -> l2
    tpu_sdma_cpy_S2S(src_l2_addr_1, dst_l2_addr_1, &shape, NULL, NULL, GDMA_FORMAT);
    // l2 -> global
    tpu_sdma_cpy_S2S(output_addr + output_offset3, src_l2_addr_1, &shape, NULL, NULL, GDMA_FORMAT);
    tpu_poll();
}

