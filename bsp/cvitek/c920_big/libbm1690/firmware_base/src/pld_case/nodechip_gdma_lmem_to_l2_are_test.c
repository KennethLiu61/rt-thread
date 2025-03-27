#include "nodechip_gdma_are_test_utils.h"

void nodechip_gdma_lmem_to_l2_are_test(
    unsigned long long input_addr, unsigned long long output_addr
) {
    tpu_initialize();
    global_addr_t input = (global_addr_t)input_addr;
    global_addr_t output = (global_addr_t)output_addr;
    global_addr_t reduce = input_addr + cnt * sizeof(float);
    global_addr_t l2_reduce = tpu_l2_sram_get_start_addr();
    global_addr_t l2_reduce_buffer = ALIGN(l2_reduce + cnt * sizeof(float), 128);
    local_addr_t local_input = 0;
    dim4 shape = {
        .n = N,
        .c = C,
        .h = H,
        .w = W
    };
    tpu_gdma_cpy_S2S(l2_reduce, reduce, &shape, NULL, NULL, DT_FP32);
    tpu_gdma_cpy_S2L(local_input, input, &shape, NULL, NULL, DT_FP32);
    tpu_gdma_cpy_S2S(l2_reduce_buffer, input, &shape, NULL, NULL, DT_FP32);
    // test gdma tensor reduce global to l2
    // TEST_GDMA_TENSOR_REDUCE(S);
    // test gdma tensor reduce local(l1) to l2
    TEST_GDMA_TENSOR_REDUCE_L(L1);
    // test gdma tensor reduce l2 to l2
    // TEST_GDMA_TENSOR_REDUCE(L2);
    // test gdma lossy compress reduce global/local/l2 to l2
    // compress_S(REDUCE_ADD);
    // compress_L1(REDUCE_MUL);
    // compress_L2(REDUCE_MIN);
    // compress_L1(REDUCE_MAX);
    // compress_S(REDUCE_MIN);
    // compress_L2(REDUCE_ADD);
    // compress_L1(REDUCE_ADD);
    // compress_S(REDUCE_MUL);
    // compress_L1(REDUCE_MIN);
    // compress_L2(REDUCE_MAX);
    // compress_S(REDUCE_MAX);
    // compress_L2(REDUCE_MUL);
    // test gdma lossy decompress reduce global/l2 to l2
    // decompress_S(REDUCE_ADD);
    // decompress_L2(REDUCE_MUL);
    // decompress_S(REDUCE_MIN);
    // decompress_L2(REDUCE_MAX);
    // decompress_S(REDUCE_MAX);
    // decompress_L2(REDUCE_MIN);
    // decompress_S(REDUCE_MUL);
    // decompress_L2(REDUCE_ADD);
    tpu_gdma_cpy_S2S(output, l2_reduce, &shape, NULL, NULL, DT_FP32);
    tpu_poll();
}