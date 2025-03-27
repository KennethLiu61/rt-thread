#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_gather_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
    tpu_initialize();

    local_addr_t input_local_addr = 0;
    local_addr_t index_local_addr = LOCAL_MEM_SIZE / 4;
    local_addr_t output_local_addr = LOCAL_MEM_SIZE / 2;
    data_type_t dtype = DT_FP32;
    scalar_t C_value = {.u32 = 0x10};
    u32 start_pos = 0x10;

    // big test case for S2S with h > 65535
    int C = 6;
    int src_h = 65536;
    int src_w = 32;
    int index_h = 65600;
    int src_c_is1 = 0;
    int idx_c_is1 = 1;
    u64 src_size = (src_c_is1 ? 1 : C) * src_h * src_w; // [1, C or 1, src_h, src_w]
    u64 dst_size = ((src_c_is1 && idx_c_is1) ? 1 : C) * index_h * src_w;
    dim4 output_shape = {.n = 1, .c = (src_c_is1 && idx_c_is1) ? 1 : C, .h = index_h, .w = src_w};
    global_addr_t index_addr = input_addr + src_size * tpu_data_type_size(DT_FP32);

    // small test case for S2L & L2S & L2L
    int C_s = 4;
    int src_h_s = 16;
    int src_w_s = 4;
    int index_h_s = 32;
    int src_c_is1_s = 0;
    int idx_c_is1_s = 0;
    int dst_size_s = ((src_c_is1_s && idx_c_is1_s) ? 1 : C_s) * index_h_s * src_w_s;
    dim4 input_shape_s = {.n = 1, .c = C_s, .h = src_h_s, .w = src_w_s};
    dim4 index_shape = {.n = 1, .c = C_s, .h = index_h_s, .w = 1};
    dim4 output_shape_s = {.n = 1, .c = (src_c_is1_s && idx_c_is1_s) ? 1 : C_s, .h = index_h_s, .w = src_w_s};

    // test S2S
    printf("\nS2S: C=%d, src_h=%d, src_w=%d, idx_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
            C, src_h, src_w, index_h, src_c_is1, idx_c_is1);
    tpu_gdma_h_gather_S2S_ext(output_addr,
                              input_addr,
                              index_addr,
                              0,
                              C_value,
                              &output_shape,
                              src_h,
                              start_pos,
                              NULL, NULL, NULL, dtype);

    // test S2L
    printf("\nS2L: C=%d, src_h=%d, src_w=%d, idx_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
            C_s, src_h_s, src_w_s, index_h_s, src_c_is1_s, idx_c_is1_s);
    u64 output_s2l_offset = dst_size * tpu_data_type_size(dtype);
    tpu_gdma_h_gather_S2L(output_local_addr,
                          input_addr,
                          index_addr,
                          0,
                          C_value,
                          &output_shape_s,
                          src_h_s,
                          NULL, NULL, NULL, dtype);
    tpu_gdma_cpy_L2S(output_addr + output_s2l_offset,
                     output_local_addr,
                     &output_shape_s,
                     NULL,
                     NULL,
                     dtype);

    // test L2S
    printf("\nL2S: C=%d, src_h=%d, src_w=%d, idx_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
            C_s, src_h_s, src_w_s, index_h_s, src_c_is1_s, idx_c_is1_s);
    u64 output_l2s_offset = (dst_size + dst_size_s) * tpu_data_type_size(dtype);
    tpu_gdma_cpy_S2L(input_local_addr,
                     input_addr,
                     &input_shape_s,
                     NULL,
                     NULL,
                     dtype);
    tpu_gdma_cpy_S2L(index_local_addr,
                     input_addr + src_size * tpu_data_type_size(DT_UINT32),
                     &index_shape,
                     NULL,
                     NULL,
                     DT_UINT32);
    tpu_gdma_h_gather_L2S(output_addr + output_l2s_offset,
                          input_local_addr,
                          index_local_addr,
                          0,
                          C_value,
                          &output_shape_s,
                          src_h_s,
                          NULL, NULL, NULL, dtype);

    // test L2L
    printf("\nL2L: C=%d, src_h=%d, src_w=%d, idx_h=%d, src_c_is1=%d, idx_c_is1=%d \n",
            C_s, src_h_s, src_w_s, index_h_s, src_c_is1_s, idx_c_is1_s);
    u64 output_l2l_offset = (dst_size + 2 * dst_size_s) * tpu_data_type_size(dtype);
    tpu_gdma_cpy_S2L(input_local_addr,
                     input_addr,
                     &input_shape_s,
                     NULL,
                     NULL,
                     dtype);
    tpu_gdma_cpy_S2L(index_local_addr,
                     input_addr + src_size * tpu_data_type_size(DT_UINT32),
                     &index_shape,
                     NULL,
                     NULL,
                     DT_UINT32);
    tpu_gdma_h_gather_L2L(output_local_addr,
                          input_local_addr,
                          index_local_addr,
                          0,
                          C_value,
                          &output_shape_s,
                          src_h_s,
                          NULL, NULL, NULL, dtype);
    tpu_gdma_cpy_L2S(output_addr + output_l2l_offset,
                     output_local_addr,
                     &output_shape_s,
                     NULL,
                     NULL,
                     dtype);

    tpu_poll();
}
