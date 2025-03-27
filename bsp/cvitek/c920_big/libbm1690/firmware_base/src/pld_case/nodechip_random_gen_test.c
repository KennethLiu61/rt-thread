#include "nodechip_pld_test.h"
#include "tpu_kernel.h"

void nodechip_random_gen_test(
    global_addr_t r_out_addr,
    global_addr_t m_out_addr)
{
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

    dim4 tmp_shape = {
        .n = 1,
        .c = 64,
        .h = 1,
        .w = 64};

    dim4 state_shape = {
        .n = 1,
        .c = 64,
        .h = 1,
        .w = 2 * 8 * 64 / 8};

    int jump_cnt = 10;
    int c_offset = 10;
    local_addr_t res_addr = 0;
    local_addr_t state_addr = 4 * bank_size;
    int dtype[] = {DT_INT8, DT_INT16, DT_INT32};
    uint32_t offset = 0;
    uint32_t cnt = 64 * 64;
    uint32_t state_byte = 64 * 2 * 8 * 64 * 8 / 8;
    for (int i = 0; i < 1; i++)
    {
        tpu_bd_set_random_gen_seed(i * 0x123568);
        tpu_bd_rand_seed_gen();
        tpu_bdc_random_gen_init(res_addr, state_addr, 1, jump_cnt, c_offset, &tmp_shape, dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, res_addr, &tmp_shape, NULL, NULL, dtype[i]);
        offset += cnt * tpu_data_type_size(dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, state_addr, &state_shape, NULL, NULL, DT_INT8);
        offset += state_byte;
        tpu_bdc_random_gen(res_addr, state_addr, 1, &tmp_shape, dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, res_addr, &tmp_shape, NULL, NULL, dtype[i]);
        offset += cnt * tpu_data_type_size(dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, state_addr, &state_shape, NULL, NULL, DT_INT8);
        offset += state_byte;
        tpu_bdc_random_gen_load_state(res_addr, state_addr, state_addr, 1, &tmp_shape, dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, res_addr, &tmp_shape, NULL, NULL, dtype[i]);
        offset += cnt * tpu_data_type_size(dtype[i]);
        tpu_gdma_cpy_L2S(m_out_addr + offset, state_addr, &state_shape, NULL, NULL, DT_INT8);
        offset += state_byte;
    }
}