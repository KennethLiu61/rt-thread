#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"
#include "common.h"
#include <stdlib.h>

static inline data_type_t get_prec(PREC precision, bool is_sign)
 {
  data_type_t dtype = -1;
  if (precision == INT8) {
    if (is_sign) dtype = DT_INT8;
    else dtype = DT_UINT8;
  }
  else if (precision == FP8){
    if (is_sign) dtype = DT_FP8E4M3;
    else dtype = DT_FP8E5M2;
  }
  else if (precision == FP16) {
    dtype = DT_FP16;
  }
  else if (precision == FP32) {
    dtype = DT_FP32;
  }
  else if (precision == INT16) {
    if (is_sign) dtype = DT_INT16;
    else dtype = DT_UINT16;
  }
  else if (precision == INT32) {
    if (is_sign) dtype = DT_INT32;
    else dtype = DT_UINT32;
  }
  else if (precision == BFP16) {
    dtype = DT_BFP16;
  }
  else {
    // printf("Not support precision=%d \n", precision);
  }
  return dtype;
}

void nodechip_srch_bin_test(unsigned char* api_buf) {
    tpu_initialize();
    sg_api_pld_srch_bin_test_t *api = (sg_api_pld_srch_bin_test_t *)api_buf;
    printf("Test fused cmp srch bin:\n");

    int sign = api->sign;
    data_type_t src_prec = get_prec((PREC)api->src_dtype, sign);
    data_type_t dst_prec = get_prec((PREC)api->dst_dtype, 0);
    dim4 shape = {.n = api->N, .c = api->C, .h = api->H, .w = api->W};
    dim4 l_stride;
    tpu_aligned_stride(&l_stride, 0, &shape, src_prec);

    dim4 bin_shape = {.n = 1, .c = api->C, .h = 1, .w = api->bin_w};
    dim4 bin_l_stride;
    tpu_aligned_stride(&bin_l_stride, 0, &bin_shape, src_prec);

    uint64_t src_addr = api->src_addr;
    uint64_t bin_addr = api->bin_addr;
    uint64_t dst_addr = api->dst_addr;
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    local_addr_t src_l_addr = 0;
    local_addr_t bin_l_addr = ALIGN(src_l_addr + l_stride.n * shape.n * sizeof(float), bank_size);
    local_addr_t output_l_addr = ALIGN(bin_l_addr + bin_l_stride.n * bin_shape.n * sizeof(float), bank_size);
    TPUKERNEL_ASSERT(output_l_addr + l_stride.n * shape.n * sizeof(float) <= (unsigned int)LOCAL_MEM_SIZE);
    variable_t src_var = {.type = TENSOR, .context.addr = src_l_addr};
    variable_t bin_var = {.type = TENSOR, .context.addr = bin_l_addr};

    tpu_gdma_cpy_S2L(
        src_l_addr,
        src_addr,
        &shape,
        NULL,
        NULL,
        src_prec);

    tpu_gdma_cpy_S2L(
        bin_l_addr,
        bin_addr,
        &bin_shape,
        NULL,
        NULL,
        src_prec);

    tpu_bdc_srch_bin_select(
        output_l_addr,
        &src_var,
        &bin_var,
        &shape,
        api->side,
        api->bin_w,
        src_prec,
        dst_prec);

    tpu_gdma_cpy_L2S(
        dst_addr,
        output_l_addr,
        &shape,
        NULL,
        NULL,
        dst_prec);

    tpu_poll();
}
