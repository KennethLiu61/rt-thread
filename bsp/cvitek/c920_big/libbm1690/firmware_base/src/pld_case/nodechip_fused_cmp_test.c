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
    return dtype;
  }
  else if (precision == FP8){
    if (is_sign) dtype = DT_FP8E4M3;
    else dtype = DT_FP8E5M2;
    return dtype;
  }
  else if (precision == FP16) {
    return DT_FP16;
  }
  else if (precision == FP32) {
    return DT_FP32;
  }
  else if (precision == INT16) {
    if (is_sign) dtype = DT_INT16;
    else dtype = DT_UINT16;
    return dtype;
  }
  else if (precision == INT32) {
    if (is_sign) dtype = DT_INT32;
    else dtype = DT_UINT32;
    return dtype;
  }
  else if (precision == BFP16) {
    return DT_BFP16;
  }
  TPUKERNEL_ASSERT(0);
  return dtype;
}

int get_byte_size(data_type_t dtype) {
  int byte_size = 0;
  if (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2) {
    return 1;
  }
  else if (dtype == DT_FP16 || dtype == DT_INT16 || dtype == DT_UINT16 || dtype == DT_BFP16) {
    return 2;
  }
  else if (dtype == DT_FP32 || dtype == DT_INT32 || dtype == DT_UINT32) {
    return 4;
  }
  TPUKERNEL_ASSERT(0);
  return byte_size;
}

void nodechip_fused_cmp_test(unsigned char* api_buf) {
    tpu_initialize();
    sg_api_pld_fused_cmp_t *api = (sg_api_pld_fused_cmp_t *) api_buf;
    printf("Test fused cmp:\n");

    int sign = api->a_is_sign;
    data_type_t a_prec = get_prec((PREC)api->A_dtype, sign);
    data_type_t c_prec = get_prec((PREC)api->C_dtype, 0);
    dim4 shape = {.n = api->N, .c = api->C, .h = api->H, .w = api->W};
    int size = shape.n *shape.c *shape.h * shape.w;
    dim4 a_l_stride;
    tpu_aligned_stride(&a_l_stride, 0, &shape, a_prec);
    dim4 c_l_stride;
    tpu_aligned_stride(&c_l_stride, 0, &shape, c_prec);

    uint64_t src_addr = api->src_addr;
    uint64_t dst0_addr = api->dst0_addr;
    uint64_t dst1_addr = api->dst1_addr;
    const int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
    // input
    local_addr_t src0_l_addr = 0;
    local_addr_t src1_l_addr = ALIGN(src0_l_addr + a_l_stride.n * shape.n * get_byte_size(a_prec), bank_size);
    local_addr_t src2_l_addr = ALIGN(src1_l_addr + a_l_stride.n * shape.n * get_byte_size(a_prec), bank_size);
    local_addr_t src3_l_addr = ALIGN(src2_l_addr + c_l_stride.n * shape.n * get_byte_size(c_prec), bank_size);

    // output0
    int big_byte_size = get_byte_size(a_prec) > get_byte_size(c_prec) ? get_byte_size(a_prec) : get_byte_size(c_prec);
    int big_stride = a_l_stride.n > c_l_stride.n ? a_l_stride.n : c_l_stride.n;
    local_addr_t output0_l_addr = ALIGN(src3_l_addr + c_l_stride.n * shape.n * get_byte_size(c_prec), bank_size);
  
    // output1
    local_addr_t output1_l_addr = ALIGN(output0_l_addr + big_stride * shape.n * big_byte_size, bank_size);
    TPUKERNEL_ASSERT(output1_l_addr + c_l_stride.n * shape.n * get_byte_size(c_prec) < (unsigned int)tpu_local_mem_size_per_npu());

    variable_t src0_var = {.type = TENSOR, .context.addr = src0_l_addr};
    variable_t src1_var = {.type = TENSOR, .context.addr = src1_l_addr};
    variable_t src2_var = {.type = TENSOR, .context.addr = src2_l_addr};
    variable_t src3_var = {.type = TENSOR, .context.addr = src3_l_addr};
    int offset = 0;
    // load opd0 a_prec
    tpu_gdma_cpy_S2L(
        src0_l_addr,
        src_addr,
        &shape,
        NULL,
        NULL,
        a_prec);
    offset += size * get_byte_size(a_prec);
    // load opd1 a_prec
    tpu_gdma_cpy_S2L(
        src1_l_addr,
        src_addr + offset,
        &shape,
        NULL,
        NULL,
        a_prec);
    offset += size * get_byte_size(a_prec);
    // load opd2 c_prec
    tpu_gdma_cpy_S2L(
        src2_l_addr,
        src_addr + offset,
        &shape,
        NULL,
        NULL,
        c_prec);
    offset += size * get_byte_size(c_prec);
    // load opd3 c_prec
    tpu_gdma_cpy_S2L(
        src3_l_addr,
        src_addr + offset,
        &shape,
        NULL,
        NULL,
        c_prec);

    int dst0_offset = 0;
    int dst1_offset = 0;
    /*
    input:
        opd0,   opd1,   opd2,   opd3
        a_prec, a_prec, c_prec, c_prec
    output:
        cmp_gt_and_sel:
            res0,               res1
            a_prec,             c_prec
        cmp_lt_and_sel:
            res0,               res1
            a_prec,             c_prec
        cmp_sel_gt:
            res0,
            c_prec
        cmp_sel_lt:
            res0,
            c_prec
        cmp_sel_eq:
            res0,
            c_prec
        so,
        res0:
            a_prec, a_prec, c_prec, c_prec, c_prec
        res1:
            c_prec, c_prec
    */
    CORE_PRINT("------ tpu_bdc_maximum_greater_select test -----\n");
    tpu_bdc_maximum_greater_select(
        output0_l_addr,
        output1_l_addr,
        &src0_var,
        &src1_var,
        &src2_var,
        &src3_var,
        &shape,
        a_prec,
        c_prec);
    // store dst0 a_prec
    tpu_gdma_cpy_L2S(
        dst0_addr,
        output0_l_addr,
        &shape,
        NULL,
        NULL,
        a_prec);
    dst0_offset += size * get_byte_size(a_prec);
    // store dst1 c_prec
    tpu_gdma_cpy_L2S(
        dst1_addr,
        output1_l_addr,
        &shape,
        NULL,
        NULL,
        c_prec);
    dst1_offset += size * get_byte_size(c_prec);

    CORE_PRINT("------ tpu_bdc_minimum_less_select -----\n");
    tpu_bdc_minimum_less_select(
        output0_l_addr,
        output1_l_addr,
        &src0_var,
        &src1_var,
        &src2_var,
        &src3_var,
        &shape,
        a_prec,
        c_prec);
    // store dst0 a_prec
    tpu_gdma_cpy_L2S(
        dst0_addr + dst0_offset,
        output0_l_addr,
        &shape,
        NULL,
        NULL,
        a_prec);
    dst0_offset += size * get_byte_size(a_prec);
    // store dst1 c_prec
    tpu_gdma_cpy_L2S(
        dst1_addr + dst1_offset,
        output1_l_addr,
        &shape,
        NULL,
        NULL,
        c_prec);
    dst1_offset += size * get_byte_size(c_prec);

    CORE_PRINT("------ tpu_bdc_greater_select -----\n");
    tpu_bdc_greater_select(
        output0_l_addr,
        &src0_var,
        &src1_var,
        &src2_var,
        &src3_var,
        &shape,
        a_prec,
        c_prec);
    // store dst 0 c_prec
    tpu_gdma_cpy_L2S(
        dst0_addr + dst0_offset,
        output0_l_addr,
        &shape,
        NULL,
        NULL,
        c_prec);
    dst0_offset += size * get_byte_size(c_prec);

    CORE_PRINT("------ tpu_bdc_less_select -----\n");
    tpu_bdc_less_select(
        output0_l_addr,
        &src0_var,
        &src1_var,
        &src2_var,
        &src3_var,
        &shape,
        a_prec,
        c_prec);
    // store dst 0 c_prec
    tpu_gdma_cpy_L2S(
        dst0_addr + dst0_offset,
        output0_l_addr,
        &shape,
        NULL,
        NULL,
        c_prec);
    dst0_offset += size * get_byte_size(c_prec);

    CORE_PRINT("------ tpu_bdc_equal_select -----\n");
    tpu_bdc_equal_select(
        output0_l_addr,
        &src0_var,
        &src1_var,
        &src2_var,
        &src3_var,
        &shape,
        a_prec,
        c_prec);
    // store dst 0 c_prec
    tpu_gdma_cpy_L2S(
        dst0_addr + dst0_offset,
        output0_l_addr,
        &shape,
        NULL,
        NULL,
        c_prec);
    tpu_poll();
}
