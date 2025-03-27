#include "tpu_kernel_internel.h"

const var_context_t VAR_CONTEXT_ZERO = {.addr = 0};
void tpu_bdc_fp_mm_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(left_right_dtype == DT_FP16 ||
                    left_right_dtype == DT_BFP16 ||
                    left_right_dtype == DT_FP8E5M2 ||
                    left_right_dtype == DT_FP8E4M3 ||
                    left_right_dtype == DT_TF32 );
    int tf32_mode = NO_USE;
    if(left_right_dtype == DT_TF32) {
        tf32_mode = 1;
        left_right_dtype = DT_FP32;
    }
    if(PRECISION(left_right_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == left_right_dtype);
    }

    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_rows,
        left_cols,
        right_cols,
        false,
        false,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(left_right_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(left_right_dtype), //todo, L_fp8_type
        FP8TYPE(left_right_dtype), //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_mm_R_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(left_right_dtype == DT_FP16 ||
                    left_right_dtype == DT_BFP16 ||
                    left_right_dtype == DT_FP8E5M2 ||
                    left_right_dtype == DT_FP8E4M3 ||
                    left_right_dtype == DT_TF32 ||
                    left_right_dtype == DT_FP32);
    int tf32_mode = NO_USE;
    if(left_right_dtype == DT_TF32) {
        tf32_mode = 1;
        left_right_dtype = DT_FP32;
    }
    if(PRECISION(left_right_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == left_right_dtype);
    }
    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_rows,
        left_cols,
        right_rows,
        false,
        true,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(left_right_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(left_right_dtype), //todo, L_fp8_type
        FP8TYPE(left_right_dtype), //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_all_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(left_right_dtype == DT_FP16 ||
                    left_right_dtype == DT_BFP16 ||
                    left_right_dtype == DT_FP8E5M2 ||
                    left_right_dtype == DT_FP8E4M3 ||
                    left_right_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(left_right_dtype == DT_TF32) {
        tf32_mode = 1;
        left_right_dtype = DT_FP32;
    }
    if(PRECISION(left_right_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == left_right_dtype);
    }
    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_cols,
        left_rows,
        right_rows,
        true,
        true,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(left_right_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(left_right_dtype), //todo, L_fp8_type
        FP8TYPE(left_right_dtype), //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_L_const_with_bias(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(right_C_dtype == DT_FP16 || right_C_dtype == DT_BFP16 ||
                     right_C_dtype == DT_FP8E5M2 || right_C_dtype == DT_FP8E4M3 ||
                     right_C_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(right_C_dtype == DT_TF32) {
        tf32_mode = 1;
        right_C_dtype = DT_FP32;
    }
    if(PRECISION(right_C_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == right_C_dtype);
    }
    atomic_mm2_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_rows,
        left_cols,
        right_cols,
        false,
        false,
        true,
        false,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(right_C_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        (FP8_TYPE)NO_USE, //todo, L_fp8_type
        FP8TYPE(right_C_dtype), //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_R_const_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(left_C_dtype == DT_FP16 || left_C_dtype == DT_BFP16 ||
                     left_C_dtype == DT_FP8E5M2 || left_C_dtype == DT_FP8E4M3 ||
                     left_C_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(left_C_dtype == DT_TF32) {
        tf32_mode = 1;
        left_C_dtype = DT_FP32;
    }
    if(PRECISION(left_C_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == left_C_dtype);
    }
    atomic_mm2_gen_cmd(
        left_addr,
        C.u32,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_rows,
        left_cols,
        right_cols,
        false,
        false,
        false,
        true,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(left_C_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(left_C_dtype), //todo, L_fp8_type
        (FP8_TYPE)NO_USE, //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_L_const_R_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
){
    TPUKERNEL_ASSERT(right_C_dtype == DT_FP16 || right_C_dtype == DT_BFP16 ||
                     right_C_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(right_C_dtype == DT_TF32) {
        tf32_mode = 1;
        right_C_dtype = DT_FP32;
    }
    if(PRECISION(right_C_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == right_C_dtype);
    }
    atomic_mm2_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_rows,
        left_cols,
        right_rows,
        false,
        true,
        true,
        false,
        bias_is_const,
        false,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(right_C_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        (FP8_TYPE)NO_USE, //todo, L_fp8_type
        (FP8_TYPE)NO_USE, //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_L_const_all_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(right_C_dtype == DT_FP16 || right_C_dtype == DT_BFP16 ||
                     right_C_dtype == DT_FP8E5M2 || right_C_dtype == DT_FP8E4M3 ||
                     right_C_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(right_C_dtype == DT_TF32) {
        tf32_mode = 1;
        right_C_dtype = DT_FP32;
    }
    if(PRECISION(right_C_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == right_C_dtype);
    }
    atomic_mm2_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_cols,
        left_rows,
        right_rows,
        true,
        true,
        true,
        false,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(right_C_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(right_C_dtype), //todo, L_fp8_type
        (FP8_TYPE)NO_USE, //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_fp_mm_R_const_all_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu
    ) {
    TPUKERNEL_ASSERT(left_C_dtype == DT_FP16 || left_C_dtype == DT_BFP16 ||
                     left_C_dtype == DT_FP8E5M2 || left_C_dtype == DT_FP8E4M3 ||
                     left_C_dtype == DT_TF32);
    int tf32_mode = NO_USE;
    if(left_C_dtype == DT_TF32) {
        tf32_mode = 1;
        left_C_dtype = DT_FP32;
    }
    if(PRECISION(left_C_dtype) == FP8) {
        TPUKERNEL_ASSERT(IS_FLOAT(output_dtype));
    } else {
        TPUKERNEL_ASSERT(output_dtype == DT_FP32 ||
                    output_dtype == left_C_dtype);
    }
    atomic_mm2_gen_cmd(
        left_addr,
        C.u32,
        output_addr,
        bias_data.addr,
        NO_USE, //todo rq_addr
        left_cols,
        left_rows,
        right_rows,
        true,
        true,
        false,
        true,
        bias_is_const,
        result_add,
        do_relu,
        NO_USE, //todo do rq
        NO_USE, //todo is_rq_const
        PRECISION(left_C_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        FP8TYPE(left_C_dtype), //todo, L_fp8_type
        (FP8_TYPE)NO_USE, //todo, R_fp8_type
        FP8TYPE(output_dtype), //todo, Y_fp8_type
        MASTER_THREAD,
        tf32_mode, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    bool          result_add) {
    tpu_bdc_fp_mm_with_bias(
        output_addr,
        left_addr,
        right_addr,
        left_rows,
        left_cols,
        right_cols,
        output_dtype,
        left_right_dtype,
        DT_FP32,
        result_add, 1, VAR_CONTEXT_ZERO, 0);
}

void tpu_bdc_fp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype) {
    tpu_bdc_fp_mm_R_trans_with_bias(
          output_addr,
          left_addr,
          right_addr,
          left_rows,
          left_cols,
          right_rows,
          output_dtype,
          left_right_dtype,
          DT_FP32,
          0, 1, VAR_CONTEXT_ZERO, 0);
}

void tpu_bdc_fp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    bool          result_add) {
      tpu_bdc_fp_mm_all_trans_with_bias(
          output_addr,
          left_addr,
          right_addr,
          left_rows,
          left_cols,
          right_rows,
          output_dtype,
          left_right_dtype,
          DT_FP32,
          result_add, 1, VAR_CONTEXT_ZERO, 0);
}
void tpu_bdc_fp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    bool          result_add) {
      tpu_bdc_fp_mm_L_const_with_bias(
          output_addr,
          right_addr,
          C,
          left_rows,
          left_cols,
          right_cols,
          output_dtype,
          right_C_dtype,
          DT_FP32,
          result_add, 1, VAR_CONTEXT_ZERO, 0);
}
void tpu_bdc_fp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    bool          result_add) {
    tpu_bdc_fp_mm_R_const_with_bias(
          output_addr,
          left_addr,
          C,
          left_rows,
          left_cols,
          right_cols,
          output_dtype,
          left_C_dtype,
          DT_FP32,
          result_add, 1, VAR_CONTEXT_ZERO, 0);
}
void tpu_bdc_fp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype) {
      tpu_bdc_fp_mm_L_const_R_trans_with_bias(
          output_addr,
          right_addr,
          C,
          left_rows,
          left_cols,
          right_rows,
          output_dtype,
          right_C_dtype,
          DT_FP32,
          0, 1, VAR_CONTEXT_ZERO, 0);
}
void tpu_bdc_fp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    bool          result_add) {
      tpu_bdc_fp_mm_L_const_all_trans_with_bias(
          output_addr,
          right_addr,
          C,
          left_rows,
          left_cols,
          right_rows,
          output_dtype,
          right_C_dtype,
          DT_FP32,
          result_add, 1, VAR_CONTEXT_ZERO, 0);
}

void tpu_bdc_fp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    bool          result_add) {
      tpu_bdc_fp_mm_R_const_all_trans_with_bias(
          output_addr,
          left_addr,
          C,
          left_rows,
          left_cols,
          right_rows,
          output_dtype,
          left_C_dtype,
          DT_FP32,
          result_add, 1, VAR_CONTEXT_ZERO, 0);
}

void tpu_bdc_fp8_mm_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    TPUKERNEL_ASSERT(left_dtype == DT_FP8E4M3 || left_dtype == DT_FP8E5M2);
    TPUKERNEL_ASSERT(right_dtype == DT_FP8E4M3 || right_dtype == DT_FP8E5M2);
    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        bias_data.addr,
        do_rescale ? (rescale_is_const ? rescale_data.scalar.u32 : rescale_data.addr) : 0,
        left_rows,
        left_cols,
        right_cols,
        false,
        false,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        do_rescale,
        rescale_is_const,
        PRECISION(left_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        left_dtype == DT_FP8E4M3,
        right_dtype == DT_FP8E4M3,
        output_dtype == DT_FP8E4M3,
        MASTER_THREAD,
        NO_USE, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp8_mm_R_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    TPUKERNEL_ASSERT(left_dtype == DT_FP8E4M3 || left_dtype == DT_FP8E5M2);
    TPUKERNEL_ASSERT(right_dtype == DT_FP8E4M3 || right_dtype == DT_FP8E5M2);
    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        bias_data.addr,
        do_rescale ? (rescale_is_const ? rescale_data.scalar.u32 : rescale_data.addr) : 0,
        left_rows,
        left_cols,
        right_rows,
        false,
        true,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        do_rescale,
        rescale_is_const,
        PRECISION(left_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        left_dtype == DT_FP8E4M3,
        right_dtype == DT_FP8E4M3,
        output_dtype == DT_FP8E4M3,
        MASTER_THREAD,
        NO_USE, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp8_mm_all_trans_with_bias(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    data_type_t   bias_dtype,
    bool          result_add,
    bool          bias_is_const,
    var_context_t bias_data,  // addr or fp32 const value
    bool          do_relu,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    TPUKERNEL_ASSERT(left_dtype == DT_FP8E4M3 || left_dtype == DT_FP8E5M2);
    TPUKERNEL_ASSERT(right_dtype == DT_FP8E4M3 || right_dtype == DT_FP8E5M2);
    atomic_mm2_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        -1,
        do_rescale ? (rescale_is_const ? rescale_data.scalar.u32 : rescale_data.addr) : 0,
        left_rows,
        left_cols,
        right_rows,
        false,
        false,
        false,
        false,
        bias_is_const,
        result_add,
        do_relu,
        do_rescale,
        rescale_is_const,
        PRECISION(left_dtype),
        PRECISION(bias_dtype),
        PRECISION(output_dtype),
        left_dtype == DT_FP8E4M3,
        right_dtype == DT_FP8E4M3,
        output_dtype == DT_FP8E4M3,
        MASTER_THREAD,
        NO_USE, // tf32_mode
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_fp8_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    bool          result_add,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    tpu_bdc_fp8_mm_with_bias(
        output_addr,
        left_addr,
        right_addr,
        left_rows,
        left_cols,
        right_cols,
        output_dtype,
        left_dtype,
        right_dtype,
        DT_FP32,
        result_add,
        true,
        (var_context_t){0},
        false,
        do_rescale,
        rescale_is_const,
        rescale_data);
}

void tpu_bdc_fp8_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    bool          result_add,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    tpu_bdc_fp8_mm_R_trans_with_bias(
        output_addr,
        left_addr,
        right_addr,
        left_rows,
        left_cols,
        right_rows,
        output_dtype,
        left_dtype,
        right_dtype,
        DT_FP32,
        result_add,
        true,
        (var_context_t){0},
        false,
        do_rescale,
        rescale_is_const,
        rescale_data);
}

void tpu_bdc_fp8_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_dtype,
    data_type_t   right_dtype,
    bool          result_add,
    bool          do_rescale,
    bool          rescale_is_const,
    var_context_t rescale_data) {
    tpu_bdc_fp8_mm_all_trans_with_bias(
        output_addr,
        left_addr,
        right_addr,
        left_rows,
        left_cols,
        right_rows,
        output_dtype,
        left_dtype,
        right_dtype,
        DT_FP32,
        result_add,
        true,
        (var_context_t){0},
        false,
        do_rescale,
        rescale_is_const,
        rescale_data);
}

void tpu_bdc_int8_mm(
    local_addr_t   output_addr,
    local_addr_t   left_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            left_cols_per_channel,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    left_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT8 || PRECISION(output_dtype) == INT16 || PRECISION(output_dtype) == INT32);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(left_dtype) | SIGN(right_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(left_dtype) | SIGN(right_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == INT8 || PRECISION(left_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(right_dtype) == INT8 || PRECISION(right_dtype) == INT16);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);
    TPUKERNEL_ASSERT(lshift==0 && rshift==0 && !result_add);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        !has_bias,
        0, //please change add_result
        result_relu, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        (PREC)PRECISION(output_dtype),
        INT8, //please change LR_PREC
        (ROUND_MODE) 0, //please change RQ ROUND_MODE
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}
void tpu_bdc_int8_mm_L_trans(
    local_addr_t   output_addr,
    local_addr_t   left_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            left_cols_per_channel,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    left_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT8 || PRECISION(output_dtype) == INT16 || PRECISION(output_dtype) == INT32);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(left_dtype) | SIGN(right_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(left_dtype) | SIGN(right_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(left_dtype) == INT8 || PRECISION(left_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(right_dtype) == INT8 || PRECISION(right_dtype) == INT16);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);
    TPUKERNEL_ASSERT(lshift==0 && rshift==0 && !result_add);
    atomic_mm_fixed_gen_cmd(
        left_addr,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        left_cols_per_channel,
        DIV_UP(left_cols, left_cols_per_channel),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_cols,
        left_rows,
        true,
        false,
        SIGN(left_dtype),
        SIGN(right_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        !has_bias,
        0, //please change add_result
        result_relu,
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        (PREC)PRECISION(output_dtype),
        INT8, //please change LR_PREC
        (ROUND_MODE) 0, //please change RQ ROUND_MODE
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

void tpu_bdc_int8_mm_L_const(
    local_addr_t   output_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    scalar_t       C,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    C_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu) {
    TPUKERNEL_ASSERT(
        PRECISION(output_dtype) == INT8 || PRECISION(output_dtype) == INT16 || PRECISION(output_dtype) == INT32);
    if (result_relu)
        TPUKERNEL_ASSERT(!SIGN(output_dtype));
    else {
        if (has_bias)
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) ==
                (SIGN(C_dtype) | SIGN(right_dtype) | SIGN(bias_dtype)));
        else
            TPUKERNEL_ASSERT(
                SIGN(output_dtype) == (SIGN(C_dtype) | SIGN(right_dtype)));
    }
    TPUKERNEL_ASSERT(PRECISION(C_dtype) == INT8 || PRECISION(C_dtype) == INT16);
    TPUKERNEL_ASSERT(PRECISION(right_dtype) == INT8 || PRECISION(right_dtype) == INT16);
    if (has_bias)
        TPUKERNEL_ASSERT(PRECISION(bias_dtype) == INT32);
    TPUKERNEL_ASSERT(lshift==0 && rshift==0 && !result_add);
    atomic_mm_fixed_gen_cmd(
        C.u32,
        right_addr,
        output_addr,
        has_bias ? bias_addr : 0,
        tpu_eu_num(C_dtype),
        DIV_UP(left_cols, tpu_eu_num(C_dtype)),
        right_cols_per_channel,
        DIV_UP(right_cols, right_cols_per_channel),
        left_rows,
        left_cols,
        false,
        true,
        SIGN(C_dtype),
        SIGN(right_dtype),
        SIGN(bias_dtype),
        SIGN(output_dtype),
        !has_bias,
        0, //please change add_result
        result_relu, //please change if_relu
        0, //please change sym_range
        0, //do_rq
        0, // rq multiplier
        0, // rq shift
        0, // rq yzp
        (PREC)PRECISION(output_dtype),
        INT8, //please change LR_PREC
        (ROUND_MODE) 0, //please change RQ ROUND_MODE
        MASTER_THREAD,
        BDC_NODE);
    CHECK_BDC_OVERFLOW;
}

#define BIAS_DECLARE(bias)                 \
    OPTIONAL_INFO_DECLARE(bias, DT_INT32); \
    if(bias)                               \
      TPUKERNEL_ASSERT(bias_dtype == DT_INT32 || bias_dtype == DT_UINT32)

#define RZP_DECLARE(rzp)                  \
    OPTIONAL_INFO_DECLARE(rzp, DT_INT16); \
    TPUKERNEL_ASSERT(rzp_dtype == DT_INT16)

void tpu_bdc_quant_mm(
    local_addr_t output_addr,
    data_type_t output_dtype,
    optional_info_t* left,
    optional_info_t* right,
    int output_rows,
    int inner_cols,
    int output_cols,
    bool left_trans,
    bool right_trans,
    bool result_add,
    bool do_relu,
    optional_info_t *rzp,
    optional_info_t *bias,
    requant_int_info_t *requant)
{
    OPTIONAL_INFO_DECLARE(left, DT_INT8);
    OPTIONAL_INFO_DECLARE(right, DT_INT8);
    BIAS_DECLARE(bias);
    RZP_DECLARE(rzp);
    REQUANT_INFO_DECLARE(requant);
    TPUKERNEL_ASSERT(left_dtype == DT_INT8 || left_dtype == DT_INT4 ||
                     left_dtype == DT_UINT8 || left_dtype == DT_UINT4);
    TPUKERNEL_ASSERT(right_dtype == DT_INT8 || right_dtype == DT_INT4 ||
                     right_dtype == DT_UINT8 || right_dtype == DT_UINT4);
    // {
    //     dim4 print_shape={1, left_rows, 1, left_cols};
    //     tpu_print_local_mem_data(left_value, 0, &print_shape, NULL, left_dtype);
    // }
    // {
    //     dim4 print_shape={1, left_cols, 1, right_cols};
    //     tpu_print_local_mem_data(right_value, 0, &print_shape, NULL, right_dtype);
    // }
    atomic_mm2_fixed_gen_cmd(
        left_value,
        right_value,
        output_addr,
        rzp_value,
        bias_value,
        requant_addr,
        shift,
        yzp,
        output_rows,
        inner_cols,
        output_cols,
        left_trans,
        right_trans,
        left_is_const,
        right_is_const,
        rzp_is_const,
        SIGN(left_dtype),
        SIGN(right_dtype),
        result_add,
        SIGN(output_dtype),
        SIGN(bias_dtype),
        bias_is_const,
        !is_perchannel,
        do_relu,
        do_sym_saturate,
        do_requant,
        round_mode,
        PRECISION(left_dtype),
        PRECISION(right_dtype),
        PRECISION(output_dtype),
        MASTER_THREAD,
        BDC_NODE);
    // {
    //     dim4 print_shape={1, left_rows, 1, right_cols};
    //     tpu_print_local_mem_data(output_addr, 0, &print_shape, NULL, output_dtype);
    // }
    CHECK_BDC_OVERFLOW;
    }

void tpu_bdc_int8_zp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        false, false,
        result_add, false,
        &zp, NULL, NULL);
}

void tpu_bdc_int8_zp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype){
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        false, true,
        false, false,
        &zp, NULL, NULL);
}

void tpu_bdc_int8_zp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
}

void tpu_bdc_int8_zp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        false, false,
        result_add, false,
        &zp, NULL, NULL);
}

void tpu_bdc_int8_zp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_VALUE(left_dtype, C);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        false, false,
        result_add, false,
        &zp, NULL, NULL);

}

void tpu_bdc_int8_zp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        false, true,
        false, false,
        &zp, NULL, NULL);
}

void tpu_bdc_int8_zp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_zp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_VALUE(DT_INT16, zp_val);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_VALUE(left_dtype, C);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        false, false,
        result_add, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        false, true,
        false, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_ADDR(left_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add){
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        false, false,
        result_add, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        false, true,
        false, false,
        &zp, NULL, NULL);
    }

void tpu_bdc_int8_pc_zp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_VALUE(C_dtype, C);
    optional_info_t right = OPTIONAL_ADDR(C_dtype, right_addr);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
    }


void tpu_bdc_int8_pc_zp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add){
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_VALUE(left_dtype, C);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_cols,
        true, true,
        result_add, false,
        &zp, NULL, NULL);

    }

void tpu_bdc_int8_pc_zp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add) {
    optional_info_t zp = OPTIONAL_ADDR(DT_INT16, zp_addr);
    optional_info_t left = OPTIONAL_ADDR(left_dtype, left_addr);
    optional_info_t right = OPTIONAL_VALUE(left_dtype, C);
    tpu_bdc_quant_mm(
        output_addr, DT_INT32, &left, &right,
        left_rows, left_cols, right_rows,
        true, true,
        result_add, false,
        &zp, NULL, NULL);
}
