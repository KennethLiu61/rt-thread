#ifndef ATOMIC_TENSOR_ARITHMETIC_GEN_CMD_H
#define ATOMIC_TENSOR_ARITHMETIC_GEN_CMD_H
#include "firmware_common.h"
#ifdef __cplusplus
extern "C" {
#endif

// use for two_opd tensor_arithmetic
// include two_opd FP32/FP16/BFP16 AR
// can alse use for some two_opd fixed_point AR
void atomic_tensor_arithmetic_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, //len = 3, opd0, opd1, opd2
    int sym_saturate,
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node);

void atomic_tensor_arithmetic_div_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    PREC prec, // fp32, fp16, bf16
    int iter,
    int thread_id,
    CMD_ID_NODE * pid_node);

// use for ternary ternary tensor_arithmetic
void atomic_tensor_arithmetic_ternary_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int C_is_const,
    int * Short_str, // len = 3, opd0, opd1, res
    int * Sign, // len = 3, opd0, opd1, opd2
    int sym_saturate,
    PREC * Prec, // len = 4, opd0, opd1, opd2, res
    AR_OP op,
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node);

// use for SE/SG/SL
void atomic_tensor_arithmetic_select_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, // len = 2, opd0/opd1, opd2/res
    PREC * Prec, //len = 2, opd0/opd1, opd2/res
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node);

// use for two_opds with round(shift\mulDhr)
void atomic_tensor_arithmetic_with_round_gen_cmd(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, // len = 2
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op,
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node);

// use for dtype convert
void atomic_tensor_arithmetic_dtype_convert_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    int * Sign,
    int sym_saturate,
    PREC * Prec,
    ROUND_MODE round,
    int thread_id,
    CMD_ID_NODE * pid_node);

// copy/copy_mb/abs/not
//void atomic_tensor_arithmetic_copy_like_gen_cmd(
void atomic_tensor_arithmetic_single_opd_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    int sign, // for abs
    PREC Prec,
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node);

// use for get_first_zero/get_first_one
void atomic_tensor_arithmetic_get_first_gen_cmd(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    PREC * Prec, // A && R can have different dtype
    AR_OP op,
    int thread_id,
    CMD_ID_NODE * pid_node);

#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_TENSOR_ARITHMETIC_GEN_CMD_H */
