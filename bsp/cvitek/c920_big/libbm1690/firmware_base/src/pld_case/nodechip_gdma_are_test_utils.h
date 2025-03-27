#include "nodechip_pld_test.h"
#include "tpu_kernel.h"
#include "tpu_defs.h"

#define N 1
#define C 4
#define H 4
#define W 4
#define cnt 1 * 4 * 4 * 4

#define L1(dtype, reduce_op) tpu_gdma_cpy_reduce_L12L2(l2_reduce, local_input, &shape, NULL, NULL, dtype, 1, reduce_op)
#define S(dtype, reduce_op) tpu_gdma_cpy_reduce_S2L2(l2_reduce, input, &shape, NULL, NULL, dtype, 1, reduce_op)
#define L2(dtype, reduce_op) tpu_gdma_cpy_reduce_L22L2(l2_reduce, l2_reduce_buffer, &shape, NULL, NULL, dtype, 1, reduce_op)
#define compress_S(reduce_op) tpu_gdma_lossy_compress_reduce_S2L2(l2_reduce, input, &shape, NULL, NULL, 1, reduce_op)
#define compress_L1(reduce_op) tpu_gdma_lossy_compress_reduce_L12L2(l2_reduce, local_input, &shape, NULL, NULL, 1, reduce_op)
#define compress_L2(reduce_op) tpu_gdma_lossy_compress_reduce_L22L2(l2_reduce, l2_reduce_buffer, &shape, NULL, NULL, 1, reduce_op)
#define decompress_S(reduce_op) tpu_gdma_lossy_decompress_reduce_S2L2(l2_reduce, input, &shape, NULL, NULL, 1, reduce_op)
#define decompress_L2(reduce_op) tpu_gdma_lossy_decompress_reduce_L22L2(l2_reduce, l2_reduce_buffer, &shape, NULL, NULL, 1, reduce_op)
#define REDUCE_ADD 4
#define REDUCE_MUL 1
#define REDUCE_MAX 2
#define REDUCE_MIN 3

#define TEST_GDMA_TENSOR_REDUCE(src) \
    src(DT_FP32, REDUCE_ADD); \
    src(DT_FP16, REDUCE_MUL); \
    src(DT_BFP16, REDUCE_MIN); \
    src(DT_FP20, REDUCE_MAX); \
    src(DT_INT32, REDUCE_MIN); \
    src(DT_FP20, REDUCE_MUL); \
    src(DT_BFP16, REDUCE_ADD); \
    src(DT_FP32, REDUCE_MIN); \
    src(DT_FP16, REDUCE_MAX); \
    src(DT_INT32, REDUCE_ADD); \
    src(DT_FP20, REDUCE_ADD); \
    src(DT_BFP16, REDUCE_MUL); \
    src(DT_FP16, REDUCE_MIN); \
    src(DT_FP32, REDUCE_MAX); \
    src(DT_BFP16, REDUCE_MAX); \
    src(DT_FP20, REDUCE_MIN); \
    src(DT_INT32, REDUCE_MAX); \
    src(DT_FP16, REDUCE_ADD); \
    src(DT_FP32, REDUCE_MUL)

#define TEST_GDMA_TENSOR_REDUCE_L(src) \
    src(DT_FP32, REDUCE_ADD); \
    src(DT_FP16, REDUCE_MUL); \
    src(DT_BFP16, REDUCE_MIN); \
    src(DT_INT32, REDUCE_MIN); \
    src(DT_BFP16, REDUCE_ADD); \
    src(DT_FP32, REDUCE_MIN); \
    src(DT_FP16, REDUCE_MAX); \
    src(DT_INT32, REDUCE_ADD); \
    src(DT_BFP16, REDUCE_MUL); \
    src(DT_FP16, REDUCE_MIN); \
    src(DT_FP32, REDUCE_MAX); \
    src(DT_BFP16, REDUCE_MAX); \
    src(DT_INT32, REDUCE_MAX); \
    src(DT_FP16, REDUCE_ADD); \
    src(DT_FP32, REDUCE_MUL)
