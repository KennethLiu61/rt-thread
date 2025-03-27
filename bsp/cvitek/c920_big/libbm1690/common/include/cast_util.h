#include "sg_fp16.h"

template<typename Td, typename Ts>
Td fp_cast(Ts x);

template<>
float fp_cast(float x) {
    return x;
}

template<>
fp16 fp_cast(float x) {
    fp32 v_fp32 = {.fval = x};
    return fp32_to_fp16(v_fp32, ROUND_HALF_TO_EVEN, false);
}

template<>
bf16 fp_cast(float x) {
    fp32 v_fp32 = {.fval = x};
    return fp32_to_bf16(v_fp32, ROUND_HALF_TO_EVEN, false);
}

template<>
float fp_cast(fp16 x) {
    return fp16_to_fp32(x).fval;
}

template<>
float fp_cast(bf16 x) {
    return bf16_to_fp32(x).fval;
}