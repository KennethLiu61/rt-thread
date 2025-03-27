#include "sg_fp16.h"
#include "tpu_kernel.h"
#include <limits.h>
#define UINT4_MIN (0)
#define UINT4_MAX (15)
#define INT4_MIN (-8)
#define INT4_MAX (7)

ROUND_MODE convertRoundingMode(rounding_mode_t mode) {
    ROUND_MODE convertedMode = ROUND_HALF_TO_EVEN;

    switch (mode) {
        case RM_HALF_TO_EVEN:
            convertedMode = ROUND_HALF_TO_EVEN;
            break;
        case RM_HALF_AWAY_FROM_ZERO:
            convertedMode = ROUND_HALF_AWAY_FROM_ZERO;
            break;
        case RM_TOWARDS_ZERO:
            convertedMode = ROUND_TOWARDS_ZERO;
            break;
        case RM_DOWN:
            convertedMode = ROUND_DOWN;
            break;
        case RM_UP:
            convertedMode = ROUND_UP;
            break;
        case RM_HALF_UP:
            convertedMode = ROUND_HALF_UP;
            break;
        case RM_HALF_DOWN:
            convertedMode = ROUND_HALF_DOWN;
            break;
        default:
            break;
    }

    return convertedMode;
}

scalar_t tpu_int_cast(
    scalar_t     src,
    data_type_t  dst_dtype,
    data_type_t  src_dtype) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    long long val = 0;
    if (src_dtype == DT_UINT32)
        val = src.u32;
    else if (src_dtype == DT_INT32)
        val = src.s32;
    else if (src_dtype == DT_UINT16)
        val = src.u16;
    else if (src_dtype == DT_INT16)
        val = src.s16;
    else if (src_dtype == DT_UINT8)
        val = src.u8;
    else if (src_dtype == DT_INT8)
        val = src.s8;
    else if (src_dtype == DT_UINT4)
        val = src.u4;
    else if (src_dtype == DT_INT4)
        val = src.s4;
    scalar_t dst = {.u32 = 0};
    if (dst_dtype == DT_UINT32)
        dst.u32 = MIN(MAX(val, 0), UINT_MAX);
    else if (dst_dtype == DT_INT32)
        dst.s32 = MIN(MAX(val, INT_MIN), INT_MAX);
    else if (dst_dtype == DT_UINT16)
        dst.u16 = MIN(MAX(val, 0), USHRT_MAX);
    else if (dst_dtype == DT_INT16)
        dst.s16 = MIN(MAX(val, SHRT_MIN), SHRT_MAX);
    else if (dst_dtype == DT_UINT8)
        dst.u8 = MIN(MAX(val, 0), UCHAR_MAX);
    else if (dst_dtype == DT_INT8)
        dst.s8 = MIN(MAX(val, CHAR_MIN), CHAR_MAX);
    else if (dst_dtype == DT_UINT4)
        dst.u4 = MIN(MAX(val, UINT4_MIN), UINT4_MAX);
    else if (dst_dtype == DT_INT4)
        dst.s4 = MIN(MAX(val, INT4_MIN), INT4_MAX);
    return dst;
}
scalar_t tpu_fp_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(src_dtype));
    TPUKERNEL_ASSERT(mode == RM_HALF_TO_EVEN || mode == RM_HALF_AWAY_FROM_ZERO ||
                    mode == RM_TOWARDS_ZERO || mode == RM_DOWN ||
                    mode == RM_UP);
    if (dst_dtype == DT_FP16)
        TPUKERNEL_ASSERT(src_dtype != DT_BFP16);
    if (dst_dtype == DT_BFP16)
        TPUKERNEL_ASSERT(src_dtype != DT_FP16);
    scalar_t dst = {.u32 = 0};
    ROUND_MODE convert_mode = convertRoundingMode(mode);
    if (src_dtype == DT_FP32) {
        if (dst_dtype == DT_FP16) {
            fp32 f32 = {.fval = src.f32};
            dst.f16.bits = fp32_to_fp16(f32, convert_mode, false).bits;
        } else if (dst_dtype == DT_BFP16) {
            fp32 f32 = {.fval = src.f32};
            dst.bf16.bits = fp32_to_bf16(f32, convert_mode, false).bits;
        } else if (dst_dtype == DT_FP8E4M3) {
            fp32 f32 = {.fval = src.f32};
            dst.f8e3m4.bits = fp32_to_fp8(f32, false, false, convert_mode);
        } else if (dst_dtype == DT_FP8E5M2) {
            fp32 f32 = {.fval = src.f32};
            dst.f8e5m2.bits = fp32_to_fp8(f32, true, false, convert_mode);
        } else
            dst = src;
    } else if (src_dtype == DT_FP16) {
        if (dst_dtype == DT_FP32) {
            fp16 f16 = {.bits = src.f16.bits};
            dst.f32 = fp16_to_fp32(f16).fval;
        } else
            dst = src;
    } else if (src_dtype == DT_BFP16) {
        if (dst_dtype == DT_FP32) {
            bf16 bf16 = {.bits = src.bf16.bits};
            dst.f32 = bf16_to_fp32(bf16).fval;
        } else
            dst = src;
    } else if (src_dtype == DT_FP8E5M2){
        if (dst_dtype == DT_FP32){
            fp8e5m2 f8e5m2 = {.bits = src.f8e5m2.bits};
            dst.f32 = fp8_to_fp32(f8e5m2.bits, true).fval;
        }else
            dst = src;
    } else if (src_dtype == DT_FP8E4M3){
        if (dst_dtype == DT_FP32){
            fp8e4m3 f8e4m3 = {.bits = src.f8e3m4.bits};
            dst.f32 = fp8_to_fp32(f8e4m3.bits, false).fval;
        }else
            dst = src;
    }
    return dst;
}
scalar_t tpu_fp_to_int_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_int(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(src_dtype));
    TPUKERNEL_ASSERT(mode == RM_HALF_TO_EVEN || mode == RM_HALF_AWAY_FROM_ZERO ||
                    mode == RM_TOWARDS_ZERO || mode == RM_DOWN ||
                    mode == RM_UP);
    scalar_t val = tpu_fp_cast(src, DT_FP32, src_dtype, NO_USE);
    ROUND_MODE convert_mode = convertRoundingMode(mode);
    if (tpu_is_data_type_signed_int(dst_dtype)) {
        fp32 f32val = {.fval = val.f32};
        scalar_t s32 = {.s32 = fp32_to_int(f32val, convert_mode)};
        return tpu_int_cast(s32, dst_dtype, DT_INT32);
    } else {
        fp32 f32val = {.fval = val.f32};
        scalar_t u32 = {.u32 = fp32_to_u32(f32val, convert_mode)};
        return tpu_int_cast(u32, dst_dtype, DT_UINT32);
    }
}
scalar_t tpu_int_to_fp_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode) {
    TPUKERNEL_ASSERT(tpu_is_data_type_fp(dst_dtype));
    TPUKERNEL_ASSERT(tpu_is_data_type_int(src_dtype));
    TPUKERNEL_ASSERT(mode == RM_HALF_TO_EVEN || mode == RM_HALF_AWAY_FROM_ZERO ||
                    mode == RM_TOWARDS_ZERO || mode == RM_DOWN ||
                    mode == RM_UP);
    scalar_t dst = {.u32 = 0};
    long long val = 0;
    ROUND_MODE convert_mode = convertRoundingMode(mode);
    if (src_dtype == DT_UINT32)
        val = src.u32;
    else if (src_dtype == DT_INT32)
        val = src.s32;
    else if (src_dtype == DT_UINT16)
        val = src.u16;
    else if (src_dtype == DT_INT16)
        val = src.s16;
    else if (src_dtype == DT_UINT8)
        val = src.u8;
    else if (src_dtype == DT_INT8)
        val = src.s8;
    if (dst_dtype == DT_FP32){
        dst.f32 = int32_to_fp32(val, convert_mode).fval;
    }
    else if (dst_dtype == DT_FP16)
        dst.f16.bits = int32_to_fp16(val, convert_mode, false).bits;
    else /* if (dst_dtype == DT_BFP16) */
        dst.bf16.bits = int32_to_bf16(val, convert_mode).bits;
    return dst;
}
scalar_t tpu_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode) {
    bool is_dst_int = tpu_is_data_type_int(dst_dtype);
    bool is_src_int = tpu_is_data_type_int(src_dtype);
    ROUND_MODE convert_mode = convertRoundingMode(mode);
    if (is_dst_int && is_src_int)
        return tpu_int_cast(src, dst_dtype, src_dtype);
    else if (!is_dst_int && !is_src_int)
        return tpu_fp_cast(src, dst_dtype, src_dtype, convert_mode);
    else if (is_dst_int && !is_src_int)
        return tpu_fp_to_int_cast(src, dst_dtype, src_dtype, convert_mode);
    else /* if (!is_dst_int && is_src_int) */
        return tpu_int_to_fp_cast(src, dst_dtype, src_dtype, convert_mode);
}
