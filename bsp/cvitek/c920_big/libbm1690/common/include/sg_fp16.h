#ifndef SG_FP16_H_
#define SG_FP16_H_

#include "common.h"
#include "fp16.h"
#include "cast.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline bool fp32_lt(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits != b.bits);
  } else if (a.format.exp == 0 && b.format.exp == 0) {
    res = false;
  } else {
    res = (a.fval < b.fval);
  }
  return res;
}
static inline fp32 fp32_add(fp32 a, fp32 b, bool saturate) {
  fp32 res;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0){
    b.format.frac = 0;
  }
  if (a.bits == 0x80000000 && b.bits == 0x80000000) {
    res.bits = 0x80000000;
    return res;
  }
  res.fval = a.fval + b.fval;
  if (res.format.exp == 0xff && res.format.frac == 0 && saturate){
    res.bits = 0x7f7fffff | (res.format.sign << 31);
  }
  if (res.format.exp == 0) {
    res.format.frac = 0;
  }
  if (res.format.exp == 0xff && res.format.frac != 0){
    res.bits = 0x7fffffff;
  }
  return res;
}
static inline fp32 fp32_sub(fp32 a, fp32 b, bool saturate) {
  fp32 res;
  if (a.format.exp == 0) {
    a.format.frac = 0;
}
  if (b.format.exp == 0){
    b.format.frac = 0;
  }
  if (a.bits == 0x80000000 && b.bits == 0x0) {
    res.bits = 0x80000000;
    return res;
  }
  res.fval = a.fval - b.fval;
  if (res.format.exp == 0xff && res.format.frac == 0 && saturate){
    res.bits = 0x7f7fffff | (res.format.sign << 31);
  }
  if (res.format.exp == 0) {
    res.format.frac = 0;
  }
  if (res.format.exp == 0xff && res.format.frac != 0){
    res.bits = 0x7fffffff;
  }
  return res;
}

static inline bool fp32_gt(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0) ||
      (a.format.exp == 0 && b.format.exp == 0)) {
    // a or b is NAN
    res = false;
  } else {
    res= (a.fval > b.fval);
  }
  return res;
}

static inline bool fp32_eq(fp32 a, fp32 b) {
  bool res;
#if defined(__bm1684xe__)
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits == b.bits);
  } else {
    res= (a.fval == b.fval);
  }
#else
  res = (a.bits == b.bits);
#endif
  return res;
}

static inline bool fp32_eq_for_cmp_srch_bin(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits == b.bits);
  } else {
    res= (a.fval == b.fval);
  }
  return res;
}

static inline fp32 fp32_max(fp32 a, fp32 b) {
  if (a.format.exp == 0) a.bits &= 0x80000000;
  if (b.format.exp == 0) b.bits &= 0x80000000;
  fp32 res32;
  if (a.format.exp == 0 && b.format.exp == 0){
    res32 = b;
  } else {
    res32 = fp32_gt(a, b) ? a : b;
  }
  return res32;
}

static inline fp32 fp32_min(fp32 a, fp32 b) {
  if (a.format.exp == 0) a.bits &= 0x80000000;
  if (b.format.exp == 0) b.bits &= 0x80000000;
  fp32 res32;
  if (a.format.exp == 0 && b.format.exp == 0){
    res32 = b;
  } else {
    res32 = fp32_lt(a, b) ? a : b;
  }
  return res32;
}

/// a + b
static inline fp16 fp16_add(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;

    return res16;
  }
  if (a.bits == 0x8000 && b.bits == 0x8000) {
    res16.bits = 0x8000;
    return res16;
  }
  res32.fval = a32.fval + b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp == 31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }
  return res16;
}
static inline fp32 fp16_add_to_fp32(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  if (a.bits == 0x8000 && b.bits == 0x8000) {
    res32.bits = 0x80000000;
    return res32;
  }
  res32.fval = a32.fval + b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0)  {
    res32.format.frac = 0;
  }
  return res32;
}
/// a - b
#if defined(__bm1684xe__)
static inline fp16 fp16_sub(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  fp16 res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  return res16;
}
#else
static inline fp16 fp16_sub(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;
    return res16;
  }
  if (a.bits == 0x8000 && b.bits == 0x0) {
    res16.bits = 0x8000;
    return res16;
  }
  res32.fval = a32.fval - b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp == 31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
#endif
static inline fp32 fp16_sub_to_fp32(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  if (a.bits == 0x8000 && b.bits == 0x0) {
    res32.bits = 0x80000000;
    return res32;
  }
  res32.fval = a32.fval - b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0) {
    res32.format.frac = 0;
  }
  return res32;
}

/// a * b
#if defined(__bm1684xe__)
static inline fp16 fp16_mul(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  fp16 res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  return res16;
}
#else
static inline fp16 fp16_mul(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) ||
      (b32.format.exp == 0xff && b32.format.frac != 0) ||
      ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) ||
      ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;
    return res16;
  }
  res32.fval = a32.fval * b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp == 31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
#endif
static inline fp32 fp16_mul_to_fp32(fp16 a, fp16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) || ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = a32.fval * b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0) {
    res32.format.frac = 0;
  }
  return res32;
}
/// a > b
static inline bool fp16_gt(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = false;
  } else {
    res = (a32.fval > b32.fval);
  }
  return res;
}

/// a < b
static inline bool fp16_lt(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = (a.bits != b.bits);
  } else {
    res = (a32.fval < b32.fval);
  }
  return res;
}

/// a == b
static inline bool fp16_eq(fp16 a, fp16 b) {
    bool res;
#if defined(__bm1684xe__)
    fp32 a32, b32;
    a32 = fp16_to_fp32(a);
    b32 = fp16_to_fp32(b);
    if ((a.format.exp == 31 && a.format.frac != 0) ||
        (b.format.exp == 31 && b.format.frac != 0)) {
        res = (a.bits == b.bits);
    } else {
        res = (a32.fval == b32.fval);
    }
#else
    res = (a.bits == b.bits);
#endif
    return res;
}

static inline bool fp16_eq_for_cmp_srch_bin(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = (a.bits == b.bits);
  } else {
    res = (a32.fval == b32.fval);
  }
  return res;
}

/// a != b
static inline bool fp16_neq(fp16 a, fp16 b) {
    bool res;
#if defined(__bm1684xe__)
    fp32 a32, b32;
    a32 = fp16_to_fp32(a);
    b32 = fp16_to_fp32(b);
    if ((a.format.exp == 31 && a.format.frac != 0) ||
        (b.format.exp == 31 && b.format.frac != 0)) {
        res = (a.bits != b.bits);
    } else {
        res = (a32.fval != b32.fval);
    }
#else
    res = (a.bits != b.bits);
#endif
    return res;
}

/// a <= b
static inline bool fp16_leq(fp16 a, fp16 b) {
  return !fp16_gt(a, b);
}

/// a >= b
static inline bool fp16_geq(fp16 a, fp16 b) {
  return !fp16_lt(a, b);
}

/// max(a, b)
static inline fp16 fp16_max(fp16 a, fp16 b) {
  fp16 res16 = fp16_gt(a, b) ? a : b;
  return res16;
}

/// min(a, b)
static inline fp16 fp16_min(fp16 a, fp16 b) {
  fp16 res16 = fp16_lt(a, b) ? a : b;
  return res16;
}

static inline fp16 fp16_abs(fp16 a) {
  fp16 res16 = {.bits = (uint16_t)((uint32_t)(a.bits) & (uint32_t)0x7fff)};
  return res16;
}

/// a + b
static inline bf16 bf16_add(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  bf16 res16;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;
    return res16;
  }
  if (a.bits == 0x8000 && b.bits == 0x8000){
    res16.bits = 0x8000;
    return res16;
  }
  res32.fval = a32.fval + b32.fval;
  res16 = fp32_to_bf16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval + (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  if (res16.format.exp == 0xff && res16.format.frac == 0 && saturate){
    res16.bits = 0x7f7f | (res16.format.sign << 15);
  }
  return res16;
}
static inline fp32 bf16_add_to_fp32(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  if (a.bits == 0x8000 && b.bits == 0x8000){
    res32.bits = 0x80000000;
    return res32;
  }

  res32.fval = a32.fval + b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0) {
    res32.format.frac = 0;
  }

  return res32;
}
/// a - b
#if defined(__bm1684xe__)
static inline bf16 bf16_sub(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  bf16 res16 = fp32_to_bf16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval - (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  return res16;
}
#else
static inline bf16 bf16_sub(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  bf16 res16;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;
    return res16;
  }
  if (a.bits == 0x8000 && b.bits == 0x0){
    res16.bits = 0x8000;
    return res16;
  }
  res32.fval = a32.fval - b32.fval;
  res16 = fp32_to_bf16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval - (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  if (res16.format.exp == 0xff && res16.format.frac == 0 && saturate){
    res16.bits = 0x7f7f | (res16.format.sign << 15);
  }

  return res16;
}
#endif
static inline fp32 bf16_sub_to_fp32(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  if (a.bits == 0x8000 && b.bits == 0x0){
    res32.bits = 0x80000000;
    return res32;
  }

  res32.fval = a32.fval - b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0) {
    res32.format.frac = 0;
  }

  return res32;
}

/// a * b
#if defined(__bm1684xe__)
static inline bf16 bf16_mul(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  bf16 res16 = fp32_to_bf16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval * (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  return res16;
}
#else
static inline bf16 bf16_mul(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  bf16 res16;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) ||
      (b32.format.exp == 0xff && b32.format.frac != 0) ||
      ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) ||
      ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
#if defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
    if(saturate)
      res16.bits = 0;
    else
#endif
    res16.bits = 0x7fff;
    return res16;
  }
  res32.fval = a32.fval * b32.fval;
  res16 = fp32_to_bf16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval * (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  if (res16.format.exp == 0xff && res16.format.frac == 0 && saturate){
    res16.bits = 0x7f7f | (res16.format.sign << 15);
  }

  return res16;
}
#endif
static inline fp32 bf16_mul_to_fp32(bf16 a, bf16 b, bool saturate) {
  fp32 a32, b32, res32;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) || ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = a32.fval * b32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }
  if (res32.format.exp == 0) {
    res32.format.frac = 0;
  }

  return res32;
}

static inline fp32 fp32_mul(fp32 a, fp32 b, bool saturate) {
  fp32 res;
  if (a.format.exp == 0) {
    a.format.frac = 0;
  }
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  res.fval = a.fval * b.fval;
  if (res.format.exp == 0xff && res.format.frac == 0 && saturate){
    res.bits = 0x7f7fffff | (res.format.sign << 31);
  }
  if (res.format.exp == 0xff && res.format.frac != 0){
    res.bits = 0x7fffffff;
  }
  if (res.format.exp == 0) {
    res.format.frac = 0;
  }

  return res;
}
/// a > b
static inline bool bf16_gt(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0) ||
      (a.format.exp == 0 && b.format.exp == 0)) {
    res = false;
  } else {
    res = (a32.fval > b32.fval);
  }
  return res;
}

/// a < b
static inline bool bf16_lt(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits != b.bits);
  } else if (a.format.exp == 0 && b.format.exp == 0) {
    res = false;
  } else {
    res = (a32.fval < b32.fval);
  }
  return res;
}

/// a == b
static inline bool bf16_eq(bf16 a, bf16 b) {
  bool res;
#if defined(__bm1684xe__)
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = (a.bits == b.bits);
  } else {
    res = (a32.fval == b32.fval);
  }
#else
  res = (a.bits == b.bits);
#endif
  return res;
}

static inline bool bf16_eq_for_cmp_srch_bin(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = (a.bits == b.bits);
  } else {
    res = (a32.fval == b32.fval);
  }
  return res;
}

/// a != b
static inline bool bf16_neq(bf16 a, bf16 b) {
  bool res;
#if defined(__bm1684xe__)
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = (a.bits != b.bits);
  } else {
    res = (a32.fval != b32.fval);
  }
#else
  res = (a.bits != b.bits);
#endif
  return res;
}

/// a <= b
static inline bool bf16_leq(bf16 a, bf16 b) {
  return !bf16_gt(a, b);
}

/// a >= b
static inline bool bf16_geq(bf16 a, bf16 b) {
  return !bf16_lt(a, b);
}

/// max(a, b)
static inline bf16 bf16_max(bf16 a, bf16 b) {
  if (a.format.exp == 0) a.bits &= 0x8000;
  if (b.format.exp == 0) b.bits &= 0x8000;
  bf16 res16;
  if (a.format.exp == 0 && b.format.exp == 0){
    res16 = b;
  } else {
    res16 = bf16_gt(a, b) ? a : b;
  }
  return res16;
}

/// min(a, b)
static inline bf16 bf16_min(bf16 a, bf16 b) {
  if (a.format.exp == 0) a.bits &= 0x8000;
  if (b.format.exp == 0) b.bits &= 0x8000;
  bf16 res16;
  if (a.format.exp == 0 && b.format.exp == 0){
    res16 = b;
  } else {
    res16 = bf16_lt(a, b) ? a : b;
  }
  return res16;
}

static inline bf16 bf16_abs(bf16 a) {
  bf16 res16 = {.bits = (uint16_t)((uint32_t)(a.bits) & (uint32_t)0x7fff)};
  return res16;
}

#ifdef __cplusplus
}
#endif

#endif
