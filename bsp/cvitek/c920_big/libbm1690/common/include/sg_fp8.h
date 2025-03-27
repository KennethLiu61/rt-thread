#ifndef SG_FP8_H_
#define SG_FP8_H_

#include "common.h"
#include "sg_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline bool fp8_nan(uint8_t a, bool is_e5m2) {
  bool res = false;
  if (is_e5m2) {
    DataUnion x = {.u8val = a};
    res = x.f8e5m2val.exp == 0x1f && x.f8e5m2val.frac != 0;
  } else {
    res = (a & 0x7f) == 0x7f;
  }
  return res;
}

static inline bool fp8_lt(uint8_t a, uint8_t b, bool is_e5m2) {
  bool res = true;
  fp32 a32, b32;
  a32 = fp8_to_fp32(a, is_e5m2);
  b32 = fp8_to_fp32(b, is_e5m2);
  if (fp8_nan(a, is_e5m2) || fp8_nan(b, is_e5m2)) {
    // a or b is NAN
    res = (a != b);
  } else {
    res = (a32.fval < b32.fval);
  }
  return res;
}

static inline bool fp8_gt(uint8_t a, uint8_t b, bool is_e5m2) {
  bool res = true;
  fp32 a32, b32;
  a32 = fp8_to_fp32(a, is_e5m2);
  b32 = fp8_to_fp32(b, is_e5m2);
  if (fp8_nan(a, is_e5m2) || fp8_nan(b, is_e5m2)) {
    // a or b is NAN
    res = false;
  } else {
    res = (a32.fval > b32.fval);
  }
  return res;
}

static inline bool fp8_eq(uint8_t a, uint8_t b, bool is_e5m2) {
  bool res = (a == b);
  return res;
}

static inline bool fp8_eq_for_cmp_srch_bin(uint8_t a, uint8_t b, bool is_e5m2) {
  bool res = true;
  fp32 a32, b32;
  a32 = fp8_to_fp32(a, is_e5m2);
  b32 = fp8_to_fp32(b, is_e5m2);
  if (fp8_nan(a, is_e5m2) || fp8_nan(b, is_e5m2)) {
    // a or b is NAN
    res = (a == b);
  } else {
    res = (a32.fval == b32.fval);
  }
  return res;
}

static inline bool fp8_neq(uint8_t a, uint8_t b, bool is_e5m2) {
  bool res = (a != b);
  return res;
}

/// a + b
static inline uint8_t fp8_add(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool r_is_e5m2, bool saturate) {

  fp32 a32, b32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  uint8_t res8;
  if (fp8_nan(a, a_is_e5m2) || fp8_nan(b, b_is_e5m2)){
    res8 = 0x7f;
    return res8;
  }
  Double tmp;
  tmp.double_val = (double)a32.fval + (double)b32.fval;
  res8 = double_to_fp8(tmp, r_is_e5m2, saturate, ROUND_HALF_TO_EVEN);
  if (fp8_nan(res8, r_is_e5m2)){
    res8 = 0x7f;
  }

  return res8;
}
static inline fp16 fp8_add_to_fp16(uint8_t a, uint8_t b, bool a_is_e5m2,bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if (fp8_nan(a, a_is_e5m2) || fp8_nan(b, b_is_e5m2) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res16.bits = 0x7fff;
    return res16;
  }
 // Double tmp;
  //tmp.double_val = (double)a32.fval - (double)b32.fval;
  res32.fval = a32.fval + b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
static inline fp32 fp8_add_to_fp32(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res32.bits = 0x7fffffff;
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
static inline fp16 fp8_add_fp16_to_fp16(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res16.bits = 0x7fff;
    return res16;
  }
  Double tmp;
  tmp.double_val = (double)a32.fval + (double)b32.fval;
  res16 = double_to_fp16(tmp, saturate, ROUND_HALF_TO_EVEN);
 // res32.fval = a32.fval + b32.fval;
 // res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
static inline fp32 fp8_add_fp16_to_fp32(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign != b32.format.sign))){
    res32.bits = 0x7fffffff;
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
static inline fp32 fp8_add_fp32_to_fp32(uint8_t a, fp32 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  if (b.format.exp == 0) {
     b.format.frac = 0;
  }
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b.format.exp == 0xff && b.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b.format.exp == 0xff && b.format.frac == 0) && (a32.format.sign != b.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = a32.fval + b.fval;
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
static inline uint8_t fp8_sub(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool r_is_e5m2, bool saturate) {
  fp32 a32, b32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  uint8_t res8;
  if (fp8_nan(a, a_is_e5m2) || fp8_nan(b, b_is_e5m2) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res8 = 0x7f;
    return res8;
  }
  Double tmp;
  tmp.double_val = (double)a32.fval - (double)b32.fval;
  res8 = double_to_fp8(tmp, r_is_e5m2, saturate, ROUND_HALF_TO_EVEN);
  if (fp8_nan(res8, r_is_e5m2)){
    res8 = 0x7f;
  }

  return res8;
}
static inline fp16 fp8_sub_to_fp16(uint8_t a, uint8_t b, bool a_is_e5m2,bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if (fp8_nan(a, a_is_e5m2) || fp8_nan(b, b_is_e5m2) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res16.bits = 0x7fff;
    return res16;
  }
  res32.fval = a32.fval - b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
static inline fp32 fp8_sub_to_fp32(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res32.bits = 0x7fffffff;
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
static inline fp16 fp16_sub_fp8_to_fp16(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res16.bits = 0x7fff;
    return res16;
  }
  Double tmp;
  tmp.double_val = (double)b32.fval - (double)a32.fval;
  res16 = double_to_fp16(tmp, saturate, ROUND_HALF_TO_EVEN);
  //res32.fval = b32.fval - a32.fval;
  //res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, saturate);
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
static inline fp32 fp16_sub_fp8_to_fp32(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.sign == b32.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = b32.fval - a32.fval;
  if (res32.format.exp == 0xff && res32.format.frac == 0 && saturate){
    res32.bits = 0x7f7fffff | (res32.format.sign << 31);
  }
  if (res32.format.exp == 0xff && res32.format.frac != 0){
    res32.bits = 0x7fffffff;
  }

  return res32;
}
static inline fp32 fp32_sub_fp8_to_fp32(uint8_t a, fp32 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b.format.exp == 0xff && b.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b.format.exp == 0xff && b.format.frac == 0) && (a32.format.sign == b.format.sign))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = b.fval - a32.fval;
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
static inline uint8_t fp8_mul(uint8_t a, uint8_t b, bool a_is_e5m2,bool b_is_e5m2, bool r_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  uint8_t res8;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) || ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res8 = 0x7f;
    return res8;
  }
  res32.fval = a32.fval * b32.fval;
  res8 = fp32_to_fp8(res32, r_is_e5m2, saturate, ROUND_HALF_TO_EVEN);
  if (fp8_nan(res8, r_is_e5m2)){
    res8 = 0x7f;
  }
  return res8;
}
static inline fp16 fp8_mul_to_fp16(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) || ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res16.bits = 0x7fff;
    return res16;
  }
  res32.fval = a32.fval * b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, false);
  if (res16.format.exp == 0x1f && res16.format.frac == 0 && saturate){
    res16.bits = 0x7bff | (res16.format.sign << 15);
  }
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }
  return res16;
}
static inline fp32 fp8_mul_to_fp32(uint8_t a, uint8_t b, bool a_is_e5m2, bool b_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp8_to_fp32(b, b_is_e5m2);
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
static inline fp16 fp8_mul_fp16_to_fp16(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  fp16 res16;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  b32 = fp16_to_fp32(b);
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b32.format.exp == 0xff && b32.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b32.format.exp == 0 && b32.format.frac == 0)) || ((b32.format.exp == 0xff && b32.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res16.bits = 0x7fff;
    return res16;
  }
  res32.fval = a32.fval * b32.fval;
  res16 = fp32_to_fp16(res32, ROUND_HALF_TO_EVEN, false);
  if (res16.format.exp == 0x1f && res16.format.frac == 0 && saturate){
    res16.bits = 0x7bff | (res16.format.sign << 15);
  }
  if (res16.format.exp ==31 && res16.format.frac != 0){
    res16.bits = 0x7fff;
  }

  return res16;
}
static inline fp32 fp8_mul_fp16_to_fp32(uint8_t a, fp16 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, b32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
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
static inline fp32 fp8_mul_fp32_to_fp32(uint8_t a, fp32 b, bool a_is_e5m2, bool saturate) {
  fp32 a32, res32;
  a32 = fp8_to_fp32(a, a_is_e5m2);
  if (b.format.exp == 0) {
    b.format.frac = 0;
  }
  if ((a32.format.exp == 0xff && a32.format.frac != 0) || (b.format.exp == 0xff && b.format.frac != 0) || ((a32.format.exp == 0xff && a32.format.frac == 0) && (b.format.exp == 0 && b.format.frac == 0)) || ((b.format.exp == 0xff && b.format.frac == 0) && (a32.format.exp == 0 && a32.format.frac == 0))){
    res32.bits = 0x7fffffff;
    return res32;
  }
  res32.fval = a32.fval * b.fval;
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
/// max(a, b)
static inline uint8_t fp8_max(uint8_t a, uint8_t b, bool is_e5m2) {
  uint8_t res8 = fp8_gt(a, b, is_e5m2) ? a : b;
  return res8;
}

/// min(a, b)
static inline uint8_t fp8_min(uint8_t a, uint8_t b, bool is_e5m2) {
  uint8_t res8 = fp8_lt(a, b, is_e5m2) ? a : b;
  return res8;
}

static inline uint8_t fp8_abs(uint8_t a) {
  return a & 0x7f;
}

#ifdef __cplusplus
}
#endif

#endif
