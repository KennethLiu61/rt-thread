#ifndef SG_TF32_H_
#define SG_TF32_H_

#include "common.h"
#include "cast.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline tf32 fp32_to_tf32(fp32 src) {
  tf32 res;
  res.bits = (uint32_t)(src.bits) >> 13;
  return res;
}

static inline fp32 tf32_to_fp32(tf32 src) {
  fp32 res;
  res.bits = (uint32_t)(src.bits) << 13;
  return res;
}

static fp32 tf32_mul_to_fp32(tf32 a, tf32 b, ROUND_MODE round_mode) {
  fp32 res;
  res.fval = tf32_to_fp32(a).fval * tf32_to_fp32(b).fval;
  return res;
}
#ifdef __cplusplus
}
#endif
#endif