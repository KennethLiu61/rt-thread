#include "assert.h"
#include <cmath>
#include <algorithm>
#include "common.h"
#include "cast_util.h"
#include "conv_util.h"
#include "math.h"

#ifndef __x86_64__
static float erf(float &x) {
  unsigned int ERF_COEFF[] = {
      0xbfa1fc4e,
      0x3f8000c7,
      0x3ebf88fb,
      0x3dc636c9,
      0xbe3ec24c,
      0x3e8ec7cc,
      0xbf914e5d,
      0x3fbe87b0,
      0xbf527892,
      0x3e2ef945
  };
  float t = 1.f / (1.f + 0.5f * std::abs(x));
  const int len = sizeof(ERF_COEFF) / sizeof(ERF_COEFF[0]);
  float ts[len];
  ts[0] = 1.f;
  for (int i = 1; i < len; ++i)
    ts[i] = ts[i - 1] * t;
  float acc = - x * x;
  for (int i = 0; i < len; ++i)
    acc += ((float &)ERF_COEFF[i]) * ts[i];
  float tau = t * std::exp(acc);
  return x >= 0.f ? 1.f - tau : tau - 1.f;
}
#endif

static inline float sigmoid(float x) {
  return 1 / (1 + std::exp(-x));
}

static inline float active_per_elem(float x, sg_active_type_t active_type, const float* coeff) {
  float res = 0.f;
  if (active_type == ACTIVE_SQRT) res = std::sqrt(x);
  else if (active_type == ACTIVE_RSQRT) res = 1 / std::sqrt(x);
  else if (active_type == ACTIVE_SQUARE) res = std::pow(x, 2);
  else if (active_type == ACTIVE_EXP) res = std::exp(x);
  else if (active_type == ACTIVE_ELU) res = x > 0 ? x : (std::exp(x) - 1);
  else if (active_type == ACTIVE_TANH) res = std::tanh(x);
  else if (active_type == ACTIVE_TAN) res = std::tan(x);
  else if (active_type == ACTIVE_SIGMOID) res = sigmoid(x);
  else if (active_type == ACTIVE_ERF) res = erf((float)x);
  else if (active_type == ACTIVE_SILU) res = x / (1 + std::exp(-x));
  else if (active_type == ACTIVE_LN) res = std::log(x);
  else if (active_type == ACTIVE_SOFT_SIGN) res = x / (1 + std::abs(x));
  else if (active_type == ACTIVE_HSIGMOID) {
    const float alpha = coeff[1], beta = coeff[0];
    res = std::max(0.0f, std::min(1.0f, alpha * x + beta));
  }
  else if (active_type == ACTIVE_GELU) {
    res = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
  }
  else if (active_type == ACTIVE_GELU_FAST) {
    res = 0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)));
  }
  else if (active_type == ACTIVE_GELU_FAST2) {
    res = x * sigmoid(1.702 * x);
  }
  else if (active_type == ACTIVE_HSWISH) res = x * std::max(0.0f, std::min(1.0f, x / 6 + 0.5f));
  else if (active_type == ACTIVE_SIN) res = std::sin(x);
  else if (active_type == ACTIVE_COS) res = std::cos(x);
  else if (active_type == ACTIVE_ARCTANH) res = std::atanh(x);
  else assert(0);

  return res;
}

template <typename T>
void native_active(
  const T* input,
  T* output,
  const int* shape,
  int dims,
  sg_active_type_t active_type,
  const float* coeff)
{
  uint64_t numel = 1;
  for (int i = 0; i < dims; i++) {
    numel *= shape[i];
  }
  for (uint64_t i = 0; i < numel; i++) {
    output[i] = fp_cast<T>(active_per_elem(fp_cast<float>(input[i]), active_type, coeff));
  }
}

void native_active(
  const void* input,
  void* output,
  const int* shape,
  int dims,
  sg_active_type_t active_type,
  const float* coeff,
  sg_data_type_t dtype)
{
  switch (dtype) {
  case SG_DTYPE_FP32:
    native_active<float>((float*)input, (float*)output, shape, dims, active_type, coeff);
    break;
  case SG_DTYPE_FP16:
    native_active<fp16>((fp16*)input, (fp16*)output, shape, dims, active_type, coeff);
    break;
  case SG_DTYPE_BFP16:
    native_active<bf16>((bf16*)input, (bf16*)output, shape, dims, active_type, coeff);
    break;
  default: assert(0);
  }
}