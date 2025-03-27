#ifndef SG_FP20_H_
#define SG_FP20_H_

#include "common.h"
#include "sg_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MASK_12BIT ((u32)0xfff)
#define MASK_20BIT ((u32)0xfffff)

static inline fp32 fp20_to_fp32(fp20 val) {
    fp32 res = {
        .format = {
            .frac = (uint32_t)(val.format.frac << 12),
            .exp = val.format.exp,
            .sign = val.format.sign,
        }
    };
    return res;
}

static inline fp20 fp32_to_fp20(fp32 val) {
    fp20 res = {
        .format = {
            .frac = (uint32_t)(val.format.frac >> 12),
            .exp = val.format.exp,
            .sign = val.format.sign,
        }
    };
    return res;
}

/// a + b
static inline fp20 fp20_add(fp20 a, fp20 b) {
  fp32 a32, b32, res32;
  a32 = fp20_to_fp32(a);
  b32 = fp20_to_fp32(b);
  res32 = fp32_add(a32, b32, false);
  fp20 res20 = fp32_to_fp20(res32);
  return res20;
}

/// a - b
static inline fp20 fp20_sub(fp20 a, fp20 b) {
  fp32 a32, b32, res32;
  a32 = fp20_to_fp32(a);
  b32 = fp20_to_fp32(b);
  res32 = fp32_sub(a32, b32, false);
  fp20 res20 = fp32_to_fp20(res32);
  return res20;
}

/// a * b
static inline fp20 fp20_mul(fp20 a, fp20 b) {
  fp32 a32, b32, res32;
  a32 = fp20_to_fp32(a);
  b32 = fp20_to_fp32(b);
  res32 = fp32_mul(a32, b32, false);
  fp20 res20 = fp32_to_fp20(res32);
  return res20;
}

/// a > b
static inline bool fp20_gt(fp20 a, fp20 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp20_to_fp32(a);
  b32 = fp20_to_fp32(b);
  res = fp32_gt(a32, b32);
  return res;
}

/// a < b
static inline bool fp20_lt(fp20 a, fp20 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp20_to_fp32(a);
  b32 = fp20_to_fp32(b);
  res = fp32_lt(a32, b32);
  return res;
}

static inline fp20 fp20_min(fp20 a, fp20 b) {
    fp20 res = fp20_lt(a, b) ? a : b;
    if (res.format.exp == 0) {
        res.format.frac = 0;
    }
    return res;
}

static inline fp20 fp20_max(fp20 a, fp20 b) {
    fp20 res = fp20_gt(a, b) ? a : b;
    if (res.format.exp == 0) {
        res.format.frac = 0;
    }
    return res;
}

static inline bool fp20_eq(fp20 a, fp20 b) {
    return a.bits == b.bits;
}

static inline bool fp20_neq(fp20 a, fp20 b) {
    return a.bits != b.bits;
}

static inline fp20 lossy_compress_fp20(fp32 val) {
    fp20 res = {
        .format = {
            .frac = (uint32_t)(val.format.frac >> 12),
            .exp = val.format.exp,
            .sign = val.format.sign,
        }
    };
    return res;
}

static inline fp32 lossy_decompress_fp20(fp20 val) {
    fp32 res = {
        .format = {
            .frac = (uint32_t)(val.format.frac << 12),
            .exp = val.format.exp,
            .sign = val.format.sign,
        }
    };
    return res;
}

static inline fp32 lossy_compress_fp32_by_fp20(fp32 val) {
    fp32 res = {
        .format = {
            .frac = val.format.frac & (~MASK_12BIT),
            .exp = val.format.exp,
            .sign = val.format.sign,
        }
    };
    return res;
}

static inline void set_data_to_fp20_block(char* ptr, int j, fp20 x) {
    char* _ptr = ptr + (j >> 1) * 5 + (j & 1) * 2;
    int mask = MASK_20BIT;
    int valid_digits = x.bits & mask;
    if (j & 1) {
        mask <<= 4;
        valid_digits <<= 4;
    }
    *(uint32_t*)_ptr &= (~mask);
    *(uint32_t*)_ptr |= valid_digits;
}

static inline fp20 get_data_from_fp20_block(const char* ptr, int j) {
    const char* _ptr = ptr + (j >> 1) * 5 + (j & 1) * 2;
    uint32_t valid_digits = *(uint32_t*)_ptr;
    if (j & 1) {
        valid_digits >>= 4;
    }
    valid_digits &= MASK_20BIT;
    return {.bits = valid_digits};
}

#ifdef __cplusplus
}
#endif

#endif
