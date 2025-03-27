#ifndef __SDMA_REG_VALUE_H__
#define __SDMA_REG_VALUE_H__

////////////////////// descriptor value //////////////////////////
typedef enum {
  SDMA_TENSOR = 0,
  SDMA_RESERVED0 = 1,
  SDMA_RESERVED1 = 2,
  SDMA_GENERAL = 3,
  SDMA_RESERVED2 = 4,
  SDMA_RESERVED3 = 5,
  SDMA_SYS = 6,
  SDMA_GATHER = 7,
  SDMA_SCATTER = 8,
  SDMA_RESERVED4 = 9,
  SDMA_RESERVED5 = 10,
  SDMA_RESERVED6 = 11,
  SDMA_LOSSY_COMPRESS = 12,
  SDMA_LOSSY_DECOMPRESS = 13,
  SDMA_TYPE_NUM = 14,
} SDMA_TYPE;

typedef enum {
  SDMA_FUNC_NONE = 0,
  SDMA_FUNC_TRANS = 1,  // NC Transpose
  SDMA_FUNC_RESERVED0 = 2,
  SDMA_FUNC_RESERVED1 = 3,
  SDMA_FUNC_RESERVED2 = 4,
} SDMA_FUNC_TYPE;

typedef enum {
  SDMA_INT8 = 0,
  SDMA_FP16 = 1,
  SDMA_FP32 = 2,
  SDMA_INT16 = 3,
  SDMA_INT32 = 4,
  SDMA_BF16 = 5,
  SDMA_FP20 = 6,
  SDMA_FP8_E4M3 = 7,
  SDMA_FP8_E5M2 = 8,
  SDMA_FORMAT_NUM,
} SDMA_FORMAT;

typedef enum {
  // S: systerm memory: dram and l2sram
  SDMA_S2L = 0,
  SDMA_L2S = 1,
  SDMA_S2S = 2,
  SDMA_L2L = 3,
  SDMA_DIR_NUM,
} SDMA_DIRECTION;

#define SDMA_FORMAT_IS_FLOAT(format) \
    ((format) == SDMA_FP32 || (format) == SDMA_FP16 || (format) == SDMA_BF16)

typedef enum {
  SDMA_SYS_END = 0,
  SDMA_SYS_NOP = 1,
  SDMA_SYS_TRWR = 2,
  SDMA_SYS_SEND_MSG = 3,
  SDMA_SYS_WAIT_MSG = 4
} SDMA_SYS_TYPE;

typedef enum {
  SDMA_ARE_PSUM_WO = 0,
  SDMA_ARE_PSUM_RW = 1
} SDMA_ARE_PSUM_OP_TYPE;

typedef enum {
  SDMA_ARE_NOP = 0,
  SDMA_ARE_MUL = 1,
  SDMA_ARE_MAX = 2,
  SDMA_ARE_MIN = 3,
  SDMA_ARE_ADD = 4,
} SDMA_ARE_OPCODE_TYPE;

typedef enum {
  SDMA_ARE_FP32 = 0,
  SDMA_ARE_FP20 = 1,
  SDMA_ARE_FP16 = 2,
  SDMA_ARE_BF16 = 3,
  SDMA_ARE_INT32 = 4,
  SDMA_ARE_TYPE_NUM,
} SDMA_ARE_TYPE;

static inline int get_sdma_are_dtype(int dma_type) {
  if (dma_type == SDMA_FP32) {
    return SDMA_ARE_FP32;
  } else if (dma_type == SDMA_FP16) {
    return SDMA_ARE_FP16;
  } else if (dma_type == SDMA_BF16) {
    return SDMA_ARE_BF16;
  } else if (dma_type == SDMA_INT32) {
    return SDMA_ARE_INT32;
  } else if (dma_type == SDMA_FP20) {
    return SDMA_ARE_FP20;
  }
  ASSERT(0);
  return -1;
}

typedef enum {
  SDMA_SCATTER_INPLACE  = 0,
  SDMA_SCATTER_ADD      = 1,
} SDMA_SCATTER_TYPE;

static inline int get_sdma_format_type_len(int t) {
  switch (t) {
    case SDMA_INT8:
    case SDMA_FP8_E4M3:
    case SDMA_FP8_E5M2:
      return 1;
    case SDMA_FP16:
    case SDMA_BF16:
    case SDMA_INT16:
      return 2;
    case SDMA_FP32:
    case SDMA_INT32:
      return 4;
  }
  return 0;
}

static inline PREC get_sdma_format_precision(int sdma_format) {
  switch (sdma_format) {
    case SDMA_FP32:
      return FP32;
    case SDMA_FP16:
      return FP16;
    case SDMA_BF16:
      return BFP16;
    case SDMA_INT32:
      return INT32;
    case SDMA_INT16:
      return INT16;
    case SDMA_INT8:
      return INT8;
    case SDMA_FP20:
      return FP20;
    case SDMA_FP8_E4M3:
    case SDMA_FP8_E5M2:
      return FP8;
    default:
      ASSERT(0);
      return INT8;
  }
}

#endif  // __SDMA_REG_VALUE_H__
