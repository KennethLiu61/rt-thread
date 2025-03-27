#ifndef CDMA_REG_VALUE_H
#define CDMA_REG_VALUE_H
#include "common.h"
typedef enum {
    CDMA_SEND = 0,
    CDMA_READ = 1,
    CDMA_WRITE = 2,
    CDMA_GENERAL = 3,
    CDMA_RECV = 4,
    CDMA_LOSSY_COMPRESS = 5,
    CDMA_LOSSY_DECOMPRESS = 6,
    CDMA_SYS  = 7,
    CDMA_TCP_SEND = 8,
    CDMA_TCP_RECV = 9,
    CDMA_FAKE_ALL_REDUCE = 10,
    CDMA_FAKE_P2P =11,
} CDMA_TSK_TYPE;

typedef enum {
  CDMA_SYS_END = 0,
  CDMA_SYS_NOP = 1,
  CDMA_SYS_TRWR = 2,
  CDMA_SYS_TX_SEND_MSG = 3,
  CDMA_SYS_TX_WAIT_MSG = 4,
  CDMA_SYS_RX_SEND_MSG = 5,
  CDMA_SYS_RX_WAIT_MSG = 6
} CDMA_SYS_TYPE;

typedef enum {
  CDMA_DTYPE_INT8 = 0,
  CDMA_DTYPE_FP16 = 1,
  CDMA_DTYPE_FP32 = 2,
  CDMA_DTYPE_INT16 = 3,
  CDMA_DTYPE_INT32 = 4,
  CDMA_DTYPE_BF16 = 5,
  CDMA_DTYPE_FP20 = 6,
  CDMA_DTYPE_FORMAT_NUM,
} CDMA_DTYPE;

static inline int get_cdma_format_type_len(int t) {
  switch (t) {
    case CDMA_DTYPE_INT8:
      return 1;
    case CDMA_DTYPE_FP16:
    case CDMA_DTYPE_BF16:
    case CDMA_DTYPE_INT16:
      return 2;
    case CDMA_DTYPE_FP32:
    case CDMA_DTYPE_INT32:
      return 4;
  }
  return 0;
}

static inline PREC get_cdma_format_precision(int cdma_format) {
  switch (cdma_format) {
    case CDMA_DTYPE_FP32:
      return FP32;
    case CDMA_DTYPE_FP16:
      return FP16;
    case CDMA_DTYPE_BF16:
      return BFP16;
    case CDMA_DTYPE_INT32:
      return INT32;
    case CDMA_DTYPE_INT16:
      return INT16;
    case CDMA_DTYPE_INT8:
      return INT8;
    case CDMA_DTYPE_FP20:
      return FP20;
    default:
      ASSERT(0);
      return INT8;
  }
}

static inline CDMA_DTYPE get_cdma_format_from_precision(PREC prec) {
  switch (prec) {
    case FP32:
      return CDMA_DTYPE_FP32;
    case FP16:
      return CDMA_DTYPE_FP16;
    case BFP16:
      return CDMA_DTYPE_BF16;
    case INT32:
      return CDMA_DTYPE_INT32;
    case INT16:
      return CDMA_DTYPE_INT16;
    case INT8:
      return CDMA_DTYPE_INT8;
    case FP20:
      return CDMA_DTYPE_FP20;
    default:
      ASSERT(0);
      return CDMA_DTYPE_INT8;
  }
}

typedef enum {
  CDMA_OPCODE_NONE = 0,
  CDMA_OPCODE_MUL = 1,
  CDMA_OPCODE_MAX = 2,
  CDMA_OPCODE_MIN = 3,
  CDMA_OPCODE_ADD = 4,
  CDMA_OPCODE_NUM
} CDMA_OPCODE;

typedef enum {
  PSUM_OP_WO = 0,
  PSUM_OP_WR = 1,
} PSUM_OP;

typedef enum {
  CDMA_ROUTE_AXI_RN = 0,
} CDMA_ROUTE_TYPE;

#endif // CDMA_REG_VALUE_H
