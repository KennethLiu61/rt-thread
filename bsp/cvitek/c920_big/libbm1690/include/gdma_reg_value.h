#ifndef __GDMA_REG_VALUE_H__
#define __GDMA_REG_VALUE_H__

////////////////////// descriptor value //////////////////////////
typedef enum {
  GDMA_TENSOR = 0,
  GDMA_MATRIX = 1,
  GDMA_FILTER = 2,
  GDMA_GENERAL = 3,
  GDMA_CW_TRANS = 4,
  GDMA_NONZERO = 5,
  GDMA_SYS = 6,
  GDMA_GATHER = 7,
  GDMA_SCATTER = 8,
  GDMA_REVERSE = 9,
  GDMA_COMPRESS = 10,
  GDMA_DECOMPRESS = 11,
  GDMA_LOSSY_COMPRESS = 12,
  GDMA_LOSSY_DECOMPRESS = 13,
  GDMA_RANDOM_MASK = 15,
  GDMA_TRANSFER = 16,
  GDMA_TYPE_NUM,
} GDMA_TYPE;

typedef enum {
  GDMA_FUNC_NONE       = 0,
  GDMA_FUNC_TRANS      = 1, // NC Transpose or Matrix Transpose
  GDMA_FUNC_BROADCAST  = 3,
} GDMA_FUNC_TYPE;

typedef enum {
  GDMA_SCATTER_INPLACE  = 0,
  GDMA_SCATTER_ADD      = 1,
} GDMA_SCATTER_TYPE;

typedef enum {
  GDMA_SYS_END = 0,
  GDMA_SYS_NOP = 1,
  GDMA_SYS_TRWR = 2,
  GDMA_SYS_SEND_MSG = 3,
  GDMA_SYS_WAIT_MSG = 4,
  GDMA_SYS_FORK = 5,
  GDMA_SYS_JOIN = 6,
  GDMA_SYS_EXIT = 7
} GDMA_SYS_TYPE;

typedef enum {
  // S: systerm memory: dram and l2sram
  GDMA_S2L = 0,
  GDMA_L2S = 1,
  GDMA_S2S = 2,
  GDMA_L2L = 3,
  GDMA_DIR_NUM,
} GDMA_DIRECTION;

typedef enum {
  GDMA_INT8 = 0,
  GDMA_FP16 = 1,
  GDMA_FP32 = 2,
  GDMA_INT16 = 3,
  GDMA_INT32 = 4,
  GDMA_BF16 = 5,
  GDMA_FP20 = 6,
  GDMA_FP8_E4M3 = 7,
  GDMA_FP8_E5M2 = 8,
  GDMA_FORMAT_NUM,
} GDMA_FORMAT;

#define SRC_IS_LOCAL(direction) \
    ((direction) == GDMA_L2L || (direction) == GDMA_L2S)
#define DST_IS_LOCAL(direction) \
    ((direction) == GDMA_S2L || (direction) == GDMA_L2L)

#define FORMAT_IS_FLOAT(format) \
    ((format) == GDMA_FP32 || (format) == GDMA_FP16 || (format) == GDMA_BF16)

typedef enum {
  GDMA_ARE_PSUM_WO = 0,
  GDMA_ARE_PSUM_RW = 1
} GDMA_ARE_PSUM_OP_TYPE;

typedef enum {
  GDMA_ARE_NOP = 0,
  GDMA_ARE_MUL = 1,
  GDMA_ARE_MAX = 2,
  GDMA_ARE_MIN = 3,
  GDMA_ARE_ADD = 4,
} GDMA_ARE_OPCODE_TYPE;

typedef enum {
  GDMA_ARE_FP32 = 0,
  GDMA_ARE_FP20 = 1,
  GDMA_ARE_FP16 = 2,
  GDMA_ARE_BF16 = 3,
  GDMA_ARE_INT32 = 4,
  GDMA_ARE_TYPE_NUM,
} GDMA_ARE_TYPE;

static inline int get_gdma_are_dtype(int dma_type) {
  if (dma_type == GDMA_FP32) {
    return GDMA_ARE_FP32;
  } else if (dma_type == GDMA_FP16) {
    return GDMA_ARE_FP16;
  } else if (dma_type == GDMA_BF16) {
    return GDMA_ARE_BF16;
  } else if (dma_type == GDMA_INT32) {
    return GDMA_ARE_INT32;
  } else if (dma_type == GDMA_FP20) {
    return GDMA_ARE_FP20;
  }
  ASSERT(0);
  return -1;
}

static inline int get_gdma_format_type_len(int t) {
  switch (t) {
    case GDMA_INT8:
    case GDMA_FP8_E4M3:
    case GDMA_FP8_E5M2:
      return 1;
    case GDMA_FP16:
    case GDMA_BF16:
    case GDMA_INT16:
      return 2;
    case GDMA_FP32:
    case GDMA_INT32:
      return 4;
  }
  return 0;
}

static inline PREC get_gdma_format_precision(int gdma_format) {
  switch (gdma_format) {
    case GDMA_FP32:
      return FP32;
    case GDMA_FP16:
      return FP16;
    case GDMA_BF16:
      return BFP16;
    case GDMA_INT32:
      return INT32;
    case GDMA_INT16:
      return INT16;
    case GDMA_INT8:
      return INT8;
    case GDMA_FP20:
      return FP20;
    case GDMA_FP8_E4M3:
    case GDMA_FP8_E5M2:
      return FP8;
    default:
      ASSERT(0);
      return INT8;
  }
}

static int32_t max_racu_size(int32_t racu_bytes_size, int data_format,
                            bool zero_guard) {
  zero_guard = (data_format == GDMA_FP16)
                   ? zero_guard
                   : (data_format == GDMA_BF16 ? true : false);
  int32_t blk_len = get_gdma_format_type_len(data_format) == 2 ? 32 : 16;
  int32_t blk_num = (racu_bytes_size + blk_len - 1) / blk_len;
  // align 128byte
  int32_t kmap_sz = (((zero_guard ? 2 : 1) * blk_num + NNVLC_ALIGN_BYTES - 1) /
                     NNVLC_ALIGN_BYTES)
                    << NNVLC_ALIGN_SHIFT;
  int32_t payload_sz =
      ((blk_len * blk_num + NNVLC_ALIGN_BYTES - 1) >> NNVLC_ALIGN_SHIFT)
      << NNVLC_ALIGN_SHIFT;
  return (kmap_sz + payload_sz);
}
#endif  // __GDMA_REG_VALUE_H__
