#ifndef ATOMIC_DMA_UTILS_H
#define ATOMIC_DMA_UTILS_H
#include "common.h"
#include "memmap.h"
#include "firmware_common_macro.h"
#include "gdma_reg_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IS_SIZE_SAME(src, dst) \
    ((src##_nsize == dst##_nsize) && \
     (src##_csize == dst##_csize) && \
     (src##_hsize == dst##_hsize) && \
     (src##_wsize == dst##_wsize))

static inline int is_lmem(u64 addr) {
  return (addr >= LOCAL_MEM_START_ADDR &&
          addr < (LOCAL_MEM_SIZE * NPU_NUM + LOCAL_MEM_START_ADDR));
}
static inline int is_smem(u64 addr) {
  return addr >= STATIC_MEM_START_ADDR &&
         addr < (STATIC_MEM_START_ADDR + STATIC_MEM_SIZE);
}
static inline int is_gmem(u64 addr) {
  addr &= (MAX_GMEM_SIZE - 1);
  return addr >= GLOBAL_MEM_START_ADDR &&
         addr < (GLOBAL_MEM_START_ADDR + CONFIG_GLOBAL_MEM_SIZE);
}
static inline int is_l2mem(u64 addr) {
  addr &= (MAX_GMEM_SIZE - 1);
  return (addr >= L2_SRAM_START_ADDR &&
          addr < (L2_SRAM_START_ADDR  + L2_SRAM_SIZE));
}

#define  ADD_TAG(addr, mask, tag) \
  (((addr) & (mask)) | (((u64)tag) << MAX_GMEM_BIT))
#define  CALC_STATIC_ADDR(addr) \
  (ADD_TAG(addr, STATIC_MEM_SIZE - 1, SMEM_TAG) | (1 << 26))
#define  CALC_LOCAL_ADDR(mem_idx, mem_offset) \
  ADD_TAG((mem_idx * LOCAL_MEM_SIZE + mem_offset), LOCAL_MEM_SIZE * NPU_NUM - 1, LMEM_TAG)

#ifdef __cplusplus
}
#endif

#endif