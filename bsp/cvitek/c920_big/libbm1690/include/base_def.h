#ifndef BASE_DEF_H_
#define BASE_DEF_H_

#include "memmap.h"
#include "macros.h"
#include "engine_type.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct REG_ID {
    unsigned short where;  /* low bit index. */
    unsigned short len;    /* bit length. */
} reg_id_t;

typedef struct REG_PACK {
    reg_id_t id;         /* register id. */
    u64 val;    /* value to be read or written. */
} reg_pack_t;

#define BD_ID(id)    (id[0])
#define GDMA_ID(id)  (id[1])

#define LAST_INI_REG_VAL 0x76125438

#ifdef USING_CMODEL
  #define GET_GLOBAL_ADDR(ADDR) \
    ((u8 *)get_global_memaddr(get_cur_nodechip_idx()) + (ADDR) - GLOBAL_MEM_START_ADDR)
  #define GET_LOCAL_ADDR(LOCALMEM_IDX, LOCALMEM_OFFSET) \
    ((u8 *)get_local_memaddr_by_node(get_cur_nodechip_idx(), LOCALMEM_IDX) + (LOCALMEM_OFFSET))
  #define GET_SMEM_ADDR(ADDR) \
    ((u8 *)get_static_memaddr_by_node(get_cur_nodechip_idx()) + (ADDR) - STATIC_MEM_START_ADDR)
  #define GET_L2_SRAM_ADDR(ADDR) \
    ((u8 *)get_l2_sram(get_cur_nodechip_idx()) + (ADDR) - L2_SRAM_START_ADDR)
#else

  #if defined USING_EDA || defined USING_PLD_TEST
      #define map_to_kaddr(addr) (uint64_t)(addr)
  #else
    __attribute__((weak)) void *map_to_kaddr(unsigned long long addr) {
      #include <firmware_common_macro.h>
        FW_ERR(
            "Weak symbol map_to_kaddr has been invoked!\n"
            "If you see this message, it means your calling executable does not properly provide map_to_kaddr symbol.\n"
            "The calling executable MUST:\n"
            "1. have a strong symbol map_to_kaddr;\n"
            "2. linked with option -Wl,--export-dynamic;\n"
            "3. not linked with -static option.\n");
        exit(-1);
        return NULL;
    }
  #endif

  #define GET_GLOBAL_ADDR(ADDR) \
    ((u8 *)map_to_kaddr(GLOBAL_MEM_START_ADDR_ARM) + (ADDR) - GLOBAL_MEM_START_ADDR)
  #define GET_LOCAL_ADDR(LOCALMEM_IDX, LOCALMEM_OFFSET) \
    ((u8 *)map_to_kaddr(LOCAL_MEM_START_ADDR) + LOCALMEM_IDX * LOCAL_MEM_SIZE + (LOCALMEM_OFFSET))
  #define GET_SMEM_ADDR(ADDR) \
    ((u8 *)(map_to_kaddr(ADDR)))
  #define GET_L2_SRAM_ADDR(ADDR) \
    ((u8 *)(map_to_kaddr(ADDR)))
#endif

enum {
  C2C_SEND = 0,
  C2C_RECV,
};

#ifdef USING_CMODEL
  #define get_c2c_port(myself_devid, peer_devid, direction) cmodel_get_cdma_by_topology_map(myself_devid, peer_devid, direction)
#else
  #if defined USING_EDA || defined USING_PLD_TEST
    #define get_c2c_port(myself_devid, peer_devid, direction) -1
  #else
    __attribute__((weak)) int get_c2c_port(int myself_devid, int peer_devid, int direction) {
        FW_ERR(
            "Weak symbol get_c2c_port has been invoked!\n"
            "If you see this message, it means your calling executable does not properly provide get_c2c_port symbol.\n"
            "The calling executable MUST:\n"
            "1. have a strong symbol get_c2c_port;\n"
            "2. linked with option -Wl,--export-dynamic;\n"
            "3. not linked with -static option.\n");
        exit(-1);
        return -1;
    }
  #endif
#endif

#ifdef USING_CMODEL
#define GET_SHARE_MEM_ADDR(offset) cmodel_get_share_memory_addr(offset, get_cur_nodechip_idx())
#define GLOBAL_MEM_SIZE(node_idx) (cmodel_get_global_mem_size(node_idx))
#else
#define GET_SHARE_MEM_ADDR(offset) (u32 *)map_to_kaddr(SHARE_MEM_START_ADDR + (offset)*4)
#define GLOBAL_MEM_SIZE(node_idx) (CONFIG_GLOBAL_MEM_SIZE)
#endif

#define IN_L2_SRAM(addr) (((addr & (MAX_GMEM_SIZE - 1)) >= L2_SRAM_START_ADDR) && ((addr & (MAX_GMEM_SIZE - 1)) < L2_SRAM_START_ADDR + L2_SRAM_SIZE))
#define IN_GLOBAL_MEM(addr) ((addr) >= GLOBAL_MEM_START_ADDR)

INLINE static int get_eu_num(PREC precision) {
    switch(precision) {
        case INT4: return EU_NUM_4BIT;
        case FP8:
        case INT8: return EU_NUM_8BIT;
        case INT16:
        case FP16:
        case BFP16: return EU_NUM_16BIT;
        case INT32:
        case TF32:
        case FP32: return EU_NUM_32BIT;
        default: ASSERT_INFO(0, "ERROR PREC!");
    }
    return 0;
}

INLINE static int get_local_cstride(int h, int w, bool align, PREC precision) {
  return align ? ALIGN(h * w, get_eu_num(precision)) : (h * w);
}

INLINE static int get_local_nstride(int c_stride, int c, u32 local_addr) {
  int npu_idx = (local_addr & (NPU_NUM * LOCAL_MEM_SIZE - 1)) / LOCAL_MEM_SIZE;
  return (ceiling_func(c + npu_idx, NPU_NUM) * c_stride);
}

#define GET_CUBE_IC_PARALLEL(prec) \
  ((prec == INT8 || prec == FP8) ? 64 : \
   (prec == FP16 || prec == BFP16) ? 32 : \
   (prec == TF32) ? 16 : 1)

inline static int get_conv_ic_parallel(PREC prec) {
  return GET_CUBE_IC_PARALLEL(prec);
}

inline static int get_npu_index(u32 local_addr) {
  return (local_addr >> LOCAL_MEM_ADDRWIDTH);
}
#ifdef __cplusplus
}
#endif

#endif
