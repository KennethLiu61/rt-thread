#ifndef FIRMWARE_DEBUG_MACRO_H
#define FIRMWARE_DEBUG_MACRO_H
#include "firmware_common.h"

#include <setjmp.h>

#define DEBUG_PRINT_NUM 5

#ifdef USING_DEBUG_PROBE
static inline void print_float(float* data, const char* name){
    FW_DBG("%s: ",name);
    for(int i=0; i<DEBUG_PRINT_NUM; i++){
        FW_LOG("%10.7f ", data[i]);
    }
    FW_LOG("\n");
}

static inline void print_int(int* data, const char*name){
    FW_DBG("%s: ",name);
    for(int i=0; i<DEBUG_PRINT_NUM; i++){
        FW_LOG("0x%08x ", data[i]);
    }
    FW_LOG("\n");
}

static inline void print_char(char* data, const char*name){
    FW_DBG("%s: ",name);
    for(int i=0; i<DEBUG_PRINT_NUM; i++){
        FW_LOG("0x%02x ", (0xFF)&data[i]);
    }
    FW_LOG("\n");
}


#define DEBUG_PRINT_CONST_F(val) \
    FW_DBG("%s:" #val "=%f(0x%08x)\n", __func__, *(float*)&val, *(int*)&val);

#define DEBUG_PRINT_CONST_I32(val) \
    FW_DBG("%s:" #val "=0x%08x\n", __func__, *(int*)&val)

#define DEBUG_PRINT_GLOBAL(addr, type) \
    do {\
        type *ptr = (type*)GET_GLOBAL_ADDR(addr);\
        char prefix[256]; \
        sprintf(prefix, "%s(%d)-%s=0x%llx", __func__, __LINE__,#addr, addr); \
        print_##type(ptr, prefix);\
    } while(0)

#define __DEBUG_PRINT_LOCAL(idx, addr, type) \
    do {\
        if(idx<NPU_NUM){\
            u32 offset = ((addr)|LOCAL_MEM_START_ADDR)-LOCAL_MEM_START_ADDR;\
            type *ptr = (type*)GET_LOCAL_ADDR(idx, offset); \
            char prefix[256]; \
            sprintf(prefix, "%s(%d)-%s=0x%llx", __func__, __LINE__,#addr"_index_" #idx, (u64)offset); \
            print_##type(ptr, prefix);\
        }\
    } while(0)

#ifdef PRE_DEBUG_LOCAL
#define DEBUG_PRINT_LOCAL(idx, addr, type) \
    PRE_DEBUG_LOCAL;\
    __DEBUG_PRINT_LOCAL(idx, addr, type)
#else
#define DEBUG_PRINT_LOCAL(idx, addr, type) \
    __DEBUG_PRINT_LOCAL(idx, addr, type)
#endif

#define DEBUG_PRINT_LOCAL_ALL(addr, type) \
    DEBUG_PRINT_LOCAL(0, addr, type); \
    __DEBUG_PRINT_LOCAL(1, addr, type); \
    __DEBUG_PRINT_LOCAL(2, addr, type); \
    __DEBUG_PRINT_LOCAL(3, addr, type); \
    __DEBUG_PRINT_LOCAL(4, addr, type); \
    __DEBUG_PRINT_LOCAL(5, addr, type); \
    __DEBUG_PRINT_LOCAL(6, addr, type); \
    __DEBUG_PRINT_LOCAL(7, addr, type); \
    __DEBUG_PRINT_LOCAL(8, addr, type); \
    __DEBUG_PRINT_LOCAL(9, addr, type); \
    __DEBUG_PRINT_LOCAL(10, addr, type); \
    __DEBUG_PRINT_LOCAL(11, addr, type); \
    __DEBUG_PRINT_LOCAL(12, addr, type); \
    __DEBUG_PRINT_LOCAL(13, addr, type); \
    __DEBUG_PRINT_LOCAL(14, addr, type); \
    __DEBUG_PRINT_LOCAL(15, addr, type); \
    __DEBUG_PRINT_LOCAL(16, addr, type); \
    __DEBUG_PRINT_LOCAL(17, addr, type); \
    __DEBUG_PRINT_LOCAL(18, addr, type); \
    __DEBUG_PRINT_LOCAL(19, addr, type); \
    __DEBUG_PRINT_LOCAL(20, addr, type); \
    __DEBUG_PRINT_LOCAL(21, addr, type); \
    __DEBUG_PRINT_LOCAL(22, addr, type); \
    __DEBUG_PRINT_LOCAL(23, addr, type); \
    __DEBUG_PRINT_LOCAL(24, addr, type); \
    __DEBUG_PRINT_LOCAL(25, addr, type); \
    __DEBUG_PRINT_LOCAL(26, addr, type); \
    __DEBUG_PRINT_LOCAL(27, addr, type); \
    __DEBUG_PRINT_LOCAL(28, addr, type); \
    __DEBUG_PRINT_LOCAL(29, addr, type); \
    __DEBUG_PRINT_LOCAL(30, addr, type); \
    __DEBUG_PRINT_LOCAL(31, addr, type); \
    __DEBUG_PRINT_LOCAL(32, addr, type); \
    __DEBUG_PRINT_LOCAL(33, addr, type); \
    __DEBUG_PRINT_LOCAL(34, addr, type); \
    __DEBUG_PRINT_LOCAL(35, addr, type); \
    __DEBUG_PRINT_LOCAL(36, addr, type); \
    __DEBUG_PRINT_LOCAL(37, addr, type); \
    __DEBUG_PRINT_LOCAL(38, addr, type); \
    __DEBUG_PRINT_LOCAL(39, addr, type); \
    __DEBUG_PRINT_LOCAL(40, addr, type); \
    __DEBUG_PRINT_LOCAL(41, addr, type); \
    __DEBUG_PRINT_LOCAL(42, addr, type); \
    __DEBUG_PRINT_LOCAL(43, addr, type); \
    __DEBUG_PRINT_LOCAL(44, addr, type); \
    __DEBUG_PRINT_LOCAL(45, addr, type); \
    __DEBUG_PRINT_LOCAL(46, addr, type); \
    __DEBUG_PRINT_LOCAL(47, addr, type); \
    __DEBUG_PRINT_LOCAL(48, addr, type); \
    __DEBUG_PRINT_LOCAL(49, addr, type); \
    __DEBUG_PRINT_LOCAL(50, addr, type); \
    __DEBUG_PRINT_LOCAL(51, addr, type); \
    __DEBUG_PRINT_LOCAL(52, addr, type); \
    __DEBUG_PRINT_LOCAL(53, addr, type); \
    __DEBUG_PRINT_LOCAL(54, addr, type); \
    __DEBUG_PRINT_LOCAL(55, addr, type); \
    __DEBUG_PRINT_LOCAL(56, addr, type); \
    __DEBUG_PRINT_LOCAL(57, addr, type); \
    __DEBUG_PRINT_LOCAL(58, addr, type); \
    __DEBUG_PRINT_LOCAL(59, addr, type); \
    __DEBUG_PRINT_LOCAL(60, addr, type); \
    __DEBUG_PRINT_LOCAL(61, addr, type); \
    __DEBUG_PRINT_LOCAL(62, addr, type); \
    __DEBUG_PRINT_LOCAL(63, addr, type)



#define DEBUG_PROBE_GLOBAL(addr, type) \
    type *zg_##addr = (type*)GET_GLOBAL_ADDR(addr);\
    (void)zg_##addr

#define DEBUG_PROBE_LOCAL(idx, addr, type) \
    type *zl_##addr##_##idx = (type*)GET_LOCAL_ADDR(idx, addr); \
    (void)zl_##addr##_##idx

#define DEBUG_PROBE_LOCAL_LOW(idx, addr, type) \
    type *zl_##addr##_0##idx = (type*)GET_LOCAL_ADDR(idx, addr); \
    (void)zl_##addr##_0##idx

#define DEBUG_PROBE_LOCAL_ALL(addr, type) \
    DEBUG_PROBE_LOCAL_LOW(0, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(1, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(2, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(3, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(4, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(5, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(6, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(7, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(8, addr, type); \
    DEBUG_PROBE_LOCAL_LOW(9, addr, type); \
    DEBUG_PROBE_LOCAL(10, addr, type); \
    DEBUG_PROBE_LOCAL(11, addr, type); \
    DEBUG_PROBE_LOCAL(12, addr, type); \
    DEBUG_PROBE_LOCAL(13, addr, type); \
    DEBUG_PROBE_LOCAL(14, addr, type); \
    DEBUG_PROBE_LOCAL(15, addr, type); \
    DEBUG_PROBE_LOCAL(16, addr, type); \
    DEBUG_PROBE_LOCAL(17, addr, type); \
    DEBUG_PROBE_LOCAL(18, addr, type); \
    DEBUG_PROBE_LOCAL(19, addr, type); \
    DEBUG_PROBE_LOCAL(20, addr, type); \
    DEBUG_PROBE_LOCAL(21, addr, type); \
    DEBUG_PROBE_LOCAL(22, addr, type); \
    DEBUG_PROBE_LOCAL(23, addr, type); \
    DEBUG_PROBE_LOCAL(24, addr, type); \
    DEBUG_PROBE_LOCAL(25, addr, type); \
    DEBUG_PROBE_LOCAL(26, addr, type); \
    DEBUG_PROBE_LOCAL(27, addr, type); \
    DEBUG_PROBE_LOCAL(28, addr, type); \
    DEBUG_PROBE_LOCAL(29, addr, type); \
    DEBUG_PROBE_LOCAL(30, addr, type); \
    DEBUG_PROBE_LOCAL(31, addr, type); \
    DEBUG_PROBE_LOCAL(32, addr, type); \
    DEBUG_PROBE_LOCAL(33, addr, type); \
    DEBUG_PROBE_LOCAL(34, addr, type); \
    DEBUG_PROBE_LOCAL(35, addr, type); \
    DEBUG_PROBE_LOCAL(36, addr, type); \
    DEBUG_PROBE_LOCAL(37, addr, type); \
    DEBUG_PROBE_LOCAL(38, addr, type); \
    DEBUG_PROBE_LOCAL(39, addr, type); \
    DEBUG_PROBE_LOCAL(40, addr, type); \
    DEBUG_PROBE_LOCAL(41, addr, type); \
    DEBUG_PROBE_LOCAL(42, addr, type); \
    DEBUG_PROBE_LOCAL(43, addr, type); \
    DEBUG_PROBE_LOCAL(44, addr, type); \
    DEBUG_PROBE_LOCAL(45, addr, type); \
    DEBUG_PROBE_LOCAL(46, addr, type); \
    DEBUG_PROBE_LOCAL(47, addr, type); \
    DEBUG_PROBE_LOCAL(48, addr, type); \
    DEBUG_PROBE_LOCAL(49, addr, type); \
    DEBUG_PROBE_LOCAL(50, addr, type); \
    DEBUG_PROBE_LOCAL(51, addr, type); \
    DEBUG_PROBE_LOCAL(52, addr, type); \
    DEBUG_PROBE_LOCAL(53, addr, type); \
    DEBUG_PROBE_LOCAL(54, addr, type); \
    DEBUG_PROBE_LOCAL(55, addr, type); \
    DEBUG_PROBE_LOCAL(56, addr, type); \
    DEBUG_PROBE_LOCAL(57, addr, type); \
    DEBUG_PROBE_LOCAL(58, addr, type); \
    DEBUG_PROBE_LOCAL(59, addr, type); \
    DEBUG_PROBE_LOCAL(60, addr, type); \
    DEBUG_PROBE_LOCAL(61, addr, type); \
    DEBUG_PROBE_LOCAL(62, addr, type); \
    DEBUG_PROBE_LOCAL(63, addr, type);

#else

#define DEBUG_PRINT_GLOBAL(addr, type)
#define DEBUG_PRINT_LOCAL(idx, addr, type)
#define DEBUG_PRINT_LOCAL_ALL(addr, type)
#define DEBUG_PROBE_GLOBAL(addr, type)
#define DEBUG_PROBE_LOCAL(idx, addr, type)
#define DEBUG_PROBE_LOCAL_LOW(idx, addr, type)
#define DEBUG_PROBE_LOCAL_ALL(addr, type)
#define DEBUG_PRINT_CONST_F(val)
#define DEBUG_PRINT_CONST_I32(val)

#endif

#define DEBUG_PRINT_GLOBAL_I8(addr)    DEBUG_PRINT_GLOBAL(addr, char)
#define DEBUG_PRINT_GLOBAL_F(addr)     DEBUG_PRINT_GLOBAL(addr, float)
#define DEBUG_PRINT_GLOBAL_I32(addr)     DEBUG_PRINT_GLOBAL(addr, int)
#define DEBUG_PRINT_LOCAL_F(idx, addr) DEBUG_PRINT_LOCAL(idx, addr, float)
#define DEBUG_PRINT_LOCAL_I32(idx, addr) DEBUG_PRINT_LOCAL(idx, addr, int)
#define DEBUG_PRINT_LOCAL_I8(idx, addr) DEBUG_PRINT_LOCAL(idx, addr, char)
#define DEBUG_PRINT_LOCAL_ALL_F(addr)  DEBUG_PRINT_LOCAL_ALL(addr, float)
#define DEBUG_PROBE_GLOBAL_F(addr)     DEBUG_PROBE_GLOBAL(addr, float)
#define DEBUG_PROBE_LOCAL_F(idx, addr) DEBUG_PROBE_LOCAL(idx, addr, float)
#define DEBUG_PROBE_LOCAL_ALL_F(addr)  DEBUG_PROBE_LOCAL_ALL(addr, float)

#if defined(USING_CMODEL)
extern int get_atomic_cmodel_assert_enable();
#define ASSERT_FS_INFO(_cond, fmt, ...)                          \
  do {                                                           \
    if (get_atomic_cmodel_assert_enable()){                      \
	    if (!(_cond)) {                                      \
	  	printf("ASSERT_FS %s: %s: %d: %s\n",             \
         	 __FILE__, __func__, __LINE__, #_cond);          \
      		printf("ASSERT info: " fmt "\n", ##__VA_ARGS__); \
     		 _print_trace();                                  \
      		hang(-1);                                        \
   	 }                                                       \
    }                                                            \
  } while(0)
#elif defined(USING_FW_SIMULATION)
#define ASSERT_FS_INFO(_cond, fmt, ...)           \
  do {                                            \
    if (!(_cond)) {                              \
      printf("ASSERT_FS %s: %s: %d: %s\n",          \
          __FILE__, __func__, __LINE__, #_cond); \
      printf("ASSERT info: " fmt "\n", ##__VA_ARGS__); \
      _print_trace();                             \
      hang(-1);                                  \
    }                                            \
  } while(0)
#else
extern jmp_buf error_stat;
#define ASSERT_FS_INFO(_cond, fmt, ...)           \
  do {                                            \
    if (!(_cond)) {                              \
      printf("ASSERT_FS %s: %s: %d: %s\n",          \
          __FILE__, __func__, __LINE__, #_cond); \
      printf("ASSERT info: " fmt "\n", ##__VA_ARGS__); \
      _print_trace();                             \
      hang(-1);                             \
      longjmp(error_stat,1);   \
    }                                            \
  } while(0)
#endif
#define ASSERT_FS(_cond) ASSERT_FS_INFO(_cond, "none.")
#endif // FIRMWARE_DEBUG_MACRO_H

