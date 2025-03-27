#ifndef FIRMWARE_COMMON_MACRO_H
#define FIRMWARE_COMMON_MACRO_H

#include <stdlib.h>
#include <unistd.h>

#include <stdarg.h>
#include "cmd_id_proc.h"
#include "common.h"
#include "stas_gen_macro.h"
#include "firmware_timer.h"

#ifdef USING_CMODEL
#include "cmodel_common.h"

#ifdef SG_STAS_GEN
#include "get_atomic_profile.h"
#include "sg_stas_gen_util.h"
#endif

#ifdef USING_MULTI_THREAD_ENGINE
#include "cmodel_multi_thread.h"
#endif

#ifdef SG_TV_GEN
#include "sg_tv_gen_util.h"
#endif

#endif
#ifdef __cplusplus
extern "C" {
#endif

#if (defined(USING_CMODEL))
  #define FW_PRINT(fmt, args...) printf(fmt, ##args)
#elif (defined(USING_FW_SIMULATION))
  #define FW_PRINT(fmt, args...)
#elif defined(USING_FAKE_DDR_MODE)
  #define FW_PRINT(fmt, args...)
#else
  extern void fw_log(char *fmt, ...);
  #define FW_PRINT(fmt, args...) printf(fmt, ##args)
#endif

/* PLEASE use FW_ERR/FW_LOG/FW_INFO/FW_DBG rather than FW_PRINT */
#define FW_ERR(fmt, args...)               \
    do {                                   \
        FW_PRINT("[FW ERR] " fmt, ##args); \
        exit(-1);                          \
    } while (0)

#if defined(USING_FW_DEBUG)
  #define FW_LOG(fmt, args...) FW_PRINT(fmt, ##args)
  #define FW_INFO(fmt, args...) FW_PRINT("[FW INFO] " fmt, ##args)
  #define FW_DBG(fmt, args...) FW_PRINT("[FW DBG] " fmt, ##args)
#elif defined USING_FW_PRINT
  #define FW_LOG(fmt, args...) FW_PRINT(fmt, ##args)
  #define FW_INFO(fmt, args...) FW_PRINT("[FW INFO] " fmt, ##args)
  #define FW_DBG(fmt, args...)
#else
  #define FW_LOG(fmt, args...)
  #define FW_INFO(fmt, args...)
  #define FW_DBG(fmt, args...)
#endif

#define CMD_ID_OVERFLOW_VALUE 0x3fffful

#define REG_WORD_WIDTH 32
#define REG_WORD_SIZE (REG_WORD_WIDTH / 8)

#ifdef USING_CMODEL
  #define READ_REG(ADDR) cmodel_read_reg(ADDR)
  #ifdef SG_TV_GEN
    #define WRITE_REG(ADDR, DATA, reg_type) \
        cmodel_write_reg(ADDR, DATA);       \
        sg_wr_tv_dump_reg_pointer((volatile u32 *) (ADDR), DATA, reg_type);
  #else /* SG_TV_GEN */
    #define WRITE_REG(ADDR, DATA, reg_type) \
        cmodel_write_reg(ADDR, DATA)
  #endif
#else
  #define WRITE_REG(ADDR, DATA, reg_type) {*(volatile uint32_t *)map_to_kaddr((uint64_t)ADDR) = DATA;}
  #define READ_REG(ADDR) (*(volatile uint32_t *)map_to_kaddr((uint64_t)(ADDR)))
#endif

#ifdef USING_CMODEL

#define TP_DEBUG(...)                       \
    do {                                    \
        if (tpu_workitem_index() == 0)      \
        {                                   \
            printf(__VA_ARGS__);            \
            fflush(stdout);                 \
        }                                   \
    } while(0)

#else

//
// If you wanna use this, please edit /lib/firmware/tpuv7/sc11_config.ini
// like below and re-insmod sgcard
// [daemon-config]
// ...
// tp-debug-mode=memory
// log-memory-type=non-cacheable
// [eof]
//
// And use the following script to get your logs
//
// for i in $(seq 0 7); do
//     let addr=0x5e8000000+$i*0x8000000;
//     test_dump_log -d 0 -f tp${i}.txt -a $(printf "%x" $addr) -s 0x8000000;
//     sed -i '/[^[:print:]]/d' tp${i}.txt;
// done
//

__attribute__((weak)) int tp_debug(const char *, ...);

#define TP_DEBUG(...)                       \
    do {                                    \
        tp_debug(__VA_ARGS__);              \
    } while(0)

#endif

#ifdef USING_CMODEL
  #ifdef SG_TV_GEN
    #define FW_READ32_WAIT_GE(ADDR, WAIT_DATA, BITS, SHIFT) \
        sg_read32_wait_ge_tv(ADDR, WAIT_DATA, BITS, SHIFT, NODECHIP_REG);
    #define FW_READ32_WAIT_EQ(ADDR, WAIT_DATA, BITS, SHIFT) \
        sg_read32_wait_eq_tv(ADDR, WAIT_DATA, BITS, SHIFT, NODECHIP_REG);
    #define FW_CDMA_READ32_WAIT_GE(SEND_ADDR, RECV_ADDR, WAIT_DATA, BITS, SHIFT, PORT)
    #define FW_WAIT_INST_INFO_EMPTY(reg_id, engine_main_ctrl, empty_val)
    #define FW_CDMA_PERF_READ32_WAIT_GE(SEND_ADDR0, RECV_ADDR0, WAIT_DATA0, SEND_ADDR1, RECV_ADDR1, WAIT_DATA1, BITS, SHIFT, INFO_ADDR, CHIPS, PORTS, ACTIONS)
  #else
    #define FW_READ32_WAIT_GE(ADDR, WAIT_DATA, BITS, SHIFT)
    #define FW_READ32_WAIT_EQ(ADDR, WAIT_DATA, BITS, SHIFT) UNUSED(BITS); UNUSED(SHIFT)
    #define FW_CDMA_READ32_WAIT_GE(SEND_ADDR, RECV_ADDR, WAIT_DATA, BITS, SHIFT, PORT)
    #define FW_WAIT_INST_INFO_EMPTY(reg_id, engine_main_ctrl, empty_val)
    #define FW_CDMA_PERF_READ32_WAIT_GE(SEND_ADDR0, RECV_ADDR0, WAIT_DATA0, SEND_ADDR1, RECV_ADDR1, WAIT_DATA1, BITS, SHIFT, INFO_ADDR, CHIPS, PORTS, ACTIONS)
  #endif
#else /* USING_CMODEL */
  #define FW_READ32_WAIT_GE(ADDR, WAIT_DATA, BITS, SHIFT)                                   \
      volatile u32 __rd_data;                                                               \
      do {                                                                                  \
          /*int count = 0;*/                                                                \
          volatile u32 *rd_addr = (volatile u32 *) map_to_kaddr(ADDR);                      \
          u32 mask_val = (0xffffffff >> (32 - (BITS)));                                     \
          while (1) {                                                                       \
              __rd_data = ((*rd_addr) >> (SHIFT)) & mask_val;                               \
              if (__rd_data >= (WAIT_DATA))                                                 \
                  break;                                                                    \
              /*if((count++%50000) == 99) {\
                printf("tpu   wait data = %d, rd data=%d\n", WAIT_DATA ,rd_data);          \
                u32 rd1 = READ_REG(GDMA_ENGINE_MAIN_CTRL + 36);\
                u32 rd2 = READ_REG(SDMA_ENGINE_MAIN_CTRL + 36);\
                printf("gdma cmd id: %d, sdma cmd id: %d\n", rd1, rd2);\
              }*/                                                                           \
          }                                                                                 \
      } while (0)
  #define FW_CDMA_READ32_WAIT_GE(SEND_ADDR, RECV_ADDR, WAIT_DATA, BITS, SHIFT, PORT)              \
      do {                                                                                  \
          /*int count = 0;*/                                                                \
          volatile u32 *send_addr = (volatile u32 *) map_to_kaddr(SEND_ADDR);               \
          volatile u32 *recv_addr = (volatile u32 *) map_to_kaddr(RECV_ADDR);               \
          u32 mask_val = (0xffffffff >> (32 - (BITS)));\
          u32 print_num = 10;\
          u32 need_print = 0; \
          while (1) {                                                                       \
              volatile u32 send_data = ((*send_addr) >> (SHIFT)) & mask_val;                \
              volatile u32 recv_data = ((*recv_addr) >> (SHIFT)) & mask_val;                \
              if(print_num > 1 && need_print) \
              { \
                print_num--; \
                TP_DEBUG("send_data = %d, recv_data = %d, wait_data = %d, port = %d\n", send_data, recv_data, WAIT_DATA, PORT); \
              } \
              if ((send_data >= (WAIT_DATA)) || (recv_data >= (WAIT_DATA)))                 \
                  break;                                                                    \
              /*if((count++%50000) == 99) {\
                printf("CDMA wait data = %d, send data=%d, recv data=%d\n", WAIT_DATA ,send_data, recv_data);          \
                u32 rd1 = READ_REG(CDMA_ENGINE_MAIN_CTRL(0) + 68);\
                u32 rd2 = READ_REG(CDMA_ENGINE_MAIN_CTRL(0) + 72);\
                printf("cdma send cmd id: %d, recv cmd id: %d\n", rd1, rd2);\
              }*/ \
          }                                                                                \
      } while (0)

#define FW_CDMA_PERF_READ32_WAIT_GE(SEND_ADDR0, RECV_ADDR0, WAIT_DATA0, SEND_ADDR1, RECV_ADDR1, WAIT_DATA1, BITS, SHIFT, INFO_ADDR, CHIPS, PORTS, ACTIONS) \
    do {                                                                                            \
        volatile u32 *send_addr0 = (volatile u32 *) map_to_kaddr(SEND_ADDR0);                       \
        volatile u32 *recv_addr0 = (volatile u32 *) map_to_kaddr(RECV_ADDR0);                       \
        volatile u32 *send_addr1 = (volatile u32 *) map_to_kaddr(SEND_ADDR1);                       \
        volatile u32 *recv_addr1 = (volatile u32 *) map_to_kaddr(RECV_ADDR1);                       \
        u32 mask_val = (0xffffffff >> (32 - (BITS)));                                               \
        u32 print_num = 10;                                                                         \
        u32 need_print = 0;                                                                         \
        int condition1_met = 0;                                                                     \
        int condition0_met = 0;                                                                     \
        while (1) {                                                                                 \
            volatile u32 send_data0 = ((*send_addr0) >> (SHIFT)) & mask_val;                        \
            volatile u32 recv_data0 = ((*recv_addr0) >> (SHIFT)) & mask_val;                        \
            volatile u32 send_data1 = ((*send_addr1) >> (SHIFT)) & mask_val;                        \
            volatile u32 recv_data1 = ((*recv_addr1) >> (SHIFT)) & mask_val;                        \
            if(print_num > 1 && need_print) \
              { \
                print_num--; \
                TP_DEBUG("send_data0 = %d, recv_data0 = %d, wait_data0 = %d, port0 = %d\n", send_data0, recv_data0, WAIT_DATA0, PORTS[0]); \
                TP_DEBUG("send_data1 = %d, recv_data1 = %d, wait_data1 = %d, port1 = %d\n", send_data1, recv_data1, WAIT_DATA1, PORTS[1]); \
              } \
            if (!condition0_met && (send_data0 >= WAIT_DATA0 || recv_data0 >= WAIT_DATA0)) {          \
                tp_debug("[finish]chip %d %d, action %d\n", CHIPS[0], CHIPS[1], ACTIONS[0]); \
                INFO_ADDR[2] = firmware_timer_get_cycle(); \
                INFO_ADDR[1] = CHIPS[1]; \
                INFO_ADDR[3] = ACTIONS[0]; \
                condition0_met = 1;                                                                 \
            }                                                                                       \
            if (!condition1_met && (send_data1 >= WAIT_DATA1 || recv_data1 >= WAIT_DATA1)) {          \
                tp_debug("[finish]chip %d %d, action %d\n", CHIPS[2], CHIPS[3], ACTIONS[1]); \
                condition1_met = 1;                                                                 \
                INFO_ADDR[5] = firmware_timer_get_cycle(); \
                INFO_ADDR[4] = CHIPS[3]; \
                INFO_ADDR[6] = ACTIONS[1]; \
            }                                                                                       \
            if (condition0_met && condition1_met) {                                                 \
                break;                                                                              \
            }                                                                                       \
        }                                                                                           \
    } while (0)

  #define FW_READ32_WAIT_EQ(ADDR, WAIT_DATA, BITS, SHIFT)                \
      do {                                                               \
          volatile u32 *rd_addr = (volatile u32 *) map_to_kaddr(ADDR);   \
          u32 mask_val = (0xffffffff >> (32 - (BITS)));                  \
          while (1) {                                                    \
              volatile u32 rd_data = ((*rd_addr) >> (SHIFT)) & mask_val; \
              if (rd_data == (WAIT_DATA))                                \
                  break;                                                 \
          }                                                              \
      } while (0)
  #define FW_WAIT_INST_INFO_EMPTY(reg_id, engine_main_ctrl, empty_val)              \
      do {                                                               \
          volatile u32 *rd_addr = (volatile u32 *)map_to_kaddr(          \
            engine_main_ctrl + ((reg_id.where >> 5) << 2));              \
          reg_id.where &= 31;                                           \
          u32 mask_val = (u32)(1 << reg_id.len) - 1;                   \
          while (1) {                                                    \
              volatile u32 rd_data = ((*rd_addr) >> (reg_id.where)) & mask_val; \
              if (rd_data == empty_val) break;                  \
          }                                                              \
      } while (0)
#endif /* USING_CMODEL */

#define FW_REG_ID_WAIT_GE(BASE, REG_ID, WAIT_DATA)                              \
    do {                                                                        \
        int id_info[] = REG_ID;                                                 \
        int id_offset = id_info[0] / REG_WORD_WIDTH * REG_WORD_SIZE;            \
        int id_shift = id_info[0] % REG_WORD_WIDTH;                             \
        int id_len = id_info[1];                                                \
        (void) id_offset;                                                       \
        (void) id_shift;                                                        \
        (void) id_len;                                                          \
        FW_READ32_WAIT_GE(((BASE) + id_offset), (WAIT_DATA), id_len, id_shift); \
    } while (0)

#define FW_REG_ID_WAIT_EQ(BASE, REG_ID, WAIT_DATA)                              \
    do {                                                                        \
        int id_info[] = REG_ID;                                                 \
        int id_offset = id_info[0] / REG_WORD_WIDTH * REG_WORD_SIZE;            \
        int id_shift = id_info[0] % REG_WORD_WIDTH;                             \
        int id_len = id_info[1];                                                \
        (void) id_offset;                                                       \
        (void) id_shift;                                                        \
        (void) id_len;                                                          \
        FW_READ32_WAIT_EQ(((BASE) + id_offset), (WAIT_DATA), id_len, id_shift); \
    } while (0)

#define FW_REG_ID_WRITE(BASE, REG_ID, VALUE)                                         \
    do {                                                                             \
        int id_info[] = REG_ID;                                                      \
        long id_offset = id_info[0] / REG_WORD_WIDTH * REG_WORD_SIZE;                \
        int id_shift = id_info[0] % REG_WORD_WIDTH;                                  \
        int id_len = id_info[1];                                                     \
        int mask = (id_len < REG_WORD_WIDTH) ? ((1 << id_len) - 1) : -1;             \
        (void) id_offset;                                                            \
        unsigned int reg_val = READ_REG((BASE) + id_offset);                         \
        reg_val = (reg_val & (~(mask << id_shift))) | (((VALUE) &mask) << id_shift); \
        WRITE_REG(((BASE) + id_offset), (reg_val), (NODECHIP_REG));                  \
    } while (0)

#ifdef USING_CMODEL
  #define READ_SHARE_REG(IDX) \
      cmodel_read_share_reg(IDX, get_cur_nodechip_idx())
  #define WRITE_SHARE_REG(IDX, VAL) \
      nodechip_write_share_reg(IDX, VAL, get_cur_nodechip_idx())
#else
  #define READ_SHARE_REG(IDX) READ_REG((SHARE_REG_BASE_ADDR + (IDX) *4))
  #define WRITE_SHARE_REG(IDX, VAL) (*(volatile u32 *) map_to_kaddr(SHARE_REG_BASE_ADDR + (IDX) *4) = (VAL))
#endif

#ifdef __cplusplus
}
#endif

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS 8
#endif  // FW_MAX_SHAPE_DIMS

#endif  /* FIRMWARE_COMMON_H */
