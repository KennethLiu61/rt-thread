#ifndef MEMMAP_H
#define MEMMAP_H

#include "config.h"

#ifdef USING_CMODEL
#ifdef __cplusplus
extern "C" {
#endif
int get_cur_nodechip_idx();
void set_cur_nodechip_idx(int node_idx);
#ifdef __cplusplus
}
#endif
#undef CORE_ID
#define CORE_ID get_cur_nodechip_idx()
#endif

// =============================================
// The following is allocation for static memory
// For lookup table
// Align with 128byte
#define SFU_TAYLOR_TABLE_SIZE       32
#define SFU_TAYLOR_L_TABLE_SIZE     64
#define ERF_TAYLOR_SIZE             16
#define STATIC_MEM_OFFSET           0
#define SERIAL_NUMBER_SIZE          64
#define SIN_TAYLOR_SIZE             32
#define COS_TAYLOR_SIZE             32
#define ARCSIN_TAYLOR_SIZE          64
#define TAN_TAYLOR_SIZE             32
#define EXP_TAYLOR_OFFSET           (STATIC_MEM_OFFSET)
#define LOG_TAYLOR_OFFSET           (EXP_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float))
#define ERF_TAYLOR_OFFSET           (LOG_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(float))
#define SERIAL_NUMBER_OFFSET        (ERF_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float)) // align 128 byte
#define SIN_TAYLOR_OFFSET           (SERIAL_NUMBER_OFFSET + SERIAL_NUMBER_SIZE * sizeof(float))
#define COS_TAYLOR_OFFSET           (SIN_TAYLOR_OFFSET + SIN_TAYLOR_SIZE * sizeof(float))
#define ARCSIN_TAYLOR_OFFSET        (COS_TAYLOR_OFFSET + COS_TAYLOR_SIZE * sizeof(float))
#define TAN_TAYLOR_OFFSET           (ARCSIN_TAYLOR_OFFSET + ARCSIN_TAYLOR_SIZE * sizeof(float))
#define EXP_FP16_TAYLOR_OFFSET      (TAN_TAYLOR_OFFSET + TAN_TAYLOR_SIZE * sizeof(float))
#define EXP_BF16_TAYLOR_OFFSET      (EXP_FP16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short)) // align 128 byte
#define ERF_FP16_TAYLOR_OFFSET      (EXP_BF16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define ERF_BF16_TAYLOR_OFFSET      (ERF_FP16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define LOG_FP16_TAYLOR_OFFSET      (ERF_BF16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define LOG_BF16_TAYLOR_OFFSET      (LOG_FP16_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(short))
#define SIN_FP16_TAYLOR_OFFSET      (LOG_BF16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_BFP16_TAYLOR_OFFSET     (SIN_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_FP16_TAYLOR_OFFSET      (SIN_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_BFP16_TAYLOR_OFFSET     (COS_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define SMEM_STATIC_END_OFFSET      (COS_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))

// ============================================
// SMEM_STATIC_END_OFFSET must <= STATIC_MEM_SHARE_OFFSET

#if defined(USING_PLD_TEST) && !defined(USING_CMODEL) && !defined(USING_EDA)
#define GLOBAL_MEM_START_ADDR_ARM          0xa0000000UL
#define GLOBAL_MEM_START_ADDR              0xa0000000UL
#else
#define GLOBAL_MEM_START_ADDR_ARM          0x0UL
#define GLOBAL_MEM_START_ADDR              0x0UL
#endif
#define GLOBAL_MEM_CMD_START_OFFSET        0x0

#define L2_SRAM_START_ADDR                 0x6980000000UL

#ifdef USING_CMODEL
    // In case somewhere we use them in fw with hard-coded core index, e.g. GET_LMEM_START_ADDR(0)
    #define GET_LMEM_START_ADDR(core_id)       (0x6900000000UL + core_id * CORE_OFFSET)
    #define GET_SMEM_START_ADDR(core_id)       (0x6904000000UL + core_id * CORE_OFFSET)
#endif

#define LOCAL_MEM_START_ADDR               (0x6900000000UL + CORE_ID * CORE_OFFSET)
#define LOCAL_MEM_ADDRWIDTH                (CONFIG_LOCAL_MEM_ADDRWIDTH)

#define STATIC_MEM_START_OFFSET            (4000000UL)
#define STATIC_MEM_START_ADDR              (0x6904000000UL + CORE_ID * CORE_OFFSET)
#define STATIC_MEM_SHARE_SIZE              0x4000         // 16KB for share memory
#define STATIC_MEM_SHARE_OFFSET            (STATIC_MEM_SIZE - STATIC_MEM_SHARE_SIZE) // 48KB offset
#define SHARE_MEM_START_ADDR               (STATIC_MEM_START_ADDR + STATIC_MEM_SHARE_OFFSET)
#ifdef USING_FAKE_DDR_MODE
#define SHARE_REG_BASE_ADDR                (SHARE_MEM_START_ADDR + 15 * 1024)
#else
#define SHARE_REG_BASE_ADDR                (TOP_REG_CTRL_BASE_ADDR + 0x80)
#endif
#define SHARE_REG_CNT                      14
#define SHARE_REG_MESSAGE_WP               (0)
#define SHARE_REG_MESSAGE_RP               (1)
#define SHARE_REG_FW_STATUS                (9)
#define SHARE_REG_C920_FW_LOG_RP           (11)
#define SHARE_REG_C920_FW_LOG_WP           (12)
// msg share mem total 4K words, 2k words for each channel
#define SHAREMEM_SIZE_BIT                  11
#define SHAREMEM_MASK                      ((1 << SHAREMEM_SIZE_BIT) - 1)

#define BD_REG_COUNT                       (32)
#define BD_CFG_REG_COUNT                   (64)
#define GDMA_REG_COUNT                     (24)
#define GDMA_CFG_REG_COUNT                 (68)
#define SDMA_REG_COUNT                     (24)
#define SDMA_CFG_REG_COUNT                 (154)
#define HAU_REG_COUNT                      (20)
#define HAU_CFG_REG_COUNT                  (41)
#define CDMA_REG_COUNT                     (32)
#define CDMA_CFG_REG_COUNT                 (240)

#define UART_CTRL_BASE_ADDR                0x7030000000UL
#define SPI_BASE_ADDR                      0x7030004000UL
#define I2C0_REG_CTRL_BASE_ADDR            0x7040005000UL
#define I2C1_REG_CTRL_BASE_ADDR            0x7040006000UL
#define I2C2_REG_CTRL_BASE_ADDR            0x7040008000UL
#define NV_TIMER_CTRL_BASE_ADDR            0x50010180UL
#define OS_TIMER_CTRL_BASE_ADDR            0x50022000UL
#define INT0_CTRL_BASE_ADDR                0x50023000UL
#define EFUSE_BASE                         0x7040000000UL
#define GPIO_CTRL_BASE_ADDR                0x7040009000UL
#define PCIE_BUS_SCAN_ADDR                 0x60000000UL
#define PWM_CTRL_BASE_ADDR                 0x704000C000UL
#define DDR_CTRL_BASE_ADDR                 0x68000000UL
#define GPIO0_CTRL_BASE_ADDR               0x7040009000UL


#define TOP_REG_CTRL_BASE_ADDR             0x28100000UL
// There're 8 device lock register, the lower 4 for PCIe and upper 4 for SoC.
#define TOP_REG_DEVICE_LOCK0               (TOP_REG_CTRL_BASE_ADDR + 0x100)
#define TOP_REG_DEVICE_LOCK1               (TOP_REG_CTRL_BASE_ADDR + 0x104)
#define TOP_REG_DEVICE_LOCK2               (TOP_REG_CTRL_BASE_ADDR + 0x108)
#define TOP_REG_DEVICE_LOCK3               (TOP_REG_CTRL_BASE_ADDR + 0x10c)
#define TOP_GP_REG_C920_IRQ_STATUS         (TOP_REG_CTRL_BASE_ADDR + 0xf8)
#define TOP_GP_REG_C920_IRQ_SET            (TOP_REG_CTRL_BASE_ADDR + 0xf8)
#define TOP_GP_REG_C920_IRQ_CLR            (TOP_REG_CTRL_BASE_ADDR + 0x78)
#define TOP_GP_REG_A53_IRQ_STATUS          (TOP_REG_CTRL_BASE_ADDR + 0xfc)
#define TOP_GP_REG_A53_IRQ_SET             (TOP_REG_CTRL_BASE_ADDR + 0xfc)
#define TOP_GP_REG_A53_IRQ_CLR             (TOP_REG_CTRL_BASE_ADDR + 0x7c)
#define TOP_REG_DEVICE_LOCK_CDMA           TOP_REG_DEVICE_LOCK0
#define TOP_REG_DEVICE_LOCK_CLK            TOP_REG_DEVICE_LOCK1


#define MMU_ENGINE_BASE_ADDR               0x58002000UL
#define CDMA_ENGINE_BASE_ADDR              0x58003000UL

#define BD_ENGINE_BASE_ADDR                (0x6908000000UL + CORE_ID * CORE_OFFSET)
#define BD_ENGINE_BASE_ADDR_AHB            (BD_ENGINE_BASE_ADDR)
#define BD_ENGINE_MAIN_CTRL                (BD_ENGINE_BASE_ADDR + 0x100)
#define BD_THREAD_OFFSET                   (0x2000)
#define BD_ENGINE_MAIN_CTRL_AHB            (BD_ENGINE_BASE_ADDR_AHB + 0x100)
#define BDC_CMD_BASE_ADDR                  (BD_ENGINE_BASE_ADDR + 0x00)
#define BDC_CMD_BASE_ADDR_AHB              (BD_ENGINE_BASE_ADDR_AHB + 0x00) // for arm read

#define GDMA_ENGINE_BASE_ADDR              (0x6908010000UL + CORE_ID * CORE_OFFSET)
#define GDMA_ENGINE_BASE_ADDR_AHB          (GDMA_ENGINE_BASE_ADDR)
#define GDMA_ENGINE_MAIN_CTRL              (GDMA_ENGINE_BASE_ADDR + 0x1000)
#define GDMA_ENGINE_MAIN_CTRL_AHB          (GDMA_ENGINE_BASE_ADDR_AHB + 0x1000)
#define GDMA_CMD_BASE_ADDR                 (GDMA_ENGINE_BASE_ADDR)
#define GDMA_THREAD_OFFSET                 (0x2000)
#define GDMA_SLAVE_CMD_BASE_ADDR           (GDMA_ENGINE_BASE_ADDR + GDMA_THREAD_OFFSET)
#define GDMA_CMD_BASE_ADDR_AHB             (GDMA_ENGINE_BASE_ADDR_AHB) // for arm read

static inline uint64_t sdma_engine_addr(int core_id) {
  if (core_id > 3) {
    // sdma4-7 is in vc_sys
    return 0x6B00080000UL + (core_id - 4) * VC_SYS_CORE_OFFSET;
  } else {
    // sdma0-3 is in cc_sys
    return 0x6908020000UL + core_id * CORE_OFFSET;
  }
  return 0;
}
#define SDMA_ENGINE_BASE_ADDR              (sdma_engine_addr(CORE_ID))
#define SDMA_ENGINE_MAIN_CTRL              (SDMA_ENGINE_BASE_ADDR + 0x1000)
#define SDMA_CMD_BASE_ADDR                 (SDMA_ENGINE_BASE_ADDR)
#define VSDMA_ENGINE_BASE_ADDR(port)       (sdma_engine_addr(port) + 0x800)
#define VSDMA_ENGINE_MAIN_CTRL(port)       (sdma_engine_addr(port) + 0x1000)
#define VSDMA_CMD_BASE_ADDR(port)          (VSDMA_ENGINE_BASE_ADDR(port))

#define HAU_BASE_ADDR                      (0x6908030000UL + CORE_ID * CORE_OFFSET)
#define HAU_ENGINE_MAIN_CTRL               (HAU_BASE_ADDR + 0x080)
#define HAU_CMD_BASE_ADDR                  (HAU_BASE_ADDR + 0x140)

#define TPU_SYS_BASE_ADDR                  (0x6908050000UL + CORE_ID * CORE_OFFSET)
#define VC_SYS_BASE_ADDR                   (0x6B00000000UL + (CORE_ID - 4) * VC_SYS_CORE_OFFSET)
#define TPU_SYS_SOFT_RESET                 (TPU_SYS_BASE_ADDR + 0x4)
#define TPU_SYS_CMN_CTRL                   (TPU_SYS_BASE_ADDR + 0x10)
#define TPU_SYS_PMU_ENABLE                 (TPU_SYS_BASE_ADDR + 0x250) // tpsys reg, h250 reg_monitor_en: tpu gdma sdma monitor en
#define VC_SYS_PMU_ENABLE                  (VC_SYS_BASE_ADDR + 0x12c) // tpsys reg, h12c, bit[31] reg_monitor_en: tpu gdma sdma monitor en
#define TPU_SYS_L2M_CFG_CTRL               (TPU_SYS_BASE_ADDR + 0x260)
#define TPU_SYS_HNF_L2M_CTRL               (TPU_SYS_BASE_ADDR + 0x27c)
// all cc_sys use the same device_lock
#define TPU_SYS_DEVICE_LOCK                (0x6908050064UL)

#define TPU_SYS_MSG_REG_ADDR               (0x6908040000UL) // read only register

// 0~3 for c2c(0); 4~7 for c2c(1); 8~10 for cxp, 8 for p2p, 9/10 for host
#define CDMA_BASE_ADDR(port)                                                   \
  ((port < 8)                                                                   \
      ? (0x6C00790000UL + (port / 4) * 0x2000000 + (port % 4) * 0x10000)       \
      : (0x6C08790000UL + (port - 8) * 0x10000))
#define CDMA_ENGINE_MAIN_CTRL(port)        (CDMA_BASE_ADDR(port) + 0x1000)
#define CDMA_TCREDICT(port)                (CDMA_BASE_ADDR(port) + 0x800)
#define CDMA_DESCRIPTOR_UPDATE(port)       (CDMA_BASE_ADDR(port) + 0x400)
#define CDMA_CMD_BASE_ADDR(port)           (CDMA_BASE_ADDR(port) + 0x0)
#define CDMA_CMD_REG_DES_ADDR(port)        (CDMA_ENGINE_MAIN_CTRL(port) + 0x160)
#define CDMA_CSR_INTER_DIE_RW(port)        (CDMA_ENGINE_MAIN_CTRL(port) + 0xc)
#define CDMA_CSR_INTRA_DIE_RW(port)        (CDMA_ENGINE_MAIN_CTRL(port) + 0x23c)
#define CDMA_CSR_CMD_DONE_STATUS(port)     (CDMA_ENGINE_MAIN_CTRL(port) + 0x58)
#define CDMA_CSR_REG_A4S(port)             (CDMA_ENGINE_MAIN_CTRL(port) + 0x120)
#define CDMA_CSR_REG_DES_ADDR_L32(port)    (CDMA_ENGINE_MAIN_CTRL(port) + 0x2c)
#define CDMA_CSR_REG_DES_ADDR_H1(port)     (CDMA_ENGINE_MAIN_CTRL(port) + 0x30)
#define CDMA_CSR_REG_DES_RW_ADDR(port)     (CDMA_ENGINE_MAIN_CTRL(port) + 0x240)
#define C2C_CFG_BASE_ADDR(c2c_sys_id)      (0x6c00000000 + c2c_sys_id * 0x2000000)
#define C2C_TOP_BASE_ADDR(c2c_sys_id)      (C2C_CFG_BASE_ADDR(c2c_sys_id) + 0x7d0000)
#define C2C_CFG_START_ADDR(c2c_sys_id)     (C2C_CFG_BASE_ADDR(c2c_sys_id))
#define C2C_CFG_END_ADDR(c2c_sys_id)       (C2C_CFG_BASE_ADDR(c2c_sys_id) + 0x1ffffff)
#define CXP_CFG_BASE_ADDR                  (0x6c08000000)
#define CXP_TOP_BASE_ADDR                  (CXP_CFG_BASE_ADDR + 0x7d0000)
#define CXP_CFG_START_ADDR                 (CXP_CFG_BASE_ADDR)
#define CXP_CFG_END_ADDR                   (CXP_CFG_BASE_ADDR + 0x1ffffff)

#define API_MESSAGE_EMPTY_SLOT_NUM         2

#define COUNT_RESERVED_DDR_INSTR           0x3000000 // reserve 48M

#define CLINT_REG_MHART_ID                 0x6844001000

// Firmware Package(FWP) for single core
// +------------------+-------+------+-----+------+------+----------+
// | Firmware program | API   | PMU  | PMU | PMU  | PMU  | Reserved |
// |                  | param | GDMA | TIU | SDMA | SDMA |          |
// +------------------+-------+------+-----+------+------+----------+
// +-----------------------------FW_SIZE----------------------------+

// Data Package(DP) for single core
// +-----+-----+-----+---+--------+
// |Idata|Odata|Coeff|imm|Reserved|
// +-----+-----+-----+---+--------+
// +--PLD_MULTI_TASK_DATA_SIZE----+

// Frimware bin for multicore
// +-----+-----+-----+-----+----+----+----+----+----+----+
// |FWP 0|FWP 1| ... |FWP 7|DP 0|DP 1|DP 2|....|....|DP 7|
// +-----+-----+-----+-----+----+----+----+----+----+----+
// +------FW_SIZE*8--------+PLD_MULTI_TASK_DATA_SIZE * 8-+

#define PLD_BASE_ADDR                      (PLD_MULTI_TASK_DATA_SIZE * 1ul * CORE_ID)
#define FIRMWARE_START_ADDR                (GLOBAL_MEM_START_ADDR + CORE_ID * FW_SIZE)
#define FIRMWARE_MAX_SIZE                  (1*1024*1024)
#define PLD_MESSAGE_START_ADDR             (FIRMWARE_START_ADDR + FIRMWARE_MAX_SIZE)
#define PLD_SHARE_REG_BASE_ADDR            (PLD_MESSAGE_START_ADDR)  // should use SHARE_REG_BASE_ADDR, hack for tv_gen
#define PLD_MESSAGE_MAX_SIZE               (0)
#define PLD_PMU_GDMA_START_ADDR            (PLD_MESSAGE_START_ADDR + PLD_MESSAGE_MAX_SIZE)
#define PLD_PMU_GDMA_MAX_SIZE              (20*1024*1024)
#define PLD_PMU_TIU_START_ADDR             (PLD_PMU_GDMA_START_ADDR+ PLD_PMU_GDMA_MAX_SIZE)
#define PLD_PMU_TIU_MAX_SIZE               (40*1024*1024)
#define PLD_PMU_SDMA_START_ADDR            (PLD_PMU_TIU_START_ADDR + PLD_PMU_TIU_MAX_SIZE)
#define PLD_PMU_SDMA_MAX_SIZE              (20*1024*1024)
#define PLD_PMU_CDMA_START_ADDR            (PLD_PMU_SDMA_START_ADDR + PLD_PMU_SDMA_MAX_SIZE)
#define PLD_PMU_CDMA_MAX_SIZE              (5*1024*1024)


#endif  /* MEMMAP_H */
