#pragma once
#include "memmap.h"
#include "config.h"

#define CDMA_CSR_BASE_ADDR        (CDMA_ENGINE_MAIN_CTRL(DEBUG_CDMA_PORT))
#define CDMA_CSR_0            0x0
#define CDMA_CSR_RCV_ADDR_H32        0x4
#define CDMA_CSR_RCV_ADDR_M16        0x8
#define CDMA_CSR_INTER_DIE_RW_V2        0xc
#define CDMA_CSR_4            0x10
#define CDMA_CSR_DES_ADDR_L32        0x2c
#define CDMA_CSR_DES_ADDR_H1        0x30
#define CDMA_CSR_PMU_START_ADDR_L32    0x34
#define CDMA_CSR_PMU_START_ADDR_H1    0x38
#define CDMA_CSR_PMU_END_ADDR_L32    0x3c
#define CDMA_CSR_PMU_END_ADDR_H1    0x40
#define CDMA_CSR_INT_DISABLE        0x54
#define CDMA_CSR_INT            0x58
#define CDMA_CSR_BASE_ADDR_REGINE    0x5c
#define CDMA_CSR_CFG_STOPPER        0x10c
#define CDMA_CSR_PMU_WR_ADDR_L32    0x110
#define CDMA_CSR_PMU_WR_ADDR_H1         0x114
#define CDMA_CSR_INTRA_DIE_RW_V2        0x23c
#define CDMA_CSR_DES_RW            0x240

#define CDMA_CSR_INTER_DIE_READ_ADDR_L4        0
#define CDMA_CSR_INTER_DIE_READ_ADDR_H4        4
#define CDMA_CSR_INTER_DIE_WRITE_ADDR_L4    8
#define CDMA_CSR_INTER_DIE_WRITE_ADDR_H4    12

#define CDMA_CSR_INTRA_DIE_READ_ADDR_L4        0
#define CDMA_CSR_INTRA_DIE_READ_ADDR_H4        4
#define CDMA_CSR_INTRA_DIE_WRITE_ADDR_L4    8
#define CDMA_CSR_INTRA_DIE_WRITE_ADDR_H4    12

#define CDMA_CSR_RCV_CMD_OS        15

typedef enum {
    C2C_PCIE_X8_0 = 0b0101,
    C2C_PCIE_X8_1 = 0b0111,
    C2C_PCIE_X4_0 = 0b0100,
    C2C_PCIE_X4_1 = 0b0110,
    CXP_PCIE_X8 = 0b1010,
    CXP_PCIE_X4 = 0b1011,
} CDMA_PCIE_ROUTE;

typedef enum {
    // RN: K2K; RNI: CMN
    AXI_RNI = 0b1001,
    AXI_RN = 0b1000,
} CDMA_AXI_OUTPUT_NOC;

#define C2C_PCIE_CTRL            C2C_PCIE_X8_1
#define C2C0_CFG_BASE_ADDR         0x6c00000000
#define C2C_CFG_BASE_ADDR_PAIR        C2C0_CFG_BASE_ADDR
#define CDMA2_CMD_BASE_ADDR_PAIR    (C2C_CFG_BASE_ADDR_PAIR + 0x7b0000)
#define CDMA_CMD_BASE_ADDR_PAIR        CDMA2_CMD_BASE_ADDR_PAIR

#define AXI_NOC                AXI_RN
#define CDMA_TOP_BASE_ADDR        (0x6c007d0000)

#ifndef USING_CMODEL
#define WRITE_CSR_CDMA(offset, val) \
    (*(volatile u32 *)map_to_kaddr(CDMA_CSR_BASE_ADDR + offset) = (val));

#define READ_CSR_CDMA(offset) \
    (*(volatile u32 *)map_to_kaddr(CDMA_CSR_BASE_ADDR + offset))

#define WRITE_TOP_CDMA(offset, val) \
    (*(volatile u32 *)map_to_kaddr(CDMA_TOP_BASE_ADDR + offset) = (val))
#else
#define WRITE_CSR_CDMA(offset, val)
#define READ_CSR_CDMA(offset) 0
#define WRITE_TOP_CDMA(offset, val)
#endif
