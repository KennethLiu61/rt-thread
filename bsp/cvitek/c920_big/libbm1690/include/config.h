#ifndef CONFIG_H_
#define CONFIG_H_

#define NPU_SHIFT               (CONFIG_NPU_SHIFT)
#define NPU_NUM                 (1 << NPU_SHIFT)
#define NPU_MASK                (NPU_NUM - 1)

#define EU_SHIFT                (CONFIG_EU_SHIFT)
#define EU_NUM                  (1 << EU_SHIFT)
#define EU_NUM_32BIT            (EU_NUM)
#define EU_NUM_16BIT            (EU_NUM_32BIT << 1)
#define EU_NUM_8BIT             (EU_NUM_16BIT << 1)
#define EU_NUM_4BIT             (EU_NUM_8BIT << 1)
#define MSG_ID_WIDTH            (CONFIG_MSG_ID_WIDTH)
#define MSG_CNT_BIT             (7)

#define ALIGN_BYTES             (EU_NUM * sizeof(float))

#define LOCAL_MEM_SIZE          (1 << CONFIG_LOCAL_MEM_ADDRWIDTH)
#define L2_SRAM_SIZE            (CONFIG_L2_SRAM_SIZE)
#define LOCAL_MEM_BANKS         (CONFIG_LOCAL_MEM_BANKS)
#define LOCAL_BANK_SIZE         (LOCAL_MEM_SIZE / LOCAL_MEM_BANKS)
#define STATIC_MEM_SIZE         (CONFIG_STATIC_MEM_SIZE)

extern int g_core_id; // Defined in firmware_runtime.c
#define CORE_ID (g_core_id)

#define FW_SIZE                 (136ul * 1024 * 1024)
#define MAX_TPU_CORE_NUM        (CONFIG_MAX_TPU_CORE_NUM)
#define MAX_CDMA_NUM            (CONFIG_MAX_CDMA_NUM)
#define PLD_MULTI_TASK_DATA_SIZE   (CONFIG_PLD_MULTI_TASK_DATA_SIZE)
#define PLD_K2K_CDMA_TEST_PORT   (CONFIG_PLD_K2K_CDMA_TEST_PORT)

#define MAX_GMEM_BIT            (40)
#define MAX_GMEM_SIZE           (1ull << MAX_GMEM_BIT)
#define TAG_MASK                (0x1ful)
#define LMEM_TAG                (0x1ful)
#define SMEM_TAG                (0x1ful)
#define L2M_TAG                 (0x1eul)

#define CORE_OFFSET_BIT         28
#define CORE_OFFSET             (1ull << CORE_OFFSET_BIT)
#define VC_SYS_CORE_OFFSET      (0x2000000UL)

#define NNVLC_ALIGN_SHIFT       7
#define NNVLC_ALIGN_BYTES       (1<<NNVLC_ALIGN_SHIFT)
#define DEBUG_CDMA_PORT         (CONFIG_DEBUG_CDMA_PORT)
#define CDMA_API_NUM            (3)

// frequence MHz
#define FREQ (1000)
// DDR bandwidth GB
#define BW_S2L (68.25)
#define BW_S2S (34.1)
#define BW_L2S (68.25)
#define BW_L2L (68.25)
#define CONTINUOUS_BYTES (128.f)

#define MAX_ROI_NUM             200
#define KERNEL_MEM_SIZE         0

// support INT8,FP32,INT32,INT16,FP16,BFP16,INT4
#define PREC_END                8

#endif
