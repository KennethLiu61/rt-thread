#ifndef SG_API_PLD_H_
#define SG_API_PLD_H_

typedef enum {
  PLD_TEST_ID_GDMA_STRIDE           = 0,
  PLD_TEST_ID_RW_DDR                = 1,
  PLD_TEST_ID_SEND_INSTRUCTION      = 2,
  PLD_TEST_ID_BASE_BDC              = 3,
  PLD_TEST_ID_RW_SMEM               = 4,
  PLD_TEST_ID_GDMA_S2S              = 5,
  PLD_TEST_ID_GDMA_BDC_PARALLEL     = 6,
  PLD_TEST_ID_NMS                   = 7,
  PLD_TEST_ID_GDE                   = 8,
  PLD_TEST_ID_TPU_SYS               = 9,
  PLD_TEST_ID_GDMA_PERF             = 10,
  PLD_TEST_ID_RW_L2                 = 11,
  PLD_TEST_ID_GDMA_DDR_SRAM         = 12,
  PLD_TEST_ID_AR_STRIDE             = 13,
  PLD_TEST_ID_BANK_CONFLICT         = 14,
  PLD_TEST_ID_VC                    = 15,
  PLD_TEST_ID_RQ_DQ                 = 16,
  PLD_TEST_ID_GDMA_GATHER           = 17,
  PLD_TEST_ID_GDMA_SCATTER          = 18,
  PLD_TEST_ID_GDMA_TENSOR_CWTRANS   = 19,
  PLD_TEST_ID_GDMA_MATRIX           = 20,
  PLD_TEST_ID_GDMA_DDR_SMEM         = 21,
  PLD_TEST_ID_NPU_STATIC            = 22,
  PLD_TEST_ID_FUSED_LINEAR          = 23,
  PLD_TEST_ID_HAU_SRAM              = 24,
  PLD_TEST_ID_AR                    = 25,
  PLD_TEST_ID_RW_LMEM               = 26,
  PLD_TEST_ID_GDMA_GENERAL          = 27,
  PLD_TEST_ID_SPECIAL_FUNC          = 28,
  PLD_TEST_ID_GATHER                = 29,
  PLD_TEST_ID_SCATTER               = 30,
  PLD_TEST_ID_SG_OTHER              = 31,
  PLD_TEST_ID_CW_TRANS              = 32,
  PLD_TEST_ID_GDE_PERF              = 33,
  PLD_TEST_ID_MM2_PERF              = 34,
  PLD_TEST_ID_CONV_PERF             = 35,
  PLD_TEST_ID_GDMA_MASK_SELECT      = 36,
  PLD_TEST_ID_GDMA_GENERAL_CWTRANS  = 37,
  PLD_TEST_ID_MM_PERF               = 38,
  PLD_TEST_ID_SORT_PERF             = 39,
  PLD_TEST_ID_FUSED_CMP_PERF        = 40,
  PLD_TEST_ID_PORD_PERF             = 41,
  PLD_TEST_ID_GDMA_LMEM_L2          = 42,
  PLD_TEST_ID_NMS_PERF              = 43,
  PLD_TEST_ID_SEG                   = 44,
  PLD_TEST_ID_GDMA_BDC_HAU_PARALLEL_PERL = 45,
  PLD_TEST_ID_EXP                   = 46,
  PLD_TEST_ID_SORT_GDMA_PARALLEL    = 47,
  PLD_TEST_ID_GDMA_STRIDE2          = 48,
  PLD_TEST_ID_TPU_SYS_PERF          = 49,
  PLD_TEST_ID_CONV_LARGE            = 50,
  PLD_TEST_ID_GDMA_MATRIX_PERF      = 51,
  PLD_TEST_ID_STRIDE_MV_PERF        = 52,
  PLD_TEST_ID_GDMA_REVERSE          = 53,
  PLD_TEST_ID_GDMA_COMPRESS         = 54,
  PLD_TEST_ID_GDMA_MEM_RW           = 55,
  PLD_TEST_ID_HAU_MEM_RW            = 56,
  PLD_TEST_ID_DES_TEST              = 57,
  PLD_TEST_ID_READ_MSG_REG          = 58,
  PLD_TEST_ID_RANDOM_GEN            = 59,
  PLD_TEST_ID_GDMA_NONZERO          = 60,
  PLD_TEST_ID_GDMA_LOSSY_COMPRESS   = 61,
  PLD_TEST_ID_GDMA_LOSSY_DECOMPRESS = 62,
  PLD_TEST_ID_GDMA_TRANSFER         = 63,
  PLD_TEST_ID_GDMA_TENSOR           = 64,
  // PLD_TEST_ID_GDMA_ARE              = 65,
  // PLD_TEST_ID_SDMA_ARE              = 66,
  PLD_TEST_ID_CONV_TF32             = 67,
  PLD_TEST_ID_MM2_NN_TF32           = 68,
  PLD_TEST_ID_CDMA_ARE_SND          = 69,
  PLD_TEST_ID_CDMA_ARE_RCV          = 70,
  PLD_TEST_ID_RANDOM_MASK_PERF      = 71,
  PLD_TEST_ID_GDMA_GATHER_PERF      = 72,
  PLD_TEST_ID_GDMA_SCATTER_PERF     = 73,
  PLD_TEST_ID_GDMA_REVERSE_PERF     = 74,
  PLD_TEST_ID_GDMA_TENSOR_MOVE_FP20 = 75,
  PLD_TEST_ID_DMA_FILTER_NONZERO    = 76,
  PLD_TEST_ID_GDMA_DDR_TO_L2_ARE    = 77,
  PLD_TEST_ID_GDMA_LMEM_TO_L2_ARE   = 78,
  PLD_TEST_ID_GDMA_L2_TO_L2_ARE     = 79,
  PLD_TEST_ID_GDMA_LOSSY_COMPRESS_ARE    = 80,
  PLD_TEST_ID_GDMA_LOSSY_DECOMPRESS_ARE  = 81,
  PLD_TEST_ID_SDMA_DDR_TO_L2_ARE         = 82,
  PLD_TEST_ID_SDMA_L2_TO_L2_ARE          = 83,
  PLD_TEST_ID_SDMA_LOSSY_COMPRESS_ARE    = 84,
  PLD_TEST_ID_SDMA_LOSSY_DECOMPRESS_ARE  = 85,
  PLD_TEST_ID_CONV_BW_PERF          = 86,
  PLD_TEST_ID_CONV_BW_FP8           = 87,
  PLD_TEST_ID_CONV_BW_TF32          = 88,
  PLD_TEST_ID_CONV_BW_FP32          = 89,
  PLD_TEST_ID_SRCH_BIN              = 90,
  PLD_TEST_ID_FUSED_CMP             = 91,
  PLD_TEST_ID_CDMA_WRITE            = 92,
  PLD_TEST_ID_CDMA_K2K_SEND         = 93,
  PLD_TEST_ID_CDMA_K2K_LOSSY_COMPRESS    = 94,
  PLD_TEST_ID_CDMA_K2K_LOSSY_DECOMPRESS  = 95,
  PLD_TEST_ID_CDMA_K2K_SEND_FP20    = 96,
  PLD_TEST_ID_GDMA_STRESS_TEST      = 97,
} pld_test_id_t;

typedef struct sg_api_pld_test {
  union {
    pld_test_id_t test_id;
    char padding[8];  //a temporary solution for ARM 64bit align, need remove later
  };
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
} __attribute__((packed)) sg_api_pld_test_t;

typedef struct sg_api_pld_sys_power {
  unsigned long long data_global_addr;
  int  gdma_shape[4];
  int  tiu_shape0[4];
  int  tiu_shape1[4];
  int  tiu_shape_res[4];
  int cmd_id;
  int dtype;
  int loops;
} __attribute__((packed)) sg_api_pld_sys_power_t;

typedef struct sg_api_pld_send_instruction {
  union {
    pld_test_id_t test_id;
    char padding[8]; //a temporary solution for ARM 64bit align, need remove later
  };
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int                loops;
  int                N;
  int                C;
  int                H;
  int                W;
} __attribute__((packed)) sg_api_pld_send_instruction_t;

typedef struct {
  pld_test_id_t      test_id;
  unsigned long long start_addr;
  unsigned long long end_addr;
  unsigned long long elem_num;
  unsigned long long elem_stride;
  int                elem_dsize;
} __attribute__((packed)) sg_api_pld_mem_rw_test_t;
typedef struct {
  pld_test_id_t      test_id;
  unsigned long long tpu_cmd_addr;
  unsigned long long gdma_cmd_addr;
  unsigned long long hau_cmd_addr;
  unsigned long long start_addr;
  unsigned long long end_addr;
  unsigned long long addr_stride;
  unsigned long long output_data_addr;
  unsigned long long output_index_addr;
  int                tpu_cmd_num;
  int                gdma_cmd_num;
  int                hau_cmd_num;
  int                tpu_cmd_size;
  int                gdma_cmd_size;
  int                hau_cmd_size;
} __attribute__((packed)) sg_api_pld_des_test_t;

typedef struct {
  pld_test_id_t      test_id;
  unsigned long long src_addr;
  unsigned long long bin_addr;
  unsigned long long dst_addr;
  int                N;
  int                C;
  int                H;
  int                W;
  int                sign;
  int                side;
  int                bin_w;
  int                src_dtype;
  int                dst_dtype;
} __attribute__((packed)) sg_api_pld_srch_bin_test_t;

typedef struct {
  union {
    pld_test_id_t test_id;
    char padding[8];  //a temporary solution for ARM 64bit align, need remove later
  };
  unsigned long long src_addr;
  unsigned long long dst0_addr;
  unsigned long long dst1_addr;
  int                N;
  int                C;
  int                H;
  int                W;
  int                a_is_sign;
  int                A_dtype;
  int                C_dtype;
} __attribute__((packed)) sg_api_pld_fused_cmp_t;

typedef struct {
  pld_test_id_t test_id;
  unsigned long long input_addr;
  unsigned long long output_addr;
  int N;
  int C;
  int H;
  int W;
  int dtype;
  int reduce_op;
  unsigned int reduce_test_round;
} __attribute__((packed)) sg_api_pld_cdma_k2k_t;
#endif
