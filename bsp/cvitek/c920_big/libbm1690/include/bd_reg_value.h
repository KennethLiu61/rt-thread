#ifndef BD_REG_VALUE_H_
#define BD_REG_VALUE_H_

/*
 * The value of enum is according to chip registers
 * So NOT change the value at will !
 */

typedef enum {
    CONV = 0,
    PD   = 1,
    MM   = 2,
    AR   = 3,
    RQDQ = 4,
    TRANS_BC = 5,
    SG   = 6,
    LAR  = 7,
    RANDOM_GEN = 8,
    SFU  = 9,
    LIN  = 10,
    SYS_TRWR = 12,
    CMP  = 13,
    VC   = 14,
    SYS  = 15
} TSK_TYPE;

typedef enum {
    PAD_CONSTANT    = 0,
    PAD_REFLECTION  = 1,
    PAD_REPLICATION = 2,
    PAD_CIRCULAR    = 3,
    PAD_MODE_NUM    = 4
} PAD_MODE;

typedef enum {
    CONV_NORMAL = 0,
    CONV_BW = 1,
    CONV_TF32 = 2,
    CONV_DW_TF32 = 4,
    CONV_OP_NUM = 5
} CONV_OP;

typedef enum {
    LIN_MAC = 1,
    LIN_ADD_SQR = 20,
    LIN_SUB_SQR = 21
} LIN_OP;

typedef enum {
    SFU_TAYLOR_4X = 12,
    SFU_TAYLOR    = 13,
    SFU_NORM      = 15,
    SFU_RSQ       = 17
} SFU_OP;

typedef enum {
    CMP_GT_AND_SG = 22,
    CMP_SG = 23,
    CMP_SE = 24,
    CMP_LT_AND_SL = 25,
    CMP_SL = 26,
    CMP_SRCH_BIN = 27
} CMP_OP;

typedef enum {
    MM_NORMAL = 1,
    MM_WRQ = 2,
    MM_WRQ_RELU = 3,
    MM_NN = 4,
    MM_NT = 5,
    MM_TT = 6,
    MM_NN_TF32 = 7,
    MM_NT_TF32 = 8,
    MM_TT_TF32 = 9,
} MM_OP;

typedef enum {
    AR_MUL = 0,
    AR_NOT = 1,
    AR_ADD = 2,
    AR_SUB = 3,
    AR_MAX = 4,
    AR_MIN = 5,
    AR_LOGIC_SHIFT = 6,
    AR_AND = 7,
    AR_OR = 8,
    AR_XOR = 9,
    AR_SG = 10,
    AR_SE = 11,
    AR_DIV = 12,
    AR_SL = 13,
    AR_DATA_CONVERT = 14,
    AR_ADD_SATU = 15,
    AR_SUB_SATU = 16,
    // AR_CLAMP = 17,
    AR_MAC = 18,
    AR_COPY = 19,
    AR_MUL_SATU = 20,
    AR_ARITH_SHIFT = 21,
    AR_ROTATE_SHIFT = 22,
    // AR_MULDHR = 23, // not support
    // AR_EU_IDX_GEN = 24,
    // AR_NPU_IDX_GEN = 25,
    AR_ABS = 26,
    AR_FSUBABS = 27,
    // AR_COPY_MB = 28, // not support
    AR_GET_FIRST_ONE = 29,
    AR_GET_FIRST_ZERO = 30
} AR_OP;

typedef enum {
    PD_DEPTHWISE = 0,
    PD_AVG_POOLING = 1,
    PD_MIN_POOLING = 3,
    PD_MAX_POOLING = 4,
    PD_ROI_DEPTHWISE = 5,
    PD_ROI_AVG_POOLING = 6,
    PD_ROI_MAX_POOLING = 7,
    PD_ROI_MIN_POOLING = 8,
    PD_MIN_POOLING_W_INDEX = 9,
    PD_MAX_POOLING_W_INDEX = 10,
} PD_OP;

typedef enum {
    LANE_COPY = 2,
    LANE_BROAD = 3,
    STATIC_BROAD = 4,
    STATIC_DISTRIBUTE = 5,
} BC_OP;

typedef enum {
    TRAN_C_W_TRANSPOSE = 0,
    TRAN_W_C_TRANSPOSE = 1,
} TRAN_OP;

typedef enum {
    PL_gather_d1coor = 0,
    PL_gather_d2coor = 1,
    // PL_gather_rec = 2,
    PL_scatter_d1coor = 3,
    PL_scatter_d2coor = 4,
    PE_S_gather_d1coor = 5,
    PE_S_scatter_d1coor = 6,
    PE_M_gather_d1coor = 7,
    PE_S_mask_select = 8,
    PE_S_nonzero = 9,
    // PE_S_scatter_pp_d1coor = 10,
    PE_S_gather_hzd = 13,
    PE_S_scatter_hzd = 14,
    PE_S_mask_selhzd = 15,
    PE_S_nonzero_hzd = 16,
    PE_S_gather_line = 17,
    PE_S_scatter_line = 18,
    // PE_S_mask_seline = 19,
} SG_OP;

typedef enum {
    RQ_0 = 0,
    RQ_1 = 1,
    DQ_0 = 3,
    DQ_1 = 4,
} RQDQ_OP;

typedef enum {
    // BD_INSTR_BARRIER = 0, // useless
    BD_SYS_SPB = 1, // software power boot
    BD_SYS_SWR = 2, // set bd lane_mask
    BD_SYS_SWR_FROM_LMEM = 3, // set bd lane mask
    BD_SYS_SWR_COL_FROM_LMEM = 4, // set bd lane mask
    // BD_SYNC_ID = 5,
    // BD_DATA_BARRIER = 6, // useless
    BD_SYS_SEND_MSG = 8,
    BD_SYS_WAIT_MSG = 9,
    BD_SYS_FORK = 10,
    BD_SYS_JOIN = 11,
    BD_SYS_EXIT = 12,
    BD_SYS_RANDOM_SEED = 13,
    BD_SYS_NOP = 30,
    BD_SYS_END = 31 // end instruction for descriptor mode
} BD_SYS_TYPE;

typedef enum {
    PRNG = 0, // use global state to generate random number
    PRNG_WITH_INTIAL_SEED = 1, // set seed
    PRNG_WITH_LOADED_STATES = 2 // load state from lmem
} RAND_OP;

#endif /* OP_CODE_H_ */
