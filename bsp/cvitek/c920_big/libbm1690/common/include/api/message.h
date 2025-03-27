#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SG_API_ID_RESERVED                         = 0,
    /*
     * API ID that is used in bmlib
     */
    SG_API_ID_MEM_SET                          = 1,
    SG_API_ID_MEM_CPY                          = 2,

    /*
     * Layer APIs
     */
    SG_API_ID_POOLING_PARALLEL                 = 3,
    SG_API_ID_ARITHMETIC                       = 4,
    SG_API_ID_SORT                             = 5,
    SG_API_ID_SCALE                            = 6,
    SG_API_ID_FC                               = 7,
    SG_API_ID_BATCH_MATMUL                     = 8,
    SG_API_ID_POOLING_FIX8B                    = 9,
    SG_API_ID_BNSCALE_FIX8B                    = 10,
    SG_API_ID_ELTWISE                          = 11,
    SG_API_ID_ELTWISE_FIX8B                    = 12,
    SG_API_ID_PRELU                            = 13,
    SG_API_ID_PRELU_FIX8B                      = 14,
    SG_API_ID_RELU                             = 15,
    SG_API_ID_DEPTHWISE_NORMAL                 = 16,
    SG_API_ID_CONV_QUANT_SYM                   = 17,
    SG_API_ID_CONV_FLOAT                       = 18,
    SG_API_ID_DEPTHWISE_FIX8B_NORMAL           = 19,
    SG_API_ID_SOFTMAX                          = 20,
    SG_API_ID_FC_FIX8B                         = 21,
    SG_API_ID_SOFTMAX_TFLITE_FIX8B             = 22,
    SG_API_ID_BATCH_MATMUL_FIX8B               = 23,
    SG_API_ID_REDUCE                           = 24,
    SG_API_ID_BCBINARY_FIX8B                   = 25,
    SG_API_ID_BN_FWD_INF_PARALLEL              = 26,
    SG_API_ID_CONST_BINARY_FIX8B               = 27,
    SG_API_ID_REDUCE_FIX8B                     = 28,
    SG_API_ID_INDEX_SELECT                     = 29,
    SG_API_ID_ACTIVE                           = 30,
    SG_API_ID_TRANSPOSE                        = 31,
    SG_API_ID_CONCAT                           = 32,
    SG_API_ID_UPSAMPLE                         = 33,
    SG_API_ID_DEPTH2SPACE                      = 34,
    SG_API_ID_REQUANT_INT                      = 35,
    SG_API_ID_REQUANT_FLOAT                    = 36,
    SG_API_ID_TOPK                             = 37,
    SG_API_ID_INTERP_PARALLEL                  = 38,
    SG_API_ID_GEMM                             = 39,
    SG_API_ID_DEQUANT_INT                      = 40,
    SG_API_ID_DEQUANT_FLOAT                    = 41,
    SG_API_ID_ROUND_FP                         = 42,
    SG_API_ID_ARG                              = 43,
    SG_API_ID_GATHER_ND_TF                     = 44,
    SG_API_ID_YOLO                             = 45,
    SG_API_ID_SORT_PER_DIM                     = 46,
    SG_API_ID_CONSTANT_FILL                    = 47,
    SG_API_ID_ROI_POOLING                      = 48,
    SG_API_ID_NMS                              = 49,
    SG_API_ID_SSD_DETECT_OUT                   = 50,
    SG_API_ID_ARITHMETIC_SHIFT                 = 51,
    SG_API_ID_ROI_ALIGN                        = 52,
    SG_API_ID_WHERE                            = 53,
    SG_API_ID_PSROI_POOLING                    = 54,
    SG_API_ID_COLUMN_HASH                      = 55,
    SG_API_ID_SHIFT                            = 56,
    SG_API_ID_TILE                             = 57,
    SG_API_ID_SPLIT                            = 58,
    SG_API_ID_YOLOV3_DETECT_OUT                = 59,
    SG_API_ID_MASKED_SELECT                    = 60,
    SG_API_ID_UPSAMPLE_MASK                    = 61,
    SG_API_ID_BINARY_SHIFT                     = 62,
    SG_API_ID_PRIOR_BOX                        = 63,
    SG_API_ID_QUANT_DIV                        = 64,
    SG_API_ID_STRIDESLICE                      = 65,
    SG_API_ID_MASKED_FILL                      = 66,
    SG_API_ID_BATCH2SPACE                      = 67,
    SG_API_ID_GRID_SAMPLE                      = 68,
    SG_API_ID_SHUFFLE_CHANNEL                  = 69,
    SG_API_ID_ADAPTIVE_POOL                    = 70,
    SG_API_ID_BCBINARY_FLOAT                   = 71,
    SG_API_ID_EMBEDDING_BAG                    = 72,
    SG_API_ID_TRIANGULARIZE                    = 73,
    SG_API_ID_EMBEDDING_BAG_FIX8B              = 74,
    SG_API_ID_PROPOSAL_LAYER                   = 75,
    SG_API_ID_INDEX_PUT                        = 76,
    SG_API_ID_DEFORM_GATHER                    = 77,
    SG_API_ID_BILINEAR_INTERPOLATE             = 78,
    SG_API_ID_CONV_REQUANT                     = 79,
    SG_API_ID_GROUP_NORM                       = 80,
    SG_API_ID_LAYER_NORM                       = 81,
    SG_API_ID_COL2IM                           = 82,
    SG_API_ID_REVERSE                          = 83,
    SG_API_ID_CONST_BINARY_FLOAT               = 84,
    SG_API_ID_BATCH_MATMUL_FIX8B_EXT           = 85,
    SG_API_ID_PIXEL_NORM                       = 86,
    SG_API_ID_MSG_SYNC                         = 87,
    SG_API_ID_CLIP_FLOAT                       = 88,
    SG_API_ID_PIXEL_NORM_FIX8B                 = 89,
    SG_API_ID_SPARSE_CONV3D                    = 90,
    SG_API_ID_ONNX_NMS                         = 91,
    SG_API_ID_GROUP_NORM_MULTI_CORE            = 92,
    SG_API_ID_POOLING_FP8                      = 93,
    SG_API_ID_CONV_BW_FLOAT                    = 94,
    SG_API_ID_DEPEND_ID_WRAPAROUND             = 95,
    SG_API_ID_ATTENTION                        = 96,
    SG_API_ID_ATTENTION_FIX8B                  = 97,
    SG_API_ID_RELATIVE_POSITION_ENCODING       = 98,
    SG_API_ID_YOLOV5_DETECT_OUT                = 99,
    SG_API_ID_MSG_SYNC_CDMA                    = 100,
    SG_API_ID_LLAMA2_ATTENTION                 = 101,
    SG_API_ID_CONV_LOOP_TEST                   = 102,
    SG_API_ID_MSG_CENTRAL_STRESS_TEST          = 103,
    SG_API_ID_SWAP_DIM                         = 104,
    SG_API_ID_RMS_NORM                         = 105,
    SG_API_ID_QR_HOUSEHOLDER                   = 106,
    SG_API_ID_KNN_NAIVE                        = 107,
    SG_API_ID_ROI_EXTRACTOR                    = 108,
    SG_API_ID_DEQUANT_HALF_PREC                = 109,
    SG_API_ID_GDMA_BASE_ADDR                   = 110,

    /*
     * API ID that is used in bmlib
     */
    SG_API_ID_MEMCPY_BYTE                      = 136,
    SG_API_ID_MEMCPY_WSTRIDE                   = 137,

    // for PROFILE, same as BM1684
    SG_API_ID_SET_PROFILE_ENABLE               = 986,
    SG_API_ID_GET_PROFILE_DATA                 = 987,
    // for BM1686, SG2260
    SG_API_ID_SET_ENGINE_PROFILE_PARAM         = 988,

    SG_API_ID_UNLOAD_LIBRARY                   = 0x8000000b,
    SG_API_ID_TPU_LOAD_LIB                     = 0x90000001,
    SG_API_ID_TPU_GET_FUNC                     = 0x90000002,
    SG_API_ID_TPU_LAUNCH_FUNC                  = 0x90000003,
    SG_API_ID_TPU_UNLOAD_LIB                   = 0x90000004,


    // SG_API_ID_CONV_LOOP_TEST                   = 0x0ffffff6,
    //  ID for GDMA loop test
    SG_API_ID_GDMA_LOOP_TEST                   = 0x0ffffff7,
    //  ID for TIU loop test
    SG_API_ID_TIU_LOOP_TEST                    = 0x0ffffff8,

    //  ID for TPU full test
    SG_API_ID_TPU_FULL_TEST                    = 0x0ffffff9,

    //  ID for bringup of asic
    SG_API_ID_BRINGUP                          = 0x0ffffffa,
    // ID for fullnet
    SG_API_ID_MULTI_FULLNET                    = 0x0ffffffb,
    SG_API_ID_DYNAMIC_FULLNET                  = 0x0ffffffc,
    // ID for Palladium Test
    SG_API_PLD_TEST                            = 0x0ffffffd,
    // ID for TPUKernel
    SG_API_TPUKERNEL                           = 0x0ffffffe,
    SG_API_TPUKERNEL_MULTICORE                 = 0x0fffffff,

    // SG_API_DEVICE_TEST                         =  0x0FFF0000,
    SG_API_ID_LOCAL_REVERSE_TEST               =  0x0FFF0001,
    SG_API_ID_COMPRESS_NORMAL_TEST             =  0x0FFF0002,
    SG_API_ID_COMPRESS_RACU_TEST               =  0x0FFF0003,
    SG_API_ID_CWTRANS_TEST                     =  0x0FFF0004,
    SG_API_ID_TPU_KERNEL_DEBUG                 =  0x0FFF0005,
    SG_API_ID_MSG_SYNC_DEBUG                   =  0x0FFF0006,
    SG_API_ID_LOSSY_COMPRESS_TEST              =  0x0FFF0007,

    // ID for Test all instructions
    SG_API_TEST_ALL_STRUCTION_PERF                 = 0x8fffffff,
    // QUIT ID
} sg_api_id_t;

#define SG_API_QUIT 0xffffffff


#ifdef __cplusplus
}
#endif

#endif  /* _MESSAGE_H_ */
