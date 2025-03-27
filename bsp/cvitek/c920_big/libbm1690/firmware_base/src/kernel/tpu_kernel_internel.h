#pragma once

#include "atomic_gen_cmd.h"
#include "atomic_tensor_arithmetic_gen_cmd.h"
#include "atomic_pooling_depthwise_gen_cmd.h"
#include "atomic_random_gen_gen_cmd.h"
#include "atomic_gdma_gen_cmd.h"
#include "atomic_sys_gen_cmd.h"
#include "atomic_cdma_gen_cmd.h"
#include "atomic_sdma_gen_cmd.h"
#include "atomic_conv_bw_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"

#define MAX_FUNC_NAME_LENGTH (64)
#define MAX_NUM_KERNEL_FUNCS (2048)
#ifndef NO_USE
#define NO_USE (0)
#endif

#define SIGN(dtype) ((dtype) & 0x1)
#define PRECISION(dtype) (((dtype) >> 1) & 0xf)
#define FP8TYPE(dtype) ((dtype) >> 5)
#define WIDTH(dtype) tpu_data_type_bits(dtype)
#define DSIZE(dtype) tpu_data_type_size(dtype)
#define ALIGNED_OR_USER(stride) ((stride) == NULL ? 0 : 3)
#define BDC_NODE (id_node.in_parallel_state ? &bdc_id_node : &id_node)
#define GDMA_NODE (id_node.in_parallel_state ? &gdma_id_node : &id_node)

#if defined(USING_MULTI_THREAD_ENGINE)
#define CHECK_BDC_OVERFLOW check_bdc_overflow()
#define CHECK_GDMA_OVERFLOW check_gdma_overflow()
#else
// Let CMD storer handle overflow, it knows about descriptor mode, cmd grouping and whatnots.
#define CHECK_BDC_OVERFLOW
#define CHECK_GDMA_OVERFLOW
#endif

#define IS_FLOAT(dtype) (dtype==DT_FP32 || dtype==DT_FP16 || dtype==DT_BFP16 || dtype==DT_FP8E5M2 || dtype==DT_FP8E4M3 || dtype==DT_FP20)
#define VALUE_OR_ADDR(t) \
    (t->type == SCALAR ? t->context.scalar.u32 : t->context.addr)
#define SWAP(a, b, t) do {(t) = (a); (a) = (b); (b) = (t);} while(0)
#define LOCAL_INDEX_MUST_BE_0(addr) ASSERT_INFO(tpu_npu_index(addr)==0, "addr=0x%x, index=%d, offset=0x%x", addr, tpu_npu_index(addr), tpu_npu_addr(addr))
typedef struct {
    char name[MAX_FUNC_NAME_LENGTH + 1];
    tpu_kernel_func_t func;
} func_pair_t;
typedef struct {
    u32 physical_core_id;
    u64 group_num;
    u64 workitem_num;
    u32 group_id;
    u32 workitem_id;
} __attribute__((packed)) tpu_groupset_info_t;

#ifdef USING_CMODEL
#define THREAD __thread
#else
#define THREAD
#endif
extern THREAD bool check_id_node;
extern THREAD CMD_ID_NODE id_node;
extern THREAD CMD_ID_NODE bdc_id_node;
extern THREAD CMD_ID_NODE gdma_id_node;
extern int rsqrt_iter_num;
extern int div_iter_num;
extern int sfu_taylor_sin_len;
extern int sfu_taylor_cos_len;
extern int sfu_taylor_tan_len;
extern int sfu_taylor_arcsin_len;

#define HANDLE_LOCAL_STRIDE(p_valid_stride, p_stride, tmp_stride, npu_index, p_shape, dtype) \
    const dim4 *p_valid_stride = p_stride; \
    dim4 tmp_stride;                       \
    if (p_stride == NULL) {                \
        tpu_aligned_stride(                \
            &tmp_stride,                   \
            npu_index,                     \
            p_shape,                       \
            dtype);                        \
        p_valid_stride = &tmp_stride;      \
    }
#define HANDLE_GLOBAL_STRIDE(p_valid_stride, p_stride, tmp_stride, p_shape) \
    const dim4 *p_valid_stride = p_stride; \
    dim4 tmp_stride;                       \
    if (p_stride == NULL) {                \
        tpu_continuous_stride(             \
            &tmp_stride,                   \
            p_shape);                      \
        p_valid_stride = &tmp_stride;      \
    }

static inline void check_bdc_overflow() {
    if (!check_id_node || id_node.in_parallel_state)
        return;
    if (id_node.bd_cmd_id > CMD_ID_OVERFLOW_VALUE) {
        poll_all_engine_done(&id_node);
        resync_cmd_id(&id_node);
    }
}
static inline void check_gdma_overflow() {
    if (!check_id_node || id_node.in_parallel_state)
        return;
    if (id_node.gdma_cmd_id > CMD_ID_OVERFLOW_VALUE) {
        poll_all_engine_done(&id_node);
        resync_cmd_id(&id_node);
    }
}

#define REQUANT_INFO_DECLARE(requant)                                                 \
    int do_requant = requant != NULL;                                                 \
    int shift = 0;                                                                    \
    int yzp = 0;                                                                      \
    local_addr_t requant_addr = 0;                                                    \
    int is_perchannel = 0;                                                            \
    int do_sym_saturate = 0;                                                           \
    rounding_mode_t round_mode = RM_HALF_AWAY_FROM_ZERO;                              \
    do                                                                                \
    {                                                                                 \
        if ((do_requant))                                                             \
        {                                                                             \
            is_perchannel = (requant)->is_perchannel;                                 \
            do_sym_saturate = (requant)->do_sym_saturate;                               \
            round_mode = (requant)->round_mode;                                       \
            if (is_perchannel)                                                        \
            {                                                                         \
                requant_addr = (requant)->addr;                                       \
            }                                                                         \
            else                                                                      \
            {                                                                         \
                requant_addr = (requant)->multiplier;                                 \
                yzp = (requant)->yzp;                                                 \
                shift = (requant)->shift;                                             \
                TPUKERNEL_ASSERT((requant)->yzp >= -32768 && (requant)->yzp < 32768); \
                TPUKERNEL_ASSERT((requant)->shift >= -128 && (requant)->shift < 128); \
            }                                                                         \
        }                                                                             \
    } while (0)

#define OPTIONAL_INFO_DECLARE(opt, default_dtype)                           \
    data_type_t opt##_dtype = default_dtype;                                \
    int opt##_is_const = 1;                                                 \
    int opt##_value = 0;                                                    \
    do                                                                      \
    {                                                                       \
        if ((opt))                                                          \
        {                                                                   \
            opt##_is_const = (opt)->is_const;                               \
            opt##_value = (opt)->is_const ? (opt)->addr : (opt)->value.u32; \
            opt##_dtype = (opt)->dtype;                                     \
        }                                                                   \
    } while (0)

#define NOT_SUPPORT(func) \
    TPUKERNEL_ASSERT_INFO(0, "%s is not supported on SG2260", func);\

void set_tpu_groupset_info( tpu_groupset_info_t *tpu_groupset);
