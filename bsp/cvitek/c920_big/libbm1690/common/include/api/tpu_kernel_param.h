#pragma once
// COP: test kernel step 1: add a LIST_ITEM
#define TPU_KERNEL_LIST(list_type, name) \
    LIST_BEGIN(list_type, name)          \
    LIST_ITEM(tpu_gdma_cpy_S2S)          \
    LIST_ITEM(tpu_gdma_cpy_L2S)          \
    LIST_ITEM(tpu_gdma_cpy_S2L)          \
    LIST_ITEM(tpu_gdma_cpy_L2L)          \
    LIST_ITEM(tpu_gdma_set_C_local)      \
    LIST_ITEM(tpu_gdma_set_C_system)     \
    LIST_ITEM(tpu_parallel_start)        \
    LIST_ITEM(tpu_parallel_end)          \
    LIST_ITEM(tpu_bdc_cpy)               \
    LIST_ITEM(tpu_bdc_fp_div)            \
    LIST_ITEM(tpu_bdc_fp_tunable_C_div)  \
    LIST_ITEM(tpu_bdc_fp_tunable_sqrt)   \
    LIST_ITEM(tpu_bdc_fp_tunable_rsqrt)  \
    LIST_ITEM(tpu_bdc_fp_mul)            \
    LIST_ITEM(tpu_bdc_fp_add)            \
    LIST_ITEM(tpu_bdc_fp_sub)            \
    LIST_ITEM(tpu_bdc_fp_sub_C)          \
    LIST_ITEM(tpu_bdc_fp_add_C)          \
    LIST_ITEM(tpu_bdc_fp_mul_C)          \
    LIST_ITEM(tpu_bdc_fp_C_sub)          \
    LIST_ITEM(tpu_bdc_cast)              \
    LIST_ITEM(tpu_bdc_set_C)             \
    LIST_ITEM(tpu_poll)                  \
    LIST_END(list_type, name)

#define FUNC_DESC_tpu_parallel_start(name) \
    BEGIN_PARAM(name)                      \
    END_PARAM(name)

#define FUNC_DESC_tpu_parallel_end(name) \
    BEGIN_PARAM(name)                    \
    END_PARAM(name)

#define FUNC_DESC_tpu_poll(name) \
    BEGIN_PARAM(name)            \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_cpy_S2S(name) \
    BEGIN_PARAM(name)                    \
    VAL_PARAM(system_addr_t, dst_addr)   \
    VAL_PARAM(system_addr_t, src_addr)   \
    PTR_PARAM(dim4, shape)               \
    OPT_PARAM(dim4, dst_stride)          \
    OPT_PARAM(dim4, src_stride)          \
    VAL_PARAM(data_type_t, dtype)        \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_cpy_S2L(name) \
    BEGIN_PARAM(name)                    \
    VAL_PARAM(local_addr_t, dst_addr)    \
    VAL_PARAM(system_addr_t, src_addr)   \
    PTR_PARAM(dim4, shape)               \
    OPT_PARAM(dim4, dst_stride)          \
    OPT_PARAM(dim4, src_stride)          \
    VAL_PARAM(data_type_t, dtype)        \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_cpy_L2S(name) \
    BEGIN_PARAM(name)                    \
    VAL_PARAM(system_addr_t, dst_addr)   \
    VAL_PARAM(local_addr_t, src_addr)    \
    PTR_PARAM(dim4, shape)               \
    OPT_PARAM(dim4, dst_stride)          \
    OPT_PARAM(dim4, src_stride)          \
    VAL_PARAM(data_type_t, dtype)        \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_cpy_L2L(name) \
    BEGIN_PARAM(name)                    \
    VAL_PARAM(local_addr_t, dst_addr)    \
    VAL_PARAM(local_addr_t, src_addr)    \
    PTR_PARAM(dim4, shape)               \
    OPT_PARAM(dim4, dst_stride)          \
    OPT_PARAM(dim4, src_stride)          \
    VAL_PARAM(data_type_t, dtype)        \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_general_cpy_S2L(name) \
    BEGIN_PARAM(name)                            \
    VAL_PARAM(local_addr_t, dst_addr)            \
    VAL_PARAM(system_addr_t, src_addr)           \
    PTR_PARAM(dim4, dst_shape)                   \
    PTR_PARAM(dim4, src_shape)                   \
    OPT_PARAM(dim4, dst_stride)                  \
    OPT_PARAM(dim4, src_stride)                  \
    VAL_PARAM(data_type_t, dtype)                \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_general_cpy_L2S(name) \
    BEGIN_PARAM(name)                            \
    VAL_PARAM(system_addr_t, dst_addr)           \
    VAL_PARAM(local_addr_t, src_addr)            \
    PTR_PARAM(dim4, dst_shape)                   \
    PTR_PARAM(dim4, src_shape)                   \
    OPT_PARAM(dim4, dst_stride)                  \
    OPT_PARAM(dim4, src_stride)                  \
    VAL_PARAM(data_type_t, dtype)                \
    END_PARAM(name)

#define FUNC_DESC_tpu_gdma_set_C_system(name) \
    BEGIN_PARAM(name)                         \
    VAL_PARAM(system_addr_t, dst_addr)        \
    VAL_PARAM(scalar_t, C)                    \
    PTR_PARAM(dim4, shape)                    \
    OPT_PARAM(dim4, dst_stride)               \
    VAL_PARAM(data_type_t, dtype)             \
    END_PARAM(name)

// COP: test kernel step 2: define tpu_kernel function param as follows
#define FUNC_DESC_tpu_gdma_set_C_local(name) \
    BEGIN_PARAM(name)                        \
    VAL_PARAM(local_addr_t, dst_addr)        \
    VAL_PARAM(scalar_t, C)                   \
    PTR_PARAM(dim4, shape)                   \
    OPT_PARAM(dim4, dst_stride)              \
    VAL_PARAM(data_type_t, dtype)            \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_div(name) \
    BEGIN_PARAM(name)                  \
    VAL_PARAM(local_addr_t, dst_addr)  \
    VAL_PARAM(local_addr_t, src0_addr) \
    VAL_PARAM(local_addr_t, src1_addr) \
    PTR_PARAM(dim4, shape)             \
    OPT_PARAM(dim4, dst_stride)        \
    OPT_PARAM(dim4, src0_stride)       \
    OPT_PARAM(dim4, src1_stride)       \
    VAL_PARAM(data_type_t, dtype)      \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_tunable_C_div(name) \
    BEGIN_PARAM(name)                            \
    VAL_PARAM(local_addr_t, dst_addr)            \
    VAL_PARAM(local_addr_t, src_addr)            \
    VAL_PARAM(scalar_t, C)                       \
    PTR_PARAM(dim4, shape)                       \
    OPT_PARAM(dim4, dst_stride)                  \
    OPT_PARAM(dim4, src_stride)                  \
    VAL_PARAM(data_type_t, dtype)                \
    VAL_PARAM(int, num_iter)                     \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_tunable_sqrt(name) \
    BEGIN_PARAM(name)                           \
    VAL_PARAM(local_addr_t, dst_addr)           \
    VAL_PARAM(local_addr_t, src_addr)           \
    PTR_PARAM(dim4, shape)                      \
    VAL_PARAM(data_type_t, dtype)               \
    VAL_PARAM(int, num_iter)                    \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_tunable_rsqrt(name) \
    FUNC_DESC_tpu_bdc_fp_tunable_sqrt(name)

#define FUNC_DESC_tpu_bdc_fp_mul(name) \
    BEGIN_PARAM(name)                  \
    VAL_PARAM(local_addr_t, dst_addr)  \
    VAL_PARAM(local_addr_t, src0_addr) \
    VAL_PARAM(local_addr_t, src1_addr) \
    PTR_PARAM(dim4, shape)             \
    OPT_PARAM(dim4, dst_stride)        \
    OPT_PARAM(dim4, src0_stride)       \
    OPT_PARAM(dim4, src1_stride)       \
    VAL_PARAM(data_type_t, dtype)      \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_add(name) \
    FUNC_DESC_tpu_bdc_fp_mul(name)

#define FUNC_DESC_tpu_bdc_fp_sub(name) \
    FUNC_DESC_tpu_bdc_fp_mul(name)

#define FUNC_DESC_tpu_bdc_fp_sub_C(name) \
    BEGIN_PARAM(name)                    \
    VAL_PARAM(local_addr_t, dst_addr)    \
    VAL_PARAM(local_addr_t, src_addr)    \
    VAL_PARAM(scalar_t, C)               \
    PTR_PARAM(dim4, shape)               \
    OPT_PARAM(dim4, dst_stride)          \
    OPT_PARAM(dim4, src_stride)          \
    VAL_PARAM(data_type_t, dtype)        \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_fp_C_sub(name) \
    FUNC_DESC_tpu_bdc_fp_sub_C(name)

#define FUNC_DESC_tpu_bdc_fp_add_C(name) \
    FUNC_DESC_tpu_bdc_fp_sub_C(name)

#define FUNC_DESC_tpu_bdc_fp_mul_C(name) \
    FUNC_DESC_tpu_bdc_fp_sub_C(name)

#define FUNC_DESC_tpu_bdc_cast(name)       \
    BEGIN_PARAM(name)                      \
    VAL_PARAM(local_addr_t, dst_addr)      \
    VAL_PARAM(local_addr_t, src_addr)      \
    PTR_PARAM(dim4, shape)                 \
    OPT_PARAM(dim4, dst_stride)            \
    OPT_PARAM(dim4, src_stride)            \
    VAL_PARAM(data_type_t, dst_dtype)      \
    VAL_PARAM(data_type_t, src_dtype)      \
    VAL_PARAM(rounding_mode_t, round_mode) \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_cpy(name)   \
    BEGIN_PARAM(name)                 \
    VAL_PARAM(local_addr_t, dst_addr) \
    VAL_PARAM(local_addr_t, src_addr) \
    PTR_PARAM(dim4, shape)            \
    OPT_PARAM(dim4, dst_stride)       \
    OPT_PARAM(dim4, src_stride)       \
    VAL_PARAM(data_type_t, dtype)     \
    END_PARAM(name)

#define FUNC_DESC_tpu_bdc_set_C(name) \
    BEGIN_PARAM(name)                 \
    VAL_PARAM(local_addr_t, dst_addr) \
    VAL_PARAM(scalar_t, C)            \
    PTR_PARAM(dim4, shape)            \
    OPT_PARAM(dim4, dst_stride)       \
    VAL_PARAM(data_type_t, dtype)     \
    END_PARAM(name)

#define E4(...) E3(E3(E3(E3(E3(E3(E3(E3(E3(E3(__VA_ARGS__))))))))))
#define E3(...) E2(E2(E2(E2(E2(E2(E2(E2(E2(E2(__VA_ARGS__))))))))))
#define E2(...) E1(E1(E1(E1(E1(E1(E1(E1(E1(E1(__VA_ARGS__))))))))))
#define E1(...) __VA_ARGS__

#define EMPTY()
#define TUPLE_AT_2(x, y, ...) y
#define TUPLE_TAIL(x, ...) __VA_ARGS__

#define CHECK(...) TUPLE_AT_2(__VA_ARGS__, 0, )
#define END_EQ_END , 1

#define SCAN(...) __VA_ARGS__
#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b

#define LOOP_() LOOP
#define LOOP(x, y, ...) CAT(LOOP, CHECK(y##_EQ_END))(x, y, __VA_ARGS__)
#define LOOP1(x, ...) (TUPLE_TAIL x)
#define LOOP0(x, y, ...) LOOP_ EMPTY()()((SCAN x, y), __VA_ARGS__)

#define DTC(...) E4(LOOP((), __VA_ARGS__ END))

    typedef struct
    {
        unsigned int debug;
        unsigned int loop;
        unsigned int command_num;
        unsigned char data[0];
} tpu_kernel_debug_batch_param_t;

typedef struct {
    unsigned int func_id;
    unsigned int param_size;
    unsigned char param_data[0];
} tpu_kernel_debug_single_param_t;

static inline tpu_kernel_debug_single_param_t* next_single_param(const tpu_kernel_debug_single_param_t* current){
    unsigned char* next_ptr = ((unsigned char*)current) + sizeof(tpu_kernel_debug_single_param_t) + current->param_size;
    return (tpu_kernel_debug_single_param_t*)next_ptr;
}

#pragma pack(1)
#define PARAM_SEP ;
#define VAL_PARAM(param_type, param_name) param_type param_name PARAM_SEP
#define PTR_PARAM(param_type, param_name) param_type param_name PARAM_SEP
#define OPT_PARAM(param_type, param_name) int param_name##_is_null PARAM_SEP param_type param_name PARAM_SEP
#define BEGIN_PARAM(func) typedef struct {
#define END_PARAM(func) } func##_param_t;
#define LIST_BEGIN(list_type, list_name)
#define LIST_END(list_type, list_name)
#define LIST_ITEM(func) FUNC_DESC_##func(func)
TPU_KERNEL_LIST(,);
#undef LIST_BEGIN
#undef LIST_END
#undef LIST_ITEM
#undef PARAM_SEP
#undef VAL_PARAM
#undef PTR_PARAM
#undef OPT_PARAM
#undef BEGIN_PARAM
#undef END_PARAM
#pragma pack()
