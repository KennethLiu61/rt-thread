#include "atomic_tensor_arithmetic_gen_cmd.h"
#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

#define SIGN(dtype) ((dtype) & 0x1)
#define N 4
#define C 64
#define H 32
#define W 64
#define loop 10

static inline u32 get_tensor_size_onelocal_mem(
    u32 n, u32 c, u32 h, u32 w,
    u32 local_mem_addr, bool align, PREC precision)
{
  u32 c_stride = get_local_cstride(h, w, align, precision);
  u32 n_stride = get_local_nstride(c_stride, c, local_mem_addr);
  return n_stride * n * get_bytesize(precision);
}

typedef void(*ar_binary_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    data_type_t);

typedef ar_binary_func_t fp_ar_binary_func_t;

typedef void(*int_ar_binary_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    data_type_t,
    data_type_t,
    data_type_t,
    char,
    rounding_mode_t,
    bool);

typedef void(*ar_shift_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    data_type_t,
    data_type_t,
    data_type_t,
    rounding_mode_t);

typedef void(*ar_select_func_t)(
    local_addr_t,
    local_addr_t,
    local_addr_t,
    scalar_t,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    const dim4 *,
    data_type_t,
    data_type_t);

static const ar_binary_func_t ar_binary_funcs[] = {
    tpu_bdc_and,
    tpu_bdc_or,
    tpu_bdc_xor,
    tpu_bdc_min,
    tpu_bdc_max
};
static const char * const ar_binary_func_str[] = {
    "AR_AND",
    "AR_OR",
    "AR_XOR",
    "AR_MIN",
    "AR_MAX"
};

static const fp_ar_binary_func_t fp_ar_binary_funcs[] = {
    tpu_bdc_fp_add,
    tpu_bdc_fp_sub,
    tpu_bdc_fp_mul
};

static const char * const fp_ar_binary_func_str[] = {
    "AR_ADD",
    "AR_SUB",
    "AR_MUL"
};

static const int_ar_binary_func_t int_ar_binary_funcs[] = {
    tpu_bdc_int_add,
    tpu_bdc_int_sub,
    tpu_bdc_int_mul
};
static const char * const int_ar_binary_func_str[] = {
    "AR_ADD_SATU",
    "AR_SUB_SATU",
    "AR_MUL_SATU"
};

static const ar_shift_func_t ar_shift_funcs[] = {
    tpu_bdc_arithmetic_shift,
    tpu_bdc_logical_shift,
};

static const char * const ar_shift_func_str[] = {
    "AR_ARITH_SHIFT",
    "AR_LOGIC_SHIFT"
};

static const ar_select_func_t ar_select_funcs[] = {
    tpu_bdc_greater,
    tpu_bdc_equal,
    tpu_bdc_less
};

static const char * const ar_select_func_str[] = {
    "AR_SG",
    "AR_SE",
    "AR_SL"
};

static const int dtypes[] = {
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_FP16,
    DT_BFP16,
    DT_INT32,
    DT_UINT32,
    DT_FP32,
    DT_FP8E5M2,
    DT_FP8E4M3
};

static const char * const dtypes_str[] = {
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_FP16",
    "DT_BFP16",
    "DT_INT32",
    "DT_UINT32",
    "DT_FP32",
    "DT_FP8E5M2",
    "DT_FP8E4M3"
};

static const int fp_dtypes[] = {
    DT_FP16,
    DT_BFP16,
    DT_FP32,
    DT_FP8E5M2,
    DT_FP8E4M3
};

static const char * const fp_dtypes_str[] = {
    "DT_FP16",
    "DT_BFP16",
    "DT_FP32",
    "DT_FP8E5M2",
    "DT_FP8E4M3"
};

static const int int_dtypes[] = {
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_INT32,
    DT_UINT32
};

static const char * const int_dtypes_str[] = {
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_INT32",
    "DT_UINT32"
};

void nodechip_ar_test() {
    tpu_initialize();

    dim4 shape = {N, C, H, W};
    float len = N * C * H * W;
    int A_size, B_size, R_size;

    unsigned long long A_addr = 0;
    unsigned long long B_addr = 0;
    unsigned long long R_addr = 0;

    u64 ar_start_time = 0ull;
    u64 ar_end_time = 0ull;

    A_size = get_tensor_size_onelocal_mem(N, C, H, W, A_addr, true, FP32);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(N, C, H, W, B_addr, true, FP32);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(N, C, H, W, R_addr, true, FP32);
    TPUKERNEL_ASSERT(R_addr + R_size <= LOCAL_MEM_SIZE);
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);

    // test ar and/or/xor/min/max
    for (unsigned int i = 0; i < sizeof(ar_binary_funcs) / sizeof(ar_binary_funcs[0]); ++i) {
        for (unsigned int d = 0; d < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d) {
            ar_start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; ++iter)
                ar_binary_funcs[i](
                    R_addr,
                    A_addr,
                    B_addr,
                    &shape,
                    NULL,
                    NULL,
                    NULL,
                    (data_type_t)int_dtypes[d]);
            tpu_poll();
            ar_end_time = firmware_timer_get_time_us();
            printf("%s: "
                   "N = %d "
                   "C = %d "
                   "H = %d "
                   "W = %d "
                   "dtype = %s "
                   "loops = %d "
                   "total time = %lldus "
                   "Tops:%.2f"
                   "\n",
                   ar_binary_func_str[i],
                   N,
                   C,
                   H,
                   W,
                   int_dtypes_str[d],
                   loop,
                   ar_end_time - ar_start_time,
                   (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
        }
    }

    // test ar float point add/sub/mul
    for (unsigned int i = 0; i < sizeof(fp_ar_binary_funcs) / sizeof(fp_ar_binary_funcs[0]); ++i) {
        for (unsigned int d = 0; d < sizeof(fp_dtypes) / sizeof(fp_dtypes[0]); ++d) {
            ar_start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; ++iter)
                fp_ar_binary_funcs[i](
                    R_addr,
                    A_addr,
                    B_addr,
                    &shape,
                    NULL,
                    NULL,
                    NULL,
                    (data_type_t)fp_dtypes[d]);
            tpu_poll();
            ar_end_time = firmware_timer_get_time_us();
            printf("%s: "
                   "N = %d "
                   "C = %d "
                   "H = %d "
                   "W = %d "
                   "dtype = %s "
                   "loops = %d "
                   "total time = %lldus "
                   "Tops:%.2f"
                   "\n",
                   fp_ar_binary_func_str[i],
                   N,
                   C,
                   H,
                   W,
                   fp_dtypes_str[d],
                   loop,
                   ar_end_time - ar_start_time,
                   (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
        }
    }

    // test ar fixed_point add/sub/mul, all with saturation
    for (unsigned int i = 0; i < sizeof(int_ar_binary_funcs) / sizeof(int_ar_binary_funcs[0]); ++i) {
        for (unsigned int d0 = 0; d0 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d0) {
            for (unsigned int d1 = 0; d1 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d1) {
                for (unsigned int d2 = 0; d2 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d2) {
                    if (int_ar_binary_funcs[i] == tpu_bdc_int_sub) {
                        if (!SIGN(int_dtypes[d2]))
                            continue;
                    } else {
                        if ((SIGN(int_dtypes[d0]) | SIGN(int_dtypes[d1])) != SIGN(int_dtypes[d2]))
                            continue;
                    }
                    ar_start_time = firmware_timer_get_time_us();
                    for (int iter = 0; iter < loop; ++iter)
                        int_ar_binary_funcs[i](
                            R_addr,
                            A_addr,
                            B_addr,
                            &shape,
                            NULL,
                            NULL,
                            NULL,
                            (data_type_t)int_dtypes[d2],
                            (data_type_t)int_dtypes[d0],
                            (data_type_t)int_dtypes[d1],
                            0,
                            0,
                            true);
                    tpu_poll();
                    ar_end_time = firmware_timer_get_time_us();
                    printf("%s: "
                           "N = %d "
                           "C = %d "
                           "H = %d "
                           "W = %d "
                           "dst_dtype = %s "
                           "src0_dtype = %s "
                           "src1_dtype = %s "
                           "saturation = %d "
                           "loops = %d "
                           "total time = %lldus "
                           "Tops:%.2f"
                           "\n",
                           int_ar_binary_func_str[i],
                           N,
                           C,
                           H,
                           W,
                           int_dtypes_str[d2],
                           int_dtypes_str[d0],
                           int_dtypes_str[d1],
                           1,
                           loop,
                           ar_end_time - ar_start_time,
                           (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
                }
            }
        }
    }

    // test ar shift
    for (unsigned int i = 0; i < sizeof(ar_shift_funcs) / sizeof(ar_shift_funcs[0]); ++i) {
        for (unsigned int d0 = 0; d0 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d0) {
            for (unsigned int d1 = 0; d1 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d1) {
                for (unsigned int d2 = 0; d2 < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++d2) {
                    if (tpu_data_type_size((data_type_t)int_dtypes[d1]) >
                        tpu_data_type_size((data_type_t)int_dtypes[d0])) {
                        continue;
                    } else if (!SIGN(int_dtypes[d1])) {
                        continue;
                    } else if (ar_shift_funcs[i] == tpu_bdc_logical_shift) {
                        if (SIGN(int_dtypes[d0]) || SIGN(int_dtypes[d2]))
                            continue;
                    } else if (ar_shift_funcs[i] == tpu_bdc_arithmetic_shift) {
                        if (SIGN(int_dtypes[d0]) != SIGN(int_dtypes[d2]))
                            continue;
                    }
                    ar_start_time = firmware_timer_get_time_us();
                    for (int iter = 0; iter < loop; ++iter)
                        ar_shift_funcs[i](
                            R_addr,
                            A_addr,
                            B_addr,
                            &shape,
                            NULL,
                            NULL,
                            NULL,
                            (data_type_t)int_dtypes[d2],
                            (data_type_t)int_dtypes[d0],
                            (data_type_t)int_dtypes[d1],
                            0);
                    tpu_poll();
                    ar_end_time = firmware_timer_get_time_us();
                    printf("%s: "
                           "N = %d "
                           "C = %d "
                           "H = %d "
                           "W = %d "
                           "dst_dtype = %s "
                           "src0_dtype = %s "
                           "src1_dtype = %s "
                           "loops = %d "
                           "total time = %lldus "
                           "Tops = %.2f"
                           "\n",
                           ar_shift_func_str[i],
                           N,
                           C,
                           H,
                           W,
                           int_dtypes_str[d2],
                           int_dtypes_str[d0],
                           int_dtypes_str[d1],
                           loop,
                           ar_end_time - ar_start_time,
                           (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
                }
            }
        }
    }

    // test ar select
    scalar_t val;
    val.u32 = 0xffffffff;
    for (unsigned int i = 0; i < sizeof(ar_select_funcs) / sizeof(ar_select_funcs[0]); ++i) {
        for (unsigned int d0 = 0; d0 < sizeof(dtypes) / sizeof(dtypes[0]); ++d0) {
            for (unsigned int d1 = 0; d1 < sizeof(dtypes) / sizeof(dtypes[0]); ++d1) {
                if (tpu_data_type_size((data_type_t)dtypes[d1]) >
                    tpu_data_type_size((data_type_t)dtypes[d0])) continue;
                ar_start_time = firmware_timer_get_time_us();
                for (int iter = 0; iter < loop; ++iter)
                    ar_select_funcs[i](
                        R_addr,
                        A_addr,
                        B_addr,
                        val,
                        &shape,
                        NULL,
                        NULL,
                        NULL,
                        (data_type_t)dtypes[d1],
                        (data_type_t)dtypes[d0]);
                tpu_poll();
                ar_end_time = firmware_timer_get_time_us();
                printf("%s: "
                       "N = %d "
                       "C = %d "
                       "H = %d "
                       "W = %d "
                       "dst_dtype = %s "
                       "src_dtype = %s "
                       "loops = %d "
                       "total time = %lldus "
                       "Tops = %.2f"
                       "\n",
                       ar_select_func_str[i],
                       N,
                       C,
                       H,
                       W,
                       dtypes_str[d1],
                       dtypes_str[d0],
                       loop,
                       ar_end_time - ar_start_time,
                       (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
            }
        }
    }

    // test ar data_convert
    for (unsigned int d0 = 0; d0 < sizeof(dtypes) / sizeof(dtypes[0]); ++d0) {
        for (unsigned int d1 = 0; d1 < sizeof(dtypes) / sizeof(dtypes[0]); ++d1) {
            if ((((data_type_t)dtypes[d0] == DT_FP16) && ((data_type_t)dtypes[d1] == DT_BFP16)) ||
                (((data_type_t)dtypes[d0] == DT_BFP16) && ((data_type_t)dtypes[d1] == DT_FP16))) {
                    continue;
            }
            ar_start_time = firmware_timer_get_time_us();
            for (int iter = 0; iter < loop; ++iter)
                tpu_bdc_cast(R_addr,
                             A_addr,
                             &shape,
                             NULL,
                             NULL,
                             (data_type_t)dtypes[d1],
                             (data_type_t)dtypes[d0],
                             RM_TOWARDS_ZERO);
            tpu_poll();
            ar_end_time = firmware_timer_get_time_us();
            printf("%s: "
                   "N = %d "
                   "C = %d "
                   "H = %d "
                   "W = %d "
                   "dst_dtype = %s "
                   "src_dtype = %s "
                   "loops = %d "
                   "total time = %lldus "
                   "Tops = %.2f"
                   "\n",
                   "AR_DATA_CONVERT",
                   N,
                   C,
                   H,
                   W,
                   dtypes_str[d1],
                   dtypes_str[d0],
                   loop,
                   ar_end_time - ar_start_time,
                   (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
        }
    }

    // test ar fp32 div
    data_type_t div_dtype[] = {DT_FP32, DT_FP16, DT_BFP16};
    const char *div_str[] = {"DIV_DT_FP32", "DIV_DT_FP16", "DIV_DT_BF16"};
    for (size_t i = 0; i < sizeof(div_dtype) / sizeof(div_dtype[0]); i++) {
        ar_start_time = firmware_timer_get_time_us();
        for (int iter = 0; iter < loop; ++iter)
            tpu_bdc_fp_div(
                R_addr,
                A_addr,
                B_addr,
                &shape,
                NULL,
                NULL,
                NULL,
                div_dtype[i]);
        tpu_poll();
        ar_end_time = firmware_timer_get_time_us();
        printf("%s: "
            "N = %d "
            "C = %d "
            "H = %d "
            "W = %d "
            "loops = %d "
            "total time = %lldus "
            "Tops = %.2f"
            "\n",
            div_str[i],
            N,
            C,
            H,
            W,
            loop,
            ar_end_time - ar_start_time,
            (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
    }

    // test ar copy
    for (unsigned int dt = 0; dt < sizeof(int_dtypes) / sizeof(int_dtypes[0]); ++dt) {
        ar_start_time = firmware_timer_get_time_us();
        for (int iter = 0; iter < loop; ++iter)
            tpu_bdc_cpy(R_addr,
                        A_addr,
                        &shape,
                        NULL,
                        NULL,
                        (data_type_t)int_dtypes[dt]);
        tpu_poll();
        ar_end_time = firmware_timer_get_time_us();
        printf("%s: "
               "N = %d "
               "C = %d "
               "H = %d "
               "W = %d "
               "dtype = %s "
               "loops = %d "
               "total time = %lldus "
               "Tops = %.2f"
               "\n",
               "AR_COPY",
               N,
               C,
               H,
               W,
               int_dtypes_str[dt],
               loop,
               ar_end_time - ar_start_time,
               (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
    }

    // test ar fixed_point add/sub/mul using large shape, otherwise c906 gen_cmd may block
    shape.h = 128;
    len = shape.n * shape.c * shape.h * shape.w;
    A_size = get_tensor_size_onelocal_mem(shape.n, shape.c, shape.h, shape.w, A_addr, true, INT8);
    B_addr = ALIGN(A_addr + A_size, LOCAL_BANK_SIZE);
    B_size = get_tensor_size_onelocal_mem(shape.n, shape.c, shape.h, shape.w, B_addr, true, INT8);
    R_addr = ALIGN(B_addr + B_size, LOCAL_BANK_SIZE);
    R_size = get_tensor_size_onelocal_mem(shape.n, shape.c, shape.h, shape.w, R_addr, true, INT8);
    TPUKERNEL_ASSERT(R_addr + R_size <= LOCAL_MEM_SIZE);
    printf("ar test using larger shape\n");
    printf("A_addr = 0x%08llx, B_addr = 0x%08llx, R_addr = 0x%08llx\n", A_addr, B_addr, R_addr);
    for (unsigned int i = 0; i < sizeof(int_ar_binary_funcs) / sizeof(int_ar_binary_funcs[0]); ++i) {
        int d0 = 0, d1 = 0, d2 = 0;
        if (int_ar_binary_funcs[i] == tpu_bdc_int_sub) {
            if (!SIGN(int_dtypes[d2]))
                continue;
        } else {
            if ((SIGN(int_dtypes[d0]) | SIGN(int_dtypes[d1])) != SIGN(int_dtypes[d2]))
                continue;
        }
        ar_start_time = firmware_timer_get_time_us();
        for (int iter = 0; iter < loop; ++iter)
            int_ar_binary_funcs[i](
                R_addr,
                A_addr,
                B_addr,
                &shape,
                NULL,
                NULL,
                NULL,
                (data_type_t)int_dtypes[d2],
                (data_type_t)int_dtypes[d0],
                (data_type_t)int_dtypes[d1],
                0,
                0,
                true);
        tpu_poll();
        ar_end_time = firmware_timer_get_time_us();
        printf("%s: "
                "N = %d "
                "C = %d "
                "H = %d "
                "W = %d "
                "dst_dtype = %s "
                "src0_dtype = %s "
                "src1_dtype = %s "
                "saturation = %d "
                "loops = %d "
                "total time = %lldus "
                "Tops:%.2f"
                "\n",
                int_ar_binary_func_str[i],
                shape.n,
                shape.c,
                shape.h,
                shape.w,
                int_dtypes_str[d2],
                int_dtypes_str[d0],
                int_dtypes_str[d1],
                1,
                loop,
                ar_end_time - ar_start_time,
                (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
    }

    // test ar copy
    ar_start_time = firmware_timer_get_time_us();
    for (int iter = 0; iter < loop; ++iter)
        tpu_bdc_cpy(R_addr,
                    A_addr,
                    &shape,
                    NULL,
                    NULL,
                    (data_type_t)int_dtypes[0]);
    tpu_poll();
    ar_end_time = firmware_timer_get_time_us();
    printf("%s: "
            "N = %d "
            "C = %d "
            "H = %d "
            "W = %d "
            "dtype = %s "
            "loops = %d "
            "total time = %lldus "
            "Tops = %.2f"
            "\n",
            "AR_COPY",
            shape.n,
            shape.c,
            shape.h,
            shape.w,
            int_dtypes_str[0],
            loop,
            ar_end_time - ar_start_time,
            (float)len / ((ar_end_time - ar_start_time) / (float)loop * 1e6));
}
