#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"

u64 conv_ops(
    int n,
    int ic,
    int oc,
    int oh,
    int ow,
    int kh,
    int kw,
    bool has_bias,
    bool result_add)
{
    u64 total = (u64)oc * oh * ow * n *
                (kh * kw * ic * 2 - 1 +
                 (has_bias ? 1 : 0) + (result_add ? 1 : 0));
    return total;
}

void nodechip_conv_perf_test()
{
    tpu_initialize();
    int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
    int oc = 64;

    dim4 qshape = {4, 64, 32, 128};
    dim4 tiny_shape = {1, 64, 32, 32};
    dim2 kernel = {3, 3};
    padding_t pad = {1, 1, 1, 1};
    dim2 stride = {1, 1};
    dim2 dilation = {1, 1};
    u64 st, et;
    u64 ops;
    dim4 istride;
    int loop = 5;
#if 1
    dim4 fshape = {2, 64, 32, 64};
    /////////////////// float ////////////////////////////////
    #if 1
    data_type_t fidtype[4] = {DT_FP32, DT_TF32, DT_FP16, DT_FP16};
    data_type_t fodtype[4] = {DT_FP32, DT_FP32, DT_FP16, DT_FP32};
    char *fidtype_c[4] = {"FP32", "TF32", "FP16", "FP16"};
    char *fodtype_c[4] = {"FP32", "FP32", "FP16", "FP32"};
    printf("conv float shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           fshape.n, fshape.c, fshape.h, fshape.w, oc, kernel.h, kernel.w);
    printf("conv conflict shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           tiny_shape.n, tiny_shape.c, tiny_shape.h, tiny_shape.w, oc, kernel.h, kernel.w);
    for (int i = 0; i < 4; i++)
    {
        for (int has_bias = 0; has_bias < 2; has_bias++)
        {
            for (int result_add = 0; result_add < 2; result_add++)
            {
                if (result_add && fodtype[i] != DT_FP32)
                    continue;
                for (int with_str = 0; with_str < 2; with_str++)
                {
                    tpu_compact_stride(&istride, 0, &fshape);
                    st = firmware_timer_get_time_us();
                    for (int l = 0; l < loop; l++)
                    {
                        tpu_bdc_fp_conv2d(
                            0,              // o
                            6 * bank_size,  // i
                            12 * bank_size, // w
                            15 * bank_size, // b
                            &fshape,
                            with_str ? &istride : NULL,
                            oc,
                            &kernel,
                            &pad,
                            &stride,
                            &dilation,
                            fodtype[i],
                            fidtype[i],
                            has_bias,
                            result_add);
                    }
                    tpu_poll();
                    et = firmware_timer_get_time_us();
                    ops = conv_ops(fshape.n, fshape.c, oc, fshape.h, fshape.w,
                                   kernel.h, kernel.w, has_bias, result_add);
                    printf("loop:%d, idtype=%s odtype=%s has_bias=%d result_add=%d "
                           "with_istride=%d time=%lldus Tops/s=%.2f\n",
                           loop, fidtype_c[i], fodtype_c[i], has_bias, result_add,
                           with_str, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
                    ///////////// all opd in one bank ////////////////////////
                    // tpu_compact_stride(&istride, 0, &tiny_shape);
                    // st = firmware_timer_get_time_us();
                    // for (int l = 0; l < loop; l++)
                    // {
                    //     tpu_bdc_fp_conv2d(
                    //         0,                                                                                     // o
                    //         tiny_shape.h * tiny_shape.w * sizeof(float),                                           // i
                    //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2,                                       // w
                    //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float), // b
                    //         &tiny_shape,
                    //         with_str ? &istride : NULL,
                    //         oc,
                    //         &kernel,
                    //         &pad,
                    //         &stride,
                    //         &dilation,
                    //         fodtype[i],
                    //         fidtype[i],
                    //         has_bias,
                    //         result_add);
                    // }
                    // tpu_poll();
                    // et = firmware_timer_get_time_us();
                    // ops = conv_ops(tiny_shape.n, tiny_shape.c, oc, tiny_shape.h, tiny_shape.w,
                    //                kernel.h, kernel.w, has_bias, result_add);
                    // printf("conflict loop=%d: idtype=%s odtype=%s has_bias=%d result_add=%d "
                    //        "with_istride=%d time=%lldus Tops/s=%.2f\n",
                    //        loop, fidtype_c[i], fodtype_c[i], has_bias, result_add,
                    //        with_str, et - st, ops * loop / ((float)(et - st) * 1e6));
                }
            }
        }
    }
    #endif
    data_type_t f8_idtype[2] = {DT_FP8E5M2, DT_FP8E4M3};
    data_type_t f8_odtype[4] = {DT_FP8E5M2, DT_FP8E4M3, DT_FP16, DT_FP32};
    char *f8_idtype_c[2] = {"DT_FP8E5M2", "DT_FP8E4M3"};
    char *f8_odtype_c[4] = {"DT_FP8E5M2", "DT_FP8E4M3", "DT_FP16", "FP32"};
    printf("conv float shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           fshape.n, fshape.c, fshape.h, fshape.w, oc, kernel.h, kernel.w);
    printf("conv conflict shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           tiny_shape.n, tiny_shape.c, tiny_shape.h, tiny_shape.w, oc, kernel.h, kernel.w);
    for (int do_rescale = 1; do_rescale < 2; do_rescale++)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int w = 0; w < 2; w++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int has_bias = 0; has_bias < 2; has_bias++)
                    {
                        for (int result_add = 0; result_add < 2; result_add++)
                        {
                            for (int with_str = 0; with_str < 2; with_str++)
                            {
                                tpu_compact_stride(&istride, 0, &fshape);
                                st = firmware_timer_get_time_us();
                                for (int l = 0; l < loop; l++)
                                {
                                    tpu_bdc_fp_conv2d_with_rescale(
                                        0,              // o
                                        5 * bank_size,  // i
                                        11 * bank_size, // w
                                        14 * bank_size, // b
                                        15 * bank_size, // rescale
                                        &fshape,
                                        with_str ? &istride : NULL,
                                        oc,
                                        &kernel,
                                        &pad,
                                        &stride,
                                        &dilation,
                                        f8_odtype[j],
                                        f8_idtype[i],
                                        f8_idtype[w],
                                        DT_FP32,
                                        has_bias,
                                        result_add,
                                        false,
                                        do_rescale,
                                        false);
                                }
                                tpu_poll();
                                et = firmware_timer_get_time_us();
                                ops = conv_ops(fshape.n, fshape.c, oc, fshape.h, fshape.w,
                                               kernel.h, kernel.w, has_bias, result_add);
                                printf("loop:%d, idtype=%s wdtype=%s odtype=%s has_bias=%d result_add=%d do_rescale=%d "
                                       "with_istride=%d time=%lldus Tops/s=%.2f\n",
                                       loop, f8_idtype_c[i], f8_idtype_c[w],f8_odtype_c[j], has_bias, result_add, do_rescale,
                                       with_str, et - st, ops * (float)loop / ((float)(et - st) * 1e6));
                                ///////////// all opd in one bank ////////////////////////
                                // tpu_compact_stride(&istride, 0, &tiny_shape);
                                // st = firmware_timer_get_time_us();
                                // for (int l = 0; l < loop; l++)
                                // {
                                //     tpu_bdc_fp_conv2d_with_rescale(
                                //         0,                                                                                      // o
                                //         tiny_shape.h * tiny_shape.w * sizeof(float),                                            // i
                                //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2,                                        // w
                                //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float),  // b
                                //         (tiny_shape.h * tiny_shape.w * 2 + kernel.h * kernel.w + tiny_shape.c) * sizeof(float), // rescale
                                //         &tiny_shape,
                                //         with_str ? &istride : NULL,
                                //         oc,
                                //         &kernel,
                                //         &pad,
                                //         &stride,
                                //         &dilation,
                                //         f8_odtype[j],
                                //         f8_idtype[i],
                                //         f8_idtype[w],
                                //         DT_FP32,
                                //         has_bias,
                                //         result_add,
                                //         false,
                                //         do_rescale,
                                //         false);
                                // }
                                // tpu_poll();
                                // et = firmware_timer_get_time_us();
                                // ops = conv_ops(tiny_shape.n, tiny_shape.c, oc, tiny_shape.h, tiny_shape.w,
                                //                kernel.h, kernel.w, has_bias, result_add);
                                // printf("conflict loop=%d: idtype=%s wdtype=%s odtype=%s has_bias=%d result_add=%d do_rescale=%d "
                                //        "with_istride=%d time=%lldus Tops/s=%.2f\n",
                                //        loop, f8_idtype_c[i], f8_idtype_c[w], f8_odtype_c[j], has_bias, result_add, do_rescale,
                                //        with_str, et - st, ops * loop / ((float)(et - st) * 1e6));
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    /////////////////// INT8 requant C ////////////////////
    printf("conv sym requant C shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           qshape.n, qshape.c, qshape.h, qshape.w, oc, kernel.h, kernel.w);
    {
        data_type_t sqdtype[2] = {DT_INT8, DT_INT16};
        data_type_t squdtype[2] = {DT_UINT8, DT_UINT16};
        char *sqdtype_c[2] = {"INT8", "INT16"};
        for (int i = 0; i < 2; i++)
        {
            for (int has_bias = 0; has_bias < 2; has_bias++)
            {
                for (int if_relu = 0; if_relu < 2; if_relu++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        for (int with_str = 0; with_str < 2; with_str++)
                        {
                            tpu_compact_stride(&istride, 0, &qshape);
                            st = firmware_timer_get_time_us();
                            for (int l = 0; l < loop; l++)
                            {
                                tpu_bdc_conv2d_requant_C(
                                    0,
                                    6 * bank_size,
                                    12 * bank_size,
                                    15 * bank_size,
                                    &qshape,
                                    with_str ? &istride : NULL,
                                    oc,
                                    &kernel,
                                    &pad,
                                    &stride,
                                    &dilation,
                                    if_relu ? squdtype[i] : sqdtype[i],
                                    DT_INT8,
                                    DT_INT8,
                                    DT_INT32,
                                    100000, // multipler
                                    22,     // rshift
                                    0,      // yzp
                                    has_bias,
                                    if_relu,
                                    true, // do_rq
                                    false);
                            }
                            tpu_poll();
                            et = firmware_timer_get_time_us();
                            ops = conv_ops(qshape.n, qshape.c, oc, qshape.h, qshape.w,
                                           kernel.h, kernel.w, has_bias, if_relu);
                            printf("lopp=%d, idtype=%s odtype=%s has_bias=%d if_relu=%d "
                                   "with_istride=%d time=%lldus Tops/s=%.2f\n",
                                   loop, sqdtype_c[j], sqdtype_c[i], has_bias, if_relu,
                                   with_str, et - st, (float)loop * ops / ((float)(et - st) * 1e6));
                            ///////////// all opd in one bank ////////////////////////
                            // tpu_compact_stride(&istride, 0, &tiny_shape);
                            // st = firmware_timer_get_time_us();
                            // for (int l = 0; l < loop; l++)
                            // {
                            //     tpu_bdc_conv2d_requant_C(
                            //         0,
                            //         tiny_shape.h * tiny_shape.w * sizeof(float),                                           // i
                            //         ALIGN(tiny_shape.h * tiny_shape.w * sizeof(float) * 2, 16),                            // w
                            //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float), // b
                            //         &tiny_shape,
                            //         with_str ? &istride : NULL,
                            //         oc,
                            //         &kernel,
                            //         &pad,
                            //         &stride,
                            //         &dilation,
                            //         if_relu ? squdtype[i] : sqdtype[i],
                            //         DT_INT8,
                            //         DT_INT8,
                            //         DT_INT32,
                            //         100000,
                            //         22,
                            //         0,
                            //         has_bias,
                            //         if_relu,
                            //         true,
                            //         false);
                            // }
                            // tpu_poll();
                            // et = firmware_timer_get_time_us();
                            // ops = conv_ops(tiny_shape.n, tiny_shape.c, oc, tiny_shape.h, tiny_shape.w,
                            //                kernel.h, kernel.w, has_bias, if_relu);
                            // printf("conflict loop=%d: idtype=%s odtype=%s has_bias=%d if_relu=%d "
                            //        "with_istride=%d time=%lldus Tops/s=%.2f\n",
                            //        loop, sqdtype_c[j], sqdtype_c[i], has_bias, if_relu,
                            //        with_str, et - st, ops * loop / ((float)(et - st) * 1e6));
                        }
                    }
                }
            }
        }
    }

    /////////////////// INT8 requant ////////////////////
    printf("conv asym pc ,requant pc shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           qshape.n, qshape.c, qshape.h, qshape.w, oc, kernel.h, kernel.w);
    {
        data_type_t sqdtype[2] = {DT_INT8, DT_INT16};
        data_type_t squdtype[2] = {DT_UINT8, DT_UINT16};
        char *sqdtype_c[2] = {"INT8", "INT16"};
        for (int i = 0; i < 2; i++)
        {
            for (int has_bias = 0; has_bias < 2; has_bias++)
            {
                for (int if_relu = 0; if_relu < 2; if_relu++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        for (int with_str = 0; with_str < 2; with_str++)
                        {
                            tpu_compact_stride(&istride, 0, &qshape);
                            st = firmware_timer_get_time_us();
                            for (int l = 0; l < loop; l++)
                            {
                                tpu_bdc_conv2d_requant_pc_asym_pc(
                                    0,
                                    6 * bank_size,
                                    12 * bank_size,
                                    15 * bank_size,
                                    15 * bank_size + 0x400,
                                    15 * bank_size + 0x800,
                                    15 * bank_size + 0xc00,
                                    &qshape,
                                    with_str ? &istride : NULL,
                                    oc,
                                    &kernel,
                                    &pad,
                                    &stride,
                                    &dilation,
                                    if_relu ? squdtype[i] : sqdtype[i],
                                    DT_INT8,
                                    DT_INT8,
                                    DT_INT32,
                                    1,       //  do_requant
                                    1,       //  has_bias
                                    if_relu, //  do_relu
                                    false,   // result
                                    false,   // sym_range
                                    RM_HALF_UP);
                            }
                            tpu_poll();
                            et = firmware_timer_get_time_us();
                            ops = conv_ops(qshape.n, qshape.c, oc, qshape.h, qshape.w,
                                           kernel.h, kernel.w, has_bias, if_relu);
                            printf("loop=%d, idtype=%s odtype=%s has_bias=%d if_relu=%d do_requant=1 "
                                   "with_istride=%d time=%lldus Tops/s=%.2f\n",
                                   loop, sqdtype_c[j], sqdtype_c[i], has_bias, if_relu,
                                   with_str, et - st, (float)loop * ops / ((float)(et - st) * 1e6));
                            ///////////// all opd in one bank ////////////////////////
                            // tpu_compact_stride(&istride, 0, &tiny_shape);
                            // st = firmware_timer_get_time_us();
                            // for (int l = 0; l < loop; l++)
                            // {
                            //     tpu_bdc_conv2d_requant_pc_asym_pc(
                            //         0,
                            //         tiny_shape.h * tiny_shape.w * sizeof(float),                                           // i
                            //         ALIGN(tiny_shape.h * tiny_shape.w * sizeof(float) * 2, 16),                            // w
                            //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float), // b
                            //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float) +
                            //             DIV_UP(tiny_shape.c, NPU_NUM) * sizeof(float), // kzp
                            //         tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float) +
                            //             DIV_UP(tiny_shape.c, NPU_NUM) * sizeof(float) * 2, // pad
                            //         ALIGN(tiny_shape.h * tiny_shape.w * sizeof(float) * 2 + kernel.h * kernel.w * sizeof(float) +
                            //                   DIV_UP(tiny_shape.c, NPU_NUM) * sizeof(float) * 4,
                            //               sizeof(int) * 2), // req
                            //         &tiny_shape,
                            //         with_str ? &istride : NULL,
                            //         oc,
                            //         &kernel,
                            //         &pad,
                            //         &stride,
                            //         &dilation,
                            //         if_relu ? squdtype[i] : sqdtype[i],
                            //         DT_INT8,
                            //         DT_INT8,
                            //         DT_INT32,
                            //         1,
                            //         1,
                            //         if_relu,
                            //         false,
                            //         false,
                            //         RM_HALF_UP);
                            // }
                            // tpu_poll();
                            // et = firmware_timer_get_time_us();
                            // ops = conv_ops(tiny_shape.n, tiny_shape.c, oc, tiny_shape.h, tiny_shape.w,
                            //                kernel.h, kernel.w, has_bias, if_relu);
                            // printf("conflict loop=%d: idtype=%s odtype=%s has_bias=%d if_relu=%d do_requant=1 "
                            //        "with_istride=%d time=%lldus Tops/s=%.2f\n",
                            //        loop, sqdtype_c[j], sqdtype_c[i], has_bias, if_relu,
                            //        with_str, et - st, ops * loop / ((float)(et - st) * 1e6));
                        }
                    }
                }
            }
        }
    }

    /////////////////// INT8 convolution ////////////////////
    printf("conv asym without requant shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
           qshape.n, qshape.c, qshape.h, qshape.w, oc, kernel.h, kernel.w);
    {
        data_type_t sqdtype[] = {DT_INT8};
        data_type_t squdtype[] = {DT_INT16, DT_INT32};
        char *sqdtype_c[] = {"INT8"};
        char *squdtype_c[] = {"INT16", "INT32"};
        for (int i = 0; i < 2; i++)
        {
            for (int has_bias = 0; has_bias < 2; has_bias++)
            {
                for (int if_relu = 0; if_relu < 2; if_relu++)
                {
                    for (int itypex = 0; itypex < 1; itypex++)
                    {
                        for (int ktypex = 0; ktypex < 1; ktypex++)
                        {
                            for (int with_str = 0; with_str < 2; with_str++)
                            {
                                tpu_compact_stride(&istride, 0, &qshape);
                                st = firmware_timer_get_time_us();
                                for (int l = 0; l < loop; l++)
                                {
                                    tpu_bdc_conv2d_requant_pc_asym_pc(
                                        0,
                                        6 * bank_size,
                                        12 * bank_size,
                                        15 * bank_size,
                                        15 * bank_size + 0x400,
                                        15 * bank_size + 0x800,
                                        15 * bank_size + 0xc00,
                                        &qshape,
                                        with_str ? &istride : NULL,
                                        oc,
                                        &kernel,
                                        &pad,
                                        &stride,
                                        &dilation,
                                        squdtype[i],
                                        sqdtype[itypex],
                                        sqdtype[ktypex],
                                        DT_INT32,
                                        0, //  do_requant
                                        has_bias,
                                        if_relu, //  do_relu
                                        false,   // result
                                        false,   // sym_range
                                        RM_HALF_UP);
                                }
                                tpu_poll();
                                et = firmware_timer_get_time_us();
                                ops = conv_ops(qshape.n, qshape.c, oc, qshape.h, qshape.w,
                                               kernel.h, kernel.w, has_bias, if_relu);
                                printf("loop=%d, idtype=%s kdtype=%s odtype=%s has_bias=%d if_relu=%d do_requant=%d "
                                       "with_istride=%d time=%lldus Tops/s=%.2f\n",
                                       loop, sqdtype_c[itypex], sqdtype_c[ktypex], squdtype_c[i], has_bias, if_relu,
                                       0, with_str, et - st, (float)loop * ops / ((float)(et - st) * 1e6));
                            }
                        }
                    }
                }
            }
        }
    }

    /////////////////// INT8 requant ////////////////////
    int oc_vec[] = {128, 256};
    for (int ocx = 0; ocx < 2; ocx++)
    {
        printf("conv asym requant pc shape=(%d %d %d %d) oc=%d kh=%d kw=%d\n",
               qshape.n, qshape.c, qshape.h, qshape.w, oc_vec[ocx], kernel.h, kernel.w);
        data_type_t sqdtype[1] = {DT_INT8};
        char *sqdtype_c[1] = {"INT8"};
        for (int itypex = 0; itypex < 1; itypex++)
        {
            for (int ktypex = 0; ktypex < 1; ktypex++)
            {
                tpu_compact_stride(&istride, 0, &qshape);
                st = firmware_timer_get_time_us();
                for (int l = 0; l < loop; l++)
                {
                    tpu_bdc_conv2d_requant_pc_asym_pc(
                        0,
                        6 * bank_size,
                        12 * bank_size,
                        15 * bank_size,
                        15 * bank_size + 0x400,
                        15 * bank_size + 0x800,
                        15 * bank_size + 0xc00,
                        &qshape,
                        NULL,
                        oc_vec[ocx],
                        &kernel,
                        &pad,
                        &stride,
                        &dilation,
                        sqdtype[MAX(itypex, ktypex)],
                        sqdtype[itypex],
                        sqdtype[ktypex],
                        DT_INT32,
                        1,     //  do_requant
                        0,     //  has_bias
                        0,     //  do_relu
                        false, // result
                        false, // sym_range
                        RM_HALF_UP);
                }
                tpu_poll();
                et = firmware_timer_get_time_us();
                ops = conv_ops(qshape.n, qshape.c, oc_vec[ocx], qshape.h, qshape.w,
                               kernel.h, kernel.w, 0, 0);
                printf("loop=%d, idtype=%s kdtype=%s odtype=%s has_bias=%d if_relu=%d do_requant=%d "
                       "time=%lldus Tops/s=%.2f\n",
                       loop, sqdtype_c[itypex], sqdtype_c[ktypex], sqdtype_c[MAX(itypex, ktypex)], 0, 0,
                       0, et - st, (float)loop * ops / ((float)(et - st) * 1e6));
            }
        }
    }
}
