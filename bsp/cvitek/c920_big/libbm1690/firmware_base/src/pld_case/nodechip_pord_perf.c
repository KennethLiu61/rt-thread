#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "atomic_pooling_depthwise_gen_cmd.h"
#include "tpu_kernel.h"

#define FP8TYPE(dtype) ((dtype) >> 5)
#define PRECISION(dtype) (((dtype) >> 1) & 0xf)
static const int input_dtypes[] = {
    DT_FP16,
    DT_BFP16,
    DT_FP32,
    DT_INT8,
};
static const char *const input_dtypes_str[] = {
    "DT_FP16",
    "DT_BFP16",
    "DT_FP32",
    "DT_INT8",
};
static const int output_dtypes[] = {
    DT_INT8,
    DT_INT16,
    DT_INT32,
};
static const char *const output_dtypes_str[] = {
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
};

static const int input_float_dtypes[13] = {
    DT_FP16,
    DT_FP16,
    DT_BFP16,
    DT_BFP16,
    DT_FP32,
    DT_FP8E4M3,
    DT_FP8E5M2,
    DT_FP8E5M2,
    DT_FP8E5M2,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP8E4M3,
    DT_FP8E4M3};

static const char *input_float_dtypes_str[] = {
    "DT_FP16",
    "DT_FP16",
    "DT_BFP16",
    "DT_BFP16",
    "DT_FP32",
    "DT_FP8E4M3",
    "DT_FP8E5M2",
    "DT_FP8E5M2",
    "DT_FP8E5M2",
    "DT_FP8E5M2",
    "DT_FP8E4M3",
    "DT_FP8E4M3",
    "DT_FP8E4M3"};

static const int output_float_dtypes[13] = {
    DT_FP16,
    DT_FP32,
    DT_BFP16,
    DT_FP32,
    DT_FP32,
    DT_FP32,
    DT_FP32,
    DT_FP16,
    DT_FP8E5M2,
    DT_FP8E4M3,
    DT_FP16,
    DT_FP8E5M2,
    DT_FP8E4M3};

static const char *const output_float_dtypes_str[] = {
    "DT_FP16",
    "DT_FP32",
    "DT_BFP16",
    "DT_FP32",
    "DT_FP32",
    "DT_FP32",
    "DT_FP32",
    "DT_FP16",
    "DT_FP8E5M2",
    "DT_FP8E4M3",
    "DT_FP16",
    "DT_FP8E5M2",
    "DT_FP8E4M3"};
u64 pord_ops(
    int n,
    int ic,
    int oc,
    int oh,
    int ow,
    int kh,
    int kw,
    bool has_bias,
    bool max_pooling)
{
  u64 total = (u64)oc * oh * ow * n *
              (kh * kw * ic * (max_pooling ? 1 : 2) +
               (has_bias ? 1 : 0));
  return total;
}

void nodechip_pord_perf_test(
    unsigned long long input_global_addr,
    unsigned long long output_global_addr)
{
  tpu_initialize();

  const int loop = 10;
  dim4 ishape = {2, 64, 64, 64};
  dim2 kernel = {3, 3};
  padding_t padding = {1, 1, 1, 1};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  int roi_num = 100;
  printf("Test PorD:\n");
  printf("loop time = %d per case\n", loop);
  printf("input_shape=(%d, %d, %d, %d), kernel=(%d, %d), "
         "padding=(%d, %d, %d, %d), stride=(%d, %d), dilation=(%d, %d), roi_num=%d\n",
         ishape.n, ishape.c, ishape.h, ishape.w, kernel.h, kernel.w,
         padding.top, padding.bottom, padding.left, padding.right,
         stride.h, stride.w, dilation.h, dilation.w, roi_num);

  scalar_t pad_val = {.u32 = 0};
  scalar_t scale_int = {.u8 = 30};
  scalar_t scale_f32 = {.f32 = 1 / (kernel.h * kernel.w)};
  scalar_t scale_fp[] = {
      tpu_cast(scale_f32, DT_FP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_BFP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      scale_f32,
  };
  scalar_t kernel_const[] = {
      // for roi avg
      tpu_cast(scale_f32, DT_FP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_BFP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_BFP16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      scale_f32,
      tpu_cast(scale_f32, DT_FP8E4M3, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E5M2, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E5M2, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E5M2, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E5M2, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E4M3, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E4M3, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      tpu_cast(scale_f32, DT_FP8E4M3, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      {.u8 = 30}
  };
  const int rshift = 3;

  dim4 wshape = {1, ishape.c, kernel.h, kernel.w};
  dim4 bshape = {1, ishape.c, 1, 1};
  dim4 rshape = {1, roi_num, 1, 4};
  dim4 istride, wstride, bstride, rstride;
  tpu_aligned_stride(&istride, 0, &ishape, DT_FP32);
  tpu_compact_stride(&wstride, 0, &wshape);
  tpu_compact_stride(&bstride, 0, &bshape);
  tpu_compact_stride(&rstride, 0, &rshape);

  const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  local_addr_t input_addr = 0;
  local_addr_t weight_addr = ALIGN(input_addr + istride.n * sizeof(float), bank_size);
  local_addr_t bias_addr = ALIGN(weight_addr + wstride.n * sizeof(float), bank_size);
  local_addr_t rq_addr = ALIGN(weight_addr + wstride.n * sizeof(float), bank_size);
  local_addr_t roi_addr = ALIGN(rq_addr + rstride.n * sizeof(float), bank_size);

  local_addr_t output_addr = ALIGN(roi_addr + rstride.n * sizeof(float), bank_size);
  TPUKERNEL_ASSERT(output_addr + istride.n * sizeof(float) <= LOCAL_MEM_SIZE);
  u64 st = 0, et = 0;
  u64 ops;

  // --------------------- Test max pooling ---------------------------
  printf("\nTest Max Pooling:\n");
  st = firmware_timer_get_time_us();
  for (int i = 0; i < loop; ++i)
  {
    tpu_bdc_int8_max_pool2d(
        output_addr,
        input_addr,
        &ishape,
        &kernel,
        &padding,
        &stride,
        &dilation,
        DT_INT8,
        pad_val);
  }
  tpu_poll();
  et = firmware_timer_get_time_us();
  ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h, kernel.w,
                 (bool)false, true);
  printf("Test %s(dtype = %s) time %lld us Tops/s=%.2f\n",
         "tpu_bdc_int8_max_pool2d", "DT_INT8", et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));

  for (unsigned int d = 0; d < 3; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_max_pool2d(
          output_addr,
          input_addr,
          &ishape,
          &kernel,
          &padding,
          &stride,
          &dilation,
          input_dtypes[d],
          pad_val);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, true);
    printf("Test %s(dtype = %s) time %lld us Tops/s=%.2f\n",
           "tpu_bdc_fp_max_pool2d", input_dtypes_str[d], et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  // --------------------- Test avg pooling ---------------------------
  printf("\nTest Avg Pooling:\n");
  for (unsigned int d = 0; d < 3; ++d)
  { // INT8/INT16/INT32
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_int8_avg_pool2d(
          output_addr,
          input_addr,
          &ishape,
          &kernel,
          &padding,
          &stride,
          &dilation,
          output_dtypes[d],
          DT_INT8,
          scale_int.u8,
          rshift);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(output_dtype = %s), do_rq=1, time %lld us Tops/s=%.2f\n",
           "tpu_bdc_int8_avg_pool2d", output_dtypes_str[d], et - st, ops * (u64)loop / ((float)(et - st) * 1e6));

    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_int8_avg_pool2d(
          output_addr,
          input_addr,
          &ishape,
          &kernel,
          &padding,
          &stride,
          &dilation,
          output_dtypes[d],
          DT_INT8,
          scale_int.u8,
          0);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(output_dtype = %s), do_rq=0, time %lld us Tops/s=%.2f\n",
           "tpu_bdc_int8_avg_pool2d", output_dtypes_str[d], et - st, ops * (u64)loop / ((float)(et - st) * 1e6));
  }

  for (unsigned int d = 0; d < 3; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      tpu_bdc_fp_avg_pool2d(
          output_addr,
          input_addr,
          &ishape,
          &kernel,
          &padding,
          &stride,
          &dilation,
          input_dtypes[d],
          scale_fp[d]);
    }
    tpu_poll();
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(dtype = %s) time %lld us Tops/s=%.2f\n",
           "tpu_bdc_fp_avg_pool2d", input_dtypes_str[d], et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  // --------------------- Test depthwise ---------------------------
  printf("\nTest depthwise:\n");
  for (unsigned int d = 0; d < 3; ++d)
  { // INT8/INT16/INT32
    for (int has_bias = 0; has_bias < 2; ++has_bias)
    {
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; ++i)
      {
        scalar_t pad_val = {.u32 = 0};
        tpu_bdc_int8_depthwise2d(
            output_addr,
            input_addr,
            weight_addr,
            bias_addr,
            pad_val,
            &ishape,
            &kernel,
            &padding,
            &stride,
            &dilation,
            output_dtypes[d],
            DT_INT8,
            DT_INT8,
            DT_INT32,
            rshift,
            has_bias,
            false,
            RM_UP);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                     kernel.w, (bool)false, false);
      printf("Test %s(output_dtype = %s, has_bias = %d, relu=false) time %lld us Tops/s=%.2f\n",
             "tpu_bdc_int8_depthwise2d", output_dtypes_str[d], has_bias, et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
    }
    for (int has_bias = 0; has_bias < 2; ++has_bias)
    {
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; ++i)
      {
        scalar_t pad_val = {.u32 = 0};
        tpu_bdc_int8_depthwise2d(
            output_addr,
            input_addr,
            weight_addr,
            bias_addr,
            pad_val,
            &ishape,
            &kernel,
            &padding,
            &stride,
            &dilation,
            DT_UINT8,
            DT_INT8,
            DT_INT8,
            DT_INT32,
            rshift,
            has_bias,
            true,
            RM_UP);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                     kernel.w, (bool)false, false);
      printf("Test %s(output_dtype = DT_UINT8, has_bias = %d, relu=true) time %lld us Tops/s=%.2f\n",
             "tpu_bdc_int8_depthwise2d", has_bias, et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
    }
  }
  CMD_ID_NODE id_node;
  tpu_get_id_node(&id_node);
  int do_rq = 0;
  for (unsigned int d = 0; d < 13; ++d)
  {
    if (d > 6)
      do_rq = 1;
    for (int has_bias = 0; has_bias < 2; ++has_bias)
    {
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; ++i)
      {
        atomic_depthwise_gen_cmd(
            input_addr,
            weight_addr,
            has_bias ? bias_addr : 0,
            0,
            rq_addr,
            output_addr,
            ishape.n,
            ishape.c,
            ishape.h,
            ishape.w,
            kernel.h,
            kernel.w,
            stride.h,
            stride.w,
            0,
            0,
            dilation.h,
            dilation.w,
            padding.top,
            padding.bottom,
            padding.left,
            padding.right,
            false,
            !has_bias,
            true,
            NO_USE,
            false,
            false,
            do_rq,
            false,
            PRECISION(input_float_dtypes[d]),
            PRECISION(output_float_dtypes[d]),
            FP8TYPE(input_float_dtypes[d]), // for fp8
            FP8TYPE(input_float_dtypes[d]),
            FP8TYPE(output_float_dtypes[d]),
            PAD_CONSTANT,
            MASTER_THREAD,
            &id_node);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                     kernel.w, (bool)false, false);
      printf("Test %s(input_dtype = %s, output_dtype = %s, has_bias = %d, do_rq = %d) time %lld us Tops/s=%.2f\n",
             "tpu_bdc_fp_depthwise2d", input_float_dtypes_str[d], output_float_dtypes_str[d], has_bias, do_rq, et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
    }
  }
  // --------------------- Prepare for roi test ---------------------------
  tpu_gdma_cpy_S2L(
      roi_addr,
      input_global_addr,
      &rshape,
      &rstride,
      NULL,
      DT_INT16);
  tpu_poll();

  // --------------------- Test roi max pooling ---------------------------
  printf("\nTest ROI Max Pooling:\n");
  for (unsigned int d = 0; d < 4; ++d)
  {
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      atomic_roi_max_min_pooling_gen_cmd(
          input_addr,
          roi_addr,
          output_addr,
          ishape.n,
          ishape.c,
          ishape.h,
          ishape.w,
          roi_num,
          kernel.h,
          kernel.w,
          0, // imm_const_val
          1, // input_sign
          PRECISION(input_dtypes[d]),
          0,
          PD_ROI_MAX_POOLING,
          MASTER_THREAD,
          &id_node);
    }
    poll_all_engine_done(&id_node);
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, true);
    printf("Test %s(dtype = %s) time %lld us Tops/s=%.2f\n",
           "roi_max_pooling", input_dtypes_str[d], et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  // --------------------- Test roi avg pooling ---------------------------

  printf("\nTest ROI Avg Pooling:\n");
  do_rq = 0;
  for (unsigned int d = 0; d < 13; ++d)
  { // fp16, bfp16, fp32 , fp8e5m2, fp8e4m3
    if (d > 6)
      do_rq = 1;
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      atomic_roi_avg_pooling_gen_cmd(
          input_addr,
          roi_addr,
          output_addr,
          rq_addr,
          ishape.n,
          ishape.c,
          ishape.h,
          ishape.w,
          roi_num,
          kernel.h,
          kernel.w,
          kernel_const[d % 4].s32, // kernel_const_val
          0,                   // imm_const_val
          do_rq,
          0,   // rq_is_const
          1.f, // re_scale
          0,   // sym_range
          0,   // do_relu
          FP8TYPE(input_float_dtypes[d]),
          FP8TYPE(input_float_dtypes[d]),
          FP8TYPE(output_float_dtypes[d]),
          PRECISION(input_float_dtypes[d]),
          PRECISION(output_float_dtypes[d]),
          0, // round_mode
          MASTER_THREAD,
          &id_node);
    }
    poll_all_engine_done(&id_node);
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s (in dtype = %s, out dtype = %s, do_rq=%d), time %lld us Tops/s=%.2f\n",
           "roi_avg_pooling_fp", input_float_dtypes_str[d], output_float_dtypes_str[d], do_rq, et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  for (unsigned int d = 1; d < 3; ++d)
  { // INT16, INT32
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      atomic_roi_avg_pooling_quant_gen_cmd(
          input_addr,
          roi_addr,
          output_addr,
          0, // rq_addr
          ishape.n,
          ishape.c,
          ishape.h,
          ishape.w,
          roi_num,
          kernel.h,
          kernel.w,
          kernel_const[13].s32,        // kernel_const_val
          0,                          // imm_const_val
          1,                          // input_sign
          1,                          // output_sign
          1,                          // kernel_sign
          PRECISION(input_dtypes[5]), // INT8
          PRECISION(output_dtypes[d]),
          1,  // do_relu
          1,  // do_rq
          1,  // rq_is_const
          10, // mul
          2,  // shift
          1,  // yzp
          1,  // sym_range
          ROUND_HALF_UP,
          MASTER_THREAD,
          &id_node);
    }
    poll_all_engine_done(&id_node);
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(out_dtype = %s) time %lld us Tops/s=%.2f\n",
           "roi_avg_pooling_int8", output_dtypes_str[d], et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  // --------------------- Test roi depthwise ---------------------------
  printf("\nTest ROI Depthwise Pooling:\n");
  do_rq = 0;
  for (unsigned int d = 0; d < 13; ++d)
  { // fp16, bfp16, fp32 , fp8e5m2, fp8e4m3
    if (d > 6)
      do_rq = 1;
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      atomic_roi_depthwise_gen_cmd(
          input_addr,
          weight_addr,
          roi_addr,
          rq_addr,
          output_addr,
          ishape.n,
          ishape.c,
          ishape.h,
          ishape.w,
          roi_num,
          kernel.h,
          kernel.w,
          0,     // imm_const_val
          0,     // kernel_is_const
          0,     // kernel_rotate
          false, // relu
          do_rq, // requant
          0,     // rq_is_const
          PRECISION(input_float_dtypes[d]),
          PRECISION(output_float_dtypes[d]),
          FP8TYPE(input_float_dtypes[d]),
          FP8TYPE(input_float_dtypes[d]),
          FP8TYPE(output_float_dtypes[d]),
          MASTER_THREAD,
          &id_node);
    }
    poll_all_engine_done(&id_node);
    et = firmware_timer_get_time_us();

    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(in dtype = %s, out dtype = %s, do_rq=%d), time %lld us Tops/s=%.2f\n",
           "roi_depthwise_fp", input_float_dtypes_str[d], output_float_dtypes_str[d], do_rq, et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }
  for (unsigned int d = 1; d < 3; ++d)
  { // INT16, INT32
    st = firmware_timer_get_time_us();
    for (int i = 0; i < loop; ++i)
    {
      atomic_roi_depthwise_quant_gen_cmd(
          input_addr,
          weight_addr,
          roi_addr,
          1,
          output_addr,
          ishape.n,
          ishape.c,
          ishape.h,
          ishape.w,
          roi_num,
          kernel.h,
          kernel.w,
          0,  // imm_const_val
          0,  // kernel_is_const
          0,  // kernel_rotate
          1,  // input_sign
          1,  // kernel_sign
          1,  // res_sign
          0,  // do_relu
          0,  // sym_saturate
          1,  // do_requant
          1,  // requant_is_const
          1,  // shift
          10, // ozp
          ROUND_HALF_UP,
          PRECISION(input_dtypes[3]), // INT8
          PRECISION(output_dtypes[d]),
          MASTER_THREAD, // thread_id
          &id_node);
    }
    poll_all_engine_done(&id_node);
    et = firmware_timer_get_time_us();
    ops = pord_ops(ishape.n, ishape.c, 1, ishape.h, ishape.w, kernel.h,
                   kernel.w, (bool)false, false);
    printf("Test %s(out_dtype = %s) time %lld us Tops/s=%.2f\n",
           "atomic_roi_depthwise_quant_gen_cmd", output_dtypes_str[d], et - st, (ops * (u64)loop / ((float)(et - st) * 1e6)));
  }

  tpu_set_id_node(&id_node);
}