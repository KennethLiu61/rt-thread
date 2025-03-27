#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_kernel.h"
#include "conv_util.h"

static dim4 check_mem_valid(dim4 shape, data_type_t dtype, int mem_limit) {
  dim4 stride;
  tpu_aligned_stride(&stride, 0, &shape, dtype);
  int tensor_size = tpu_data_type_size(dtype) * stride.n * shape.n;
  ASSERT(tensor_size <= mem_limit);
  return stride;
}

static void nodechip_conv_bw_fp8_perf_test() {
  tpu_initialize();
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  padding_t pad = {0, 0, 0, 0};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim2 insert = {0, 0};

  u64 st, et;
  u64 ops;
  dim4 istride;
  int loop = 5;

  data_type_t idtype[2] = {DT_FP8E5M2, DT_FP8E4M3};
  data_type_t opd1type[2] = {DT_FP8E5M2, DT_FP8E4M3};
  ///////////////////////  max perf case  ///////////////////////////
  dim2 kernel = {3, 3};
  int n = NPU_NUM;
  int ic = NPU_NUM;
  int oc = NPU_NUM;
  int ih = 16;
  int iw = 16;

  int with_stride_origin_ih = 24;
  int with_stride_origin_iw = 24;

  //forward infernece h and w, so exchange bw's dilation and stride
  int oh = cal_conv2d_out_size(ih, kernel.h, stride.h, dilation.h, pad.bottom, pad.top, insert.h);
  int ow = cal_conv2d_out_size(iw, kernel.w, stride.w, dilation.w, pad.left, pad.right, insert.w);
  dim4 opd0_shape = {n, ic, ih, iw};
  dim4 opd1_shape = {n, oc, oh, ow};

  char *fidtype_c[2] = {"FP8E5M2", "FP8E4M3"};
  for (int opd0_type_idx = 0; opd0_type_idx < 2; opd0_type_idx++) {
    for (int opd1_type_idx = 0; opd1_type_idx < 2; opd1_type_idx++) {
      printf("conv bw opd0_prec:%s opd1_prec:%s opd0_shape=(%d %d %d %d) opd1_shape=(%d %d %d %d) kh=%d kw=%d\n",
          fidtype_c[opd0_type_idx], fidtype_c[opd1_type_idx],
          opd0_shape.n, opd0_shape.c, opd0_shape.h, opd0_shape.w,
          opd1_shape.n, opd1_shape.c, opd1_shape.h, opd1_shape.w, kernel.h, kernel.w);
      for (int with_str = 0; with_str < 2; with_str++) {

        dim4 with_str_shape = {n, ic, with_stride_origin_ih, with_stride_origin_iw};
        istride = check_mem_valid(with_str ? with_str_shape : opd0_shape, idtype[opd0_type_idx], 6 * bank_size);

        dim4 opd1_stride = check_mem_valid(opd1_shape, opd1type[opd1_type_idx], 6 * bank_size);

        dim4 res_shape = {1, oc, DIV_UP(ic, 64) * kernel.h * kernel.w, 64};
        dim4 res_stirde = check_mem_valid(res_shape, opd1type[opd1_type_idx], 4 * bank_size);

        // no bank conflict
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; i++) {
          tpu_bdc_fp_conv2d_backward(12 * bank_size, 0, 6 * bank_size, NO_USE, &opd0_shape, &opd1_shape, &kernel, &insert, &pad,
                                    &stride, &dilation, with_str ? &istride : NULL,
                                    true, NO_USE, PAD_CONSTANT, idtype[opd0_type_idx], opd1type[opd1_type_idx]);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        ops = conv_bw_ops(opd0_shape.n, opd0_shape.c, opd1_shape.c, opd1_shape.h, opd1_shape.w, kernel.h, kernel.w);
        printf("loop time =%d: with_tensor_str=%d avg time=%lldus Tops/s=%.4f\n",
                loop, with_str, (et - st) / loop, ops * loop / ((float)(et - st) * 1e6));

        // bank conflict
        local_addr_t opd0_addr = 0;
        local_addr_t opd1_addr = opd0_addr + istride.n * (with_str ? with_str_shape.n : opd0_shape.n) * tpu_data_type_size(idtype[opd0_type_idx]);
        local_addr_t res_addr = opd1_addr + opd1_stride.n * opd1_shape.n * tpu_data_type_size(opd1type[opd1_type_idx]);
        ASSERT((int)res_addr + res_stirde.n * res_shape.n * tpu_data_type_size(DT_FP32) < LOCAL_MEM_SIZE);
        st = firmware_timer_get_time_us();
        for (int i = 0; i < loop; i++) {
          tpu_bdc_fp_conv2d_backward(res_addr, opd0_addr, opd1_addr, NO_USE, &opd0_shape, &opd1_shape, &kernel, &insert, &pad,
                                    &stride, &dilation, with_str ? &istride : NULL,
                                    true, NO_USE, PAD_CONSTANT, idtype[opd0_type_idx], opd1type[opd1_type_idx]);
        }
        tpu_poll();
        et = firmware_timer_get_time_us();
        printf("<bank conflict> opd0_addr:%d, opd1_addr:%d, res_addr:%d\n"
                "loop time =%d: with_tensor_str=%d avg time=%lldus Tops/s=%.4f\n",
                opd0_addr, opd1_addr, res_addr, loop, with_str, (et - st) / loop, ops * loop / ((float)(et - st) * 1e6));
      }
    }
  }
}

static void nodechip_conv_bw_fp32_tf32_perf_test() {
  tpu_initialize();
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();

  padding_t pad = {1, 1, 1, 1};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim2 insert = {0, 0};

  u64 st, et;
  u64 ops;
  dim4 istride;
  int loop = 5;

  ///////////////////////  max perf case  ///////////////////////////
  dim2 kernel = {4, 4};
  int n = 1;
  int ic = 1;
  int oc = NPU_NUM;
  int ih = 16;
  int iw = 16;

  int with_stride_origin_ih = 24;
  int with_stride_origin_iw = 24;

  //forward infernece h and w, so exchange bw's dilation and stride
  int oh = cal_conv2d_out_size(ih, kernel.h, stride.h, dilation.h, pad.bottom, pad.top, insert.h);
  int ow = cal_conv2d_out_size(iw, kernel.w, stride.w, dilation.w, pad.left, pad.right, insert.w);
  dim4 opd0_shape = {n, ic, ih, iw};
  dim4 opd1_shape = {n, oc, oh, ow};
  data_type_t opd0_dtype_li[2] = {DT_FP32, DT_TF32};
  char *fidtype_c[2] = {"FP32", "TF32"};

  for (int use_tf32 = 0; use_tf32 < 2; ++use_tf32) {
    printf("conv bw %s opd0_shape=(%d %d %d %d) opd1_shape=(%d %d %d %d) kh=%d kw=%d\n",
          fidtype_c[use_tf32],
          opd0_shape.n, opd0_shape.c, opd0_shape.h, opd0_shape.w,
          opd1_shape.n, opd1_shape.c, opd1_shape.h, opd1_shape.w, kernel.h, kernel.w);
    data_type_t opd0_dtype = opd0_dtype_li[use_tf32];
    data_type_t opd1_dtype = opd0_dtype;

    for (int with_str = 0; with_str < 2; with_str++) {
      dim4 with_str_shape = {n, ic, with_stride_origin_ih, with_stride_origin_iw};
      istride = check_mem_valid(with_str ? with_str_shape : opd0_shape, DT_FP32, 6 * bank_size);

      dim4 opd1_stride = check_mem_valid(opd1_shape, DT_FP32, 6 * bank_size);

      dim4 res_shape = {1, oc, 1, 1};
      if (opd0_dtype == DT_FP32) {
        res_shape.h = ic * kernel.h * kernel.w;
      } else if (opd0_dtype == DT_TF32) {
        res_shape.h = DIV_UP(ic, tpu_eu_num(opd0_dtype)) * kernel.h * kernel.w;
        res_shape.w = tpu_eu_num(opd0_dtype);
      }
      dim4 res_stirde = check_mem_valid(res_shape, DT_FP32, 4 * bank_size);

      // no bank conflict
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_conv2d_backward(12 * bank_size, 0, 6 * bank_size, NO_USE, &opd0_shape, &opd1_shape, &kernel, &insert, &pad,
                                      &stride, &dilation, with_str ? &istride : NULL,
                                      true, NO_USE, PAD_CONSTANT, opd0_dtype, opd1_dtype);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      ops = conv_bw_ops(opd0_shape.n, opd0_shape.c, opd1_shape.c, opd1_shape.h, opd1_shape.w, kernel.h, kernel.w);
      printf("loop time =%d: with_tensor_str=%d avg time=%lldus Tops/s=%.4f\n",
              loop, with_str, (et - st) / loop, ops * loop / ((float)(et - st) * 1e6));

      // bank conflict
      local_addr_t opd0_addr = 0;
      local_addr_t opd1_addr = opd0_addr + istride.n * (with_str ? with_str_shape.n : opd0_shape.n) * tpu_data_type_size(opd0_dtype);
      local_addr_t res_addr = opd1_addr + opd1_stride.n * opd1_shape.n * tpu_data_type_size(opd1_dtype);
      ASSERT((int)res_addr + res_stirde.n * res_shape.n * tpu_data_type_size(DT_FP32) < LOCAL_MEM_SIZE);
      st = firmware_timer_get_time_us();
      for (int i = 0; i < loop; i++) {
        tpu_bdc_fp_conv2d_backward(res_addr, opd0_addr, opd1_addr, NO_USE, &opd0_shape, &opd1_shape, &kernel, &insert, &pad,
                                      &stride, &dilation, with_str ? &istride : NULL,
                                      true, NO_USE, PAD_CONSTANT, opd0_dtype, opd1_dtype);
      }
      tpu_poll();
      et = firmware_timer_get_time_us();
      printf("<bank conflict> opd0_addr:%d, opd1_addr:%d, res_addr:%d\n"
            "loop time =%d: with_tensor_str=%d avg time=%lldus Tops/s=%.4f\n",
            opd0_addr, opd1_addr, res_addr, loop, with_str, (et - st) / loop, ops * loop / ((float)(et - st) * 1e6));
    }
  }
}

void nodechip_conv_bw_perf_test() {
  nodechip_conv_bw_fp32_tf32_perf_test();
  nodechip_conv_bw_fp8_perf_test();
}