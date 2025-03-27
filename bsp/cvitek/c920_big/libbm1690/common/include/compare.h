#ifndef COMPARE_H
#define COMPARE_H
#include "common.h"
#include "math.h"
#include "macros.h"
#include "sg_fp16.h"
#include "common_utils.h"

/* Info about data compare */
static inline int array_cmp_fix8b(
    void *p_exp,
    void *p_got,
    int sign,  // 0: unsigned, 1: signed
    int len,
    const char *info_label,
    int delta)
{
  int idx = 0;
  int first_error_idx = -1, first_expect_value = 0, first_got_value = 0;
  int max_error_idx   = -1, max_expect_value   = 0, max_got_value   = 0;
  int max_error_value = delta,  mismatch_cnt = 0;
  for (idx = 0; idx < len; idx++) {
    int error   = 0;
    int exp_int = 0;
    int got_int = 0;
    if (sign) {
      exp_int = (int)(*((char *)p_exp + idx));
      got_int = (int)(*((char *)p_got + idx));
    } else {
      exp_int = (int)(*((unsigned char *)p_exp + idx));
      got_int = (int)(*((unsigned char *)p_got + idx));
    }
    error = abs(exp_int - got_int);
    if (error > 0) {
      if (first_error_idx == -1) {
        first_error_idx = idx;
        first_expect_value = exp_int;
        first_got_value = got_int;
      }
      if(error > max_error_value) {
        max_error_idx = idx;
        max_error_value = error;
        max_expect_value = exp_int;
        max_got_value = got_int;
      }
      mismatch_cnt ++;
    }
    if (error > delta) {
      printf("%s     error      at index %d exp %d got %d\n", info_label, idx, exp_int, got_int);
      printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
      printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
      fflush(stdout);
      FILE* ofp = fopen("out.dat", "w");
      FILE* rfp = fopen("ref.dat", "w");
      for(int i = 0; i< len; i++){
          fprintf(ofp, "0x%02x\n", 0xFF&((char*)p_got)[i]);
          fprintf(rfp, "0x%02x\n", 0xFF&((char*)p_exp)[i]);
      }
      fclose(ofp);
      fclose(rfp);
      return -1;
    }
  }
  if(max_error_idx != -1) {
    printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
    printf("%s  max  mismatch at index %d exp %d got %d (delta %d)\n", info_label, max_error_idx, max_expect_value, max_got_value, delta);
    printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
    fflush(stdout);
  }

  return 0;
}

static inline int array_cmp_fix16b(
    void *p_exp,
    void *p_got,
    int sign,  // 0: unsigned, 1: signed
    int len,
    const char *info_label,
    int delta) {
  int idx = 0;
  int first_error_idx = -1, first_expect_value = 0, first_got_value = 0;
  int max_error_idx   = -1, max_expect_value   = 0, max_got_value   = 0;
  int max_error_value = delta,  mismatch_cnt = 0;
  for (idx = 0; idx < len; idx++) {
    int error   = 0;
    int exp_int = 0;
    int got_int = 0;
    if (sign) {
      exp_int = (int)(*((short *)p_exp + idx));
      got_int = (int)(*((short *)p_got + idx));
    } else {
      exp_int = (int)(*((unsigned short *)p_exp + idx));
      got_int = (int)(*((unsigned short *)p_got + idx));
    }
    error = abs(exp_int - got_int);
    if (error > 0) {
      if (first_error_idx == -1) {
        first_error_idx = idx;
        first_expect_value = exp_int;
        first_got_value = got_int;
      }
      if(error > max_error_value) {
        max_error_idx = idx;
        max_error_value = error;
        max_expect_value = exp_int;
        max_got_value = got_int;
      }
      mismatch_cnt ++;
    }
    if (error > delta) {
      printf("%s     error      at index %d exp %d got %d\n", info_label, idx, exp_int, got_int);
      printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
      printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
      fflush(stdout);
      return -1;
    }
  }
  if(max_error_idx != -1) {
    printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
    printf("%s  max  mismatch at index %d exp %d got %d (delta %d)\n", info_label, max_error_idx, max_expect_value, max_got_value, delta);
    printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
    fflush(stdout);
  }

  return 0;
}

#define IS_NAN(x) ((((x >> 23) & 0xff) == 0xff) && ((x & 0x7fffff) != 0))
static inline int array_cmp(float *p_exp, float *p_got, int len, const char *info_label, float delta) {
  int max_error_count = 128;
  int idx = 0;
  int total = 0;
  int* p_exp_raw = (int*)(p_exp);
  int* p_got_raw = (int*)(p_got);
  bool only_warning = false;
  if (1e4 <= delta) {
    delta = 1e-2;
    only_warning = true;
  }
  for (idx = 0; idx < len; idx++) {
    if (sg_max(fabs(p_exp[idx]), fabs(p_got[idx])) > 1.0) {
      // compare rel
      if (sg_min(fabs(p_exp[idx]), fabs(p_got[idx])) < 1e-20) {
        printf("%s(): %s rel warning at index %d exp %.20f got %.20f\n", __FUNCTION__, info_label, idx, p_exp[idx], p_got[idx]);
        fflush(stdout);
        total++;
        if (max_error_count < total && !only_warning) {return -1;}
      }
      if (fabs(p_exp[idx] - p_got[idx]) / sg_min(fabs(p_exp[idx]), fabs(p_got[idx])) > delta) {
        printf(
            "%s(): %s rel warning at index %d exp %.20f(0x%08X) got %.20f(0x%08X), diff=%.20f\n",
            __FUNCTION__,
            info_label, idx,
            p_exp[idx], p_exp_raw[idx],
            p_got[idx], p_got_raw[idx],
            p_exp[idx] - p_got[idx]);
        fflush(stdout);
        total++;
        if (max_error_count < total && !only_warning) {return -1;}
      }
    } else {
      // compare abs
      if (fabs(p_exp[idx] - p_got[idx]) > delta) {
        printf(
            "%s(): %s abs warning at index %d exp %.20f(0x%08X) got %.20f(0x%08X), diff=%.20f\n",
            __FUNCTION__,
            info_label, idx,
            p_exp[idx], p_exp_raw[idx],
            p_got[idx], p_got_raw[idx],
            p_exp[idx] - p_got[idx]);
        fflush(stdout);
        total++;
        if (max_error_count < total && !only_warning) {return -1;}
      }
    }

    DataUnion if_val_exp, if_val_got;
    if_val_exp.f32val = p_exp[idx];
    if_val_got.f32val = p_got[idx];
    if (IS_NAN(if_val_got.i32val) && !IS_NAN(if_val_exp.i32val)) {
      printf("There are nans in %s idx %d\n", info_label, idx);
      printf("floating form exp %.10f got %.10f\n", if_val_exp.f32val, if_val_got.f32val);
      printf("hex form exp %8.8x got %8.8x\n", if_val_exp.i32val, if_val_got.i32val);
      fflush(stdout);
      return -2;
    }
  }
  if (0 < total && !only_warning) {return -1;}
  return 0;
}

static inline int array_cmp_int32(
    int *p_exp, int *p_got, int len, const char *info_label, int delta)
{
  int idx = 0;
  int first_error_idx = -1, first_expect_value = 0, first_got_value = 0;
  int max_error_idx   = -1, max_expect_value   = 0, max_got_value   = 0;
  int max_error_value = 0,  mismatch_cnt = 0;
  for (idx = 0; idx < len; idx++) {
    int error   = 0;
    int exp_int = 0;
    int got_int = 0;
    exp_int = *(p_exp + idx);
    got_int = *(p_got + idx);
    error = abs(exp_int - got_int);
    if (error > 0) {
      if (first_error_idx == -1) {
        first_error_idx = idx;
        first_expect_value = exp_int;
        first_got_value = got_int;
      }
      if (error > max_error_value) {
        max_error_idx = idx;
        max_error_value = error;
        max_expect_value = exp_int;
        max_got_value = got_int;
      }
      mismatch_cnt++;
    }
    if (error > delta) {
      printf("%s     error      at index %d exp %d got %d\n", info_label, idx, exp_int, got_int);
      printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
      printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
      fflush(stdout);
      return -1;
    }
  }
  if(max_error_idx != -1) {
    printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
    printf("%s  max  mismatch at index %d exp %d got %d (delta %d)\n", info_label, max_error_idx, max_expect_value, max_got_value, delta);
    printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
    fflush(stdout);
  }
  return 0;
}

/* Info about data compare */
static inline int array_cmp_fix4b(
    void *p_exp,
    void *p_got,
    int sign,  // 0: unsigned, 1: signed
    int len,
    const char *info_label,
    int delta)
{
  int idx = 0;
  int first_error_idx = -1, first_expect_value = 0, first_got_value = 0;
  int max_error_idx   = -1, max_expect_value   = 0, max_got_value   = 0;
  int max_error_value = 0,  mismatch_cnt = 0;
  unsigned char* exp_ptr = (unsigned char*)p_exp;
  unsigned char* got_ptr = (unsigned char*)p_got;
  for (idx = 0; idx < len; idx++) {
    int error   = 0;
    int exp_int = int4_to_value(exp_ptr[idx>>1], idx&0x1, sign);
    int got_int = int4_to_value(got_ptr[idx>>1], idx&0x1, sign);
    error = abs(exp_int - got_int);
    if (error > 0) {
      if (first_error_idx == -1) {
        first_error_idx = idx;
        first_expect_value = exp_int;
        first_got_value = got_int;
      }
      if(error > max_error_value) {
        max_error_idx = idx;
        max_error_value = error;
        max_expect_value = exp_int;
        max_got_value = got_int;
      }
      mismatch_cnt ++;
    }
    if (error > delta) {
      printf("%s     error      at index %d exp %d got %d\n", info_label, idx, exp_int, got_int);
      printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
      printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
      fflush(stdout);
      FILE* ofp = fopen("compare.dat", "w");
      fprintf(ofp, "out  ref  ref-out\n");
      for(int i = 0; i< len; i++){
        int exp_int = int4_to_value(exp_ptr[idx >> 1], idx & 0x1, sign);
        int got_int = int4_to_value(got_ptr[idx >> 1], idx & 0x1, sign);
        fprintf(ofp, "%d[0x%x]  %d[%0x] %d\n", got_int, got_int&0xF, exp_int, exp_int&0xF, exp_int-got_int);
      }
      fclose(ofp);
      return -1;
    }
  }
  if(max_error_idx != -1) {
    printf("%s first mismatch at index %d exp %d got %d (delta %d)\n", info_label, first_error_idx, first_expect_value, first_got_value, delta);
    printf("%s  max  mismatch at index %d exp %d got %d (delta %d)\n", info_label, max_error_idx, max_expect_value, max_got_value, delta);
    printf("%s total mismatch count %d (delta %d) \n", info_label, mismatch_cnt, delta);
    fflush(stdout);
  }

  return 0;
}

static inline int array_cmp_fp16(
    void *p_exp,
    void *p_got,
    int len,
    const char *info_label,
    float delta) {
      float* fp32_exp = (float*) malloc(len * sizeof(float));
      float* fp32_got = (float*) malloc(len * sizeof(float));
      fp16* f16_exp = (fp16*)p_exp;
      fp16* f16_got = (fp16*)p_got;
      for(int i=0; i<len; i++){
        fp32_exp[i] = fp16_to_fp32(f16_exp[i]).fval;
        fp32_got[i] = fp16_to_fp32(f16_got[i]).fval;
      }
      int ret = array_cmp(fp32_exp, fp32_got, len, info_label, delta);
      free(fp32_exp);
      free(fp32_got);
      return ret;
}

static inline int array_cmp_bf16(
    void *p_exp,
    void *p_got,
    int len,
    const char *info_label,
    float delta) {
      float* fp32_exp = (float*) malloc(len * sizeof(float));
      float* fp32_got = (float*) malloc(len * sizeof(float));
      bf16* f16_exp = (bf16*)p_exp;
      bf16* f16_got = (bf16*)p_got;
      for(int i=0; i<len; i++){
        fp32_exp[i] = bf16_to_fp32(f16_exp[i]).fval;
        fp32_got[i] = bf16_to_fp32(f16_got[i]).fval;
      }
      int ret = array_cmp(fp32_exp, fp32_got, len, info_label, delta);
      free(fp32_exp);
      free(fp32_got);
      return ret;
}

static inline
int array_cmp_fp8(void* p_exp, void* p_got, int len, const char* info, float delta, bool is_e5m2)
{
  float* exp_f32 = new float[len];
  float* got_f32 = new float[len];
  for (int i = 0; i < len; i++) {
    exp_f32[i] = fp8_to_fp32(((uint8_t*)p_exp)[i], is_e5m2).fval;
    got_f32[i] = fp8_to_fp32(((uint8_t*)p_got)[i], is_e5m2).fval;
  }
  int ret = array_cmp(exp_f32, got_f32, len, info, delta);

  delete [] exp_f32;
  delete [] got_f32;
  return ret;
}

static inline int array_cmp_all(const void* exp_data, const void* got_data, int block_size, sg_data_type_t dtype, float delta){
    int cmp_res = 0;
    unsigned char* u8_exp =(unsigned char*)exp_data;
    unsigned char* u8_got =(unsigned char*)got_data;
    if (dtype == SG_DTYPE_FP32){
        cmp_res = array_cmp((float*)u8_exp, (float*)u8_got, block_size, "Compare float32 ... \n", delta);
    } else if (dtype == SG_DTYPE_FP16){
        cmp_res = array_cmp_fp16(u8_exp, u8_got, block_size, "Comare fp16 ... \n", delta);
    } else if (dtype == SG_DTYPE_BFP16){
        cmp_res = array_cmp_bf16(u8_exp, u8_got, block_size, "Comare bf16 ... \n", delta);
    } else if (dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32){
        cmp_res = array_cmp_int32((int*)u8_exp, (int*)u8_got, block_size, "Compare int32 ...\n", delta);
    } else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16) {
        cmp_res = array_cmp_fix16b((void*)u8_exp, (void*)u8_got, dtype == SG_DTYPE_INT16 ? 1 : 0, block_size, "Compare int16/uint16 ...\n", delta);
    } else if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) {
        cmp_res = array_cmp_fix8b((void*)u8_exp, (void*)u8_got, dtype == SG_DTYPE_INT8 ? 1 : 0, block_size, "Compare int8/uint8 ...\n", delta);
    } else if (dtype == SG_DTYPE_INT4 || dtype == SG_DTYPE_UINT4) {
        cmp_res = array_cmp_fix4b((void*)u8_exp, (void*)u8_got, dtype == SG_DTYPE_UINT4 ? 1 : 0, block_size, "Compare int4/uint4 ...\n", delta);
    } else if (dtype == SG_DTYPE_FP8E4M3 || dtype == SG_DTYPE_FP8E5M2) {
        cmp_res  = array_cmp_fp8((void*)u8_exp, (void*)u8_got, block_size, "fp8 cmp", 1e-2,
                   dtype == SG_DTYPE_FP8E5M2 ? true : false);
    }
    return cmp_res;
}

static inline int array_cmp_all_by_nc(int N, int C, int HW, sg_data_type_t dtype, const void* exp_data, const void* got_data, int delta=0){
        int cmp_res = 0;
        int block_size = HW;
        unsigned char* u8_exp =(unsigned char*)exp_data;
        unsigned char* u8_got =(unsigned char*)got_data;
        for(int i=0; i<N; i++){
            for(int j=0; j<C; j++){
                if (dtype == SG_DTYPE_FP32){
                    cmp_res = array_cmp((float*)u8_exp, (float*)u8_got, block_size,
                                            "Compare float32 ... \n", 0.01);
                    u8_exp += 4*block_size;
                    u8_got += 4*block_size;
                } else if (dtype == SG_DTYPE_FP16){
                    cmp_res = array_cmp_fp16(u8_exp, u8_got, block_size, "Comare fp16 ... \n", 0.01);
                    u8_exp += 2*block_size;
                    u8_got += 2*block_size;
                } else if (dtype == SG_DTYPE_BFP16){
                    cmp_res = array_cmp_bf16(u8_exp, u8_got, block_size, "Comare bf16 ... \n", 0.01);
                    u8_exp += 2*block_size;
                    u8_got += 2*block_size;
                } else if (dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32){
                    cmp_res = array_cmp_int32((int*)u8_exp, (int*)u8_got, block_size,
                                            "Compare int32 ...\n", 0);
                    u8_exp += 4*block_size;
                    u8_got += 4*block_size;
                } else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16) {
                    cmp_res = array_cmp_fix16b((void*)u8_exp, (void*)u8_got,
                                            dtype == SG_DTYPE_INT16 ? 1 : 0,
                                            block_size, "Compare int16/uint16 ...\n", delta);
                    u8_exp += 2*block_size;
                    u8_got += 2*block_size;
                } else if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) {
                    cmp_res = array_cmp_fix8b((void*)u8_exp, (void*)u8_got,
                                            dtype == SG_DTYPE_INT8 ? 1 : 0,
                                            block_size, "Compare int8/uint8 ...\n", delta);
                    u8_exp += block_size;
                    u8_got += block_size;
                } else if (dtype == SG_DTYPE_INT4 || dtype == SG_DTYPE_UINT4) {
                    cmp_res = array_cmp_fix4b((void*)u8_exp, (void*)u8_got,
                                            dtype == SG_DTYPE_UINT4 ? 1 : 0,
                                            block_size, "Compare int4/uint4 ...\n", 0);
                    u8_exp += (block_size+1)>>1;
                    u8_got += (block_size+1)>>1;
                }
                if (cmp_res != 0) {
                    printf("Comparison failed for N=%d, C=%d\n",i, j);
                    return cmp_res;
                }
            }
        }
        return cmp_res;
}


#endif
