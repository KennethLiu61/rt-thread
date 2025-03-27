#pragma once
#include <typeinfo>
#include "sg_fp16.h"
#include "common.h"
#include "common_def.h"
#include "common_utils.h"

static inline int conv_get_nIC(int npu_num, PREC dtype) {
    if (dtype == FP32)
        return 1;
    else if(dtype == INT4)
        return npu_num*2;
#if defined (__sg2380__) || defined (__sg2262__)
    else if (dtype == BFP16 || dtype == FP16) {
        return 32;
    }
#endif
    else
        return (int)(npu_num/get_bytesize(dtype));
}

static inline int conv_get_nIC(int npu_num, sg_data_type_t dtype) {
    if (dtype == SG_DTYPE_FP32)
        return 1;
    else if(dtype == SG_DTYPE_INT4 || dtype == SG_DTYPE_UINT4)
        return npu_num*2;
#if defined (__sg2380__) || defined (__sg2262__)
    else if (dtype == SG_DTYPE_BFP16 || dtype == SG_DTYPE_FP16) {
        return 32;
    }
#endif
    else
        return (int)(npu_num/sg_dtype_len(dtype));
}

template<typename T>
T RightShiftRound(T src, int shift_num, ROUND_MODE round_mode)
{
  if (shift_num == 0) return src;
  if (shift_num > 63) shift_num = 63;
  T val, res;
  val = src >> shift_num;
  res = val;
  T lo_mask = (1ull << shift_num) - 1;
  T mant = src & lo_mask;
  T mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUND_HALF_TO_EVEN) {
    if (mant == mant_0d5) {
      res = val + (val & 1);
    } else if (mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5) {
      res = val + 1;
    } else if (src < 0 && mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_TOWARDS_ZERO) {
    if (src < 0) res = val + (mant != 0);
  } else if (round_mode == ROUND_DOWN) {
    res = val;
  } else if (round_mode == ROUND_UP) {
    res = val + (mant != 0);
  } else if (round_mode == ROUND_HALF_UP) {
    if (mant >= mant_0d5) res = val + 1;
  } else if (round_mode == ROUND_HALF_DOWN) {
    if (mant > mant_0d5) res = val + 1;
  }
  return res;
}

template <typename T>
T ArithShiftRound(T src, int shift_num, ROUND_MODE rm_mode) {
  ASSERT(sizeof(T) == sizeof(int64_t));
  T res = 0;
  if (shift_num > 0) {
    int32_t bit_len = sizeof(T) * 8;
    int valid_bits = 0;
    T value = std::abs(src);
    while (value > 0) {
      valid_bits++;
      value = value >> 1;
    }

    if (typeid(T) == typeid(int64_t) || typeid(T) == typeid(long long) ||
        typeid(T) == typeid(signed long long)) {
      if (shift_num > (bit_len - valid_bits - 1)) {
        // shift
        res = src << shift_num;
        // clear signed bit
        res = res & (((T)1 << (bit_len - 1)) - 1);
        // assign signed bit
        res = res | (src & ((T)1 << (bit_len - 1)));
      } else {
        res = src << shift_num;
      }
    } else {
      res = src << shift_num;
    }
  } else if (shift_num < 0) {
    res = RightShiftRound<T>(src, std::abs(shift_num), rm_mode);
  } else {
    res = src;
  }
  return res;
}
// saturate INT32 to INT4/INT8/INT16/INT32
template<typename T>
inline static void saturate_to(T* src_ptr, int* dst_ptr, sg_data_type_t dtype, int sym_range) {
  long long satu_max = 0, satu_min = 0;
  switch (dtype)
  {
  case SG_DTYPE_INT4:
    satu_max = 7;
    satu_min = (sym_range ? -7 : -8);
    break;
  case SG_DTYPE_UINT4:
    satu_max = 15;
    satu_min = 0;
    break;
  case SG_DTYPE_INT8:
    satu_max = 127;
    satu_min = (sym_range ? -127 : -128);
    break;
  case SG_DTYPE_UINT8:
    satu_max = 255;
    satu_min = 0;
    break;
  case SG_DTYPE_INT16:
    satu_max = 32767;
    satu_min = (sym_range ? -32767 : -32768);
    break;
  case SG_DTYPE_UINT16:
    satu_max = 65535;
    satu_min = 0;
    break;
  case SG_DTYPE_INT32:
    satu_max = 2147483647;
    satu_min = (sym_range ? -2147483647 : -2147483648);
    break;
  case SG_DTYPE_UINT32:
    satu_max = 4294967295;
    satu_min = 0;
    break;
  default:
    break;
  }

    long long temp = (long long)src_ptr[0] > satu_max ? satu_max : (long long)src_ptr[0];
    temp = temp < satu_min ? satu_min : temp;
    *(int*)dst_ptr = (int)temp;
}

static inline void gen_rand_data_by_dtype(void *data_, int N, int C, int H, int W, sg_data_type_t dtype) {
    unsigned char* data = (unsigned char*)data_;
    int tensor_size = N*C*H*W;
    if (dtype == SG_DTYPE_FP32) {
        for (int idx = 0; idx < tensor_size; idx++) {
            ((float*)data)[idx] = 10*(1.0*(rand()&0xFFFF)/0xFFFF - 0.5);
        }
    } else if (dtype == SG_DTYPE_BFP16) {
      bf16* ptr = (bf16*)data;
      fp32 val;
      for(int idx = 0; idx<tensor_size; idx++){
          val.fval = 10*(1.0*(rand()&0xFFFF)/0xFFFF - 0.5);
          ptr[idx] = fp32_to_bf16(val, ROUND_HALF_TO_EVEN, false);
      }
    } else if (dtype == SG_DTYPE_FP16) {
      fp16* ptr = (fp16*)data;
      fp32 val;
      for(int idx = 0; idx<tensor_size; idx++){
          val.fval = 10*(1.0*(rand()&0xFFFF)/0xFFFF - 0.5);
          ptr[idx] = fp32_to_fp16(val, ROUND_HALF_TO_EVEN, false);
      }
    } else if (dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32) {
        for (int idx = 0; idx < tensor_size; idx++) {
            ((int*)data)[idx] = rand();
        }
    } else if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) {
        for (int idx = 0; idx < tensor_size; idx++) {
            ((unsigned char*)data)[idx] = rand()&0xFF;
        }
    } else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16) {
        for (int idx = 0; idx < tensor_size; idx++) {
            ((short*)data)[idx] = rand()&0xFFFF;
        }
    } else if (dtype == SG_DTYPE_UINT4 || dtype == SG_DTYPE_INT4) {
        int NC=N*C;
        int HW=H*W;
        int step = ALIGN(HW, 2)>>1;
        for (int i=0; i<NC; i++){
            for (int j=0; j<HW; j++){
                if(j&0x1){
                     ((unsigned char*)data)[i*step+(j>>1)] |= (rand()&0xF)<<4;
                } else {
                     ((unsigned char*)data)[i*step+(j>>1)] = (rand()&0xF);
                }
            }
        }
    } else if (dtype == SG_DTYPE_FP8E4M3 || dtype == SG_DTYPE_FP8E5M2) {
        uint8_t *ptr = (uint8_t*)data;
        fp32 val;
        for(int idx = 0; idx<tensor_size; idx++){
          val.fval = (1.0*(rand()&0xFFFF)/0xFFFF - 0.5);
          ptr[idx] = fp32_to_fp8(val, dtype == SG_DTYPE_FP8E4M3 ? false : true, false, ROUND_HALF_TO_EVEN);
        }
    } else {
        ASSERT_INFO(0, "not support dtype=%d", dtype);
    }
}

static std::vector<float> value_to_fp32(const void* data, int elem_num, sg_data_type_t dtype){
  std::vector<float> result(elem_num);
  if(dtype == SG_DTYPE_FP32){
    auto fp32_ptr = (const float*)data;
    result.assign(fp32_ptr, fp32_ptr+elem_num);
  } else if(dtype == SG_DTYPE_FP16) {
    auto fp16_ptr = (const fp16*)data;
    for(int i=0; i<elem_num; i++){
      auto value = fp16_ptr[i];
      result[i] = fp16_to_fp32(value).fval;
    }
  } else if(dtype == SG_DTYPE_BFP16) {
    auto bf16_ptr = (const bf16*)data;
    for(int i=0; i<elem_num; i++){
      auto value = bf16_ptr[i];
      result[i] = bf16_to_fp32(value).fval;
    }
  }
  return result;
}
static std::vector<long long> value_to_s64(const void* data, int elem_num, sg_data_type_t dtype){
    std::vector<long long> input(elem_num);
    if (dtype == SG_DTYPE_UINT32) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((unsigned int*)data)[i];
        }
    } else if (dtype == SG_DTYPE_INT32) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((int*)data)[i];
        }
    } else if (dtype == SG_DTYPE_INT16) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((short*)data)[i];
        }
    } else if(dtype == SG_DTYPE_UINT16) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((unsigned short*)data)[i];
        }
    } else if(dtype == SG_DTYPE_UINT8) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((unsigned char*)data)[i];
        }
    } else if(dtype == SG_DTYPE_INT8) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = ((signed char*)data)[i];
        }
    } else if(dtype == SG_DTYPE_UINT4) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = int4_to_value(((unsigned char*)data)[i>>1], i&0x1, 0);
        }
    } else if(dtype == SG_DTYPE_INT4) {
        for (int i = 0; i < elem_num; i++) {
            input[i] = int4_to_value(((unsigned char*)data)[i>>1], i&0x1, 1);
        }
    } else {
        ASSERT_INFO(0, "dtype not supported!");
    }
    return input;
}

#include <fstream>
template<typename T>
static void save_to_file(const T* data, int elem_num, const char* filename){
    std::ofstream ofs(filename);
    for (int i = 0; i < elem_num; i++) {
        ofs << data[i] << std::endl;
    }
}

static void save_to_file(const void* data, int elem_num, sg_data_type_t dtype, const char* filename){

    if (dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_BFP16) {
      auto f32_vec = value_to_fp32(data, elem_num, dtype);
      save_to_file(f32_vec.data(), elem_num, filename);
      return;
    }
    if (dtype == SG_DTYPE_INT4 || dtype == SG_DTYPE_UINT4) dtype = SG_DTYPE_INT8;
    auto s64_vec = value_to_s64(data, elem_num, dtype);
    save_to_file(s64_vec.data(), elem_num, filename);
}

#include <limits.h>
static void get_satu_limit(sg_data_type_t dtype, long long& satu_max, long long& satu_min){
  if (dtype == SG_DTYPE_INT16) {
      satu_max = 32767;
      satu_min = -32768;
  } else if (dtype == SG_DTYPE_INT32) {
      satu_max = INT_MAX;
      satu_min = INT_MIN;
  } else if (dtype == SG_DTYPE_INT8) {
      satu_max = 127;
      satu_min = -128;
  } else if (dtype == SG_DTYPE_UINT8) {
      satu_max = 255;
      satu_min = 0;
  } else if (dtype == SG_DTYPE_UINT16) {
      satu_max = 65535;
      satu_min = 0;
  } else if (dtype == SG_DTYPE_INT4) {
      satu_max = 7;
      satu_min = -8;
  } else if (dtype == SG_DTYPE_UINT4) {
      satu_max = 15;
      satu_min = 0;
  } else if (dtype == SG_DTYPE_UINT32) {
      satu_max = 0;
      satu_min = ((long long)1<<32)-1;
  }
}
