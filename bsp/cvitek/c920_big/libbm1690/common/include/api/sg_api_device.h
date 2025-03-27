#pragma once

#pragma pack(push, 1)
#include "common_def.h"

typedef enum {
  DEVICE_TEST_COMPRESS = 0,
  DEVICE_TEST_REVERSE  = 1,
} sg_device_test_id_t;

typedef struct {
  u64 input_addr;
  u64 compress_addr;
  u64 output_addr;
  int shape[4];
  sg_data_type_t dtype;
  char bias0;
  char bias1;
} sg_device_compress_normal_param_t

typedef struct {
  u64 input_addr;
  u64 compress_racu_addr;
  u64 compress_meta_addr;
  u64 output_addr;
  int shape[4];
  sg_data_type_t dtype;
  char bias0;
  char bias1;
} sg_device_compress_RACU_param_t

typedef struct {
  u64 input_addr;
  u64 output_addr;
  int shape[4];
} sg_device_local_reverse_param_t;

typedef struct {
  uint64_t addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  sg_data_type_t dtype;
} sg_device_tensor_t;

typedef struct {
  sg_device_test_id_t id;
  int input_num;
  int output_num;
  int param_size;
} sg_device_test_param_t;

#pragma pack(pop)