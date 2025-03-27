#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#include "split_util.h"

static inline void fill_data(u64 addr, u64 num_elem, int elem_size){
    void *data = tpu_global_mem_addr(addr);
    int zero_ratio = 4;
    int offset = -8;
    if(elem_size==1){
        u8 *ptr = (u8 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = ((i+1)&((1<<(elem_size*zero_ratio))-1))+offset;
        }
    } else if(elem_size==2){
        u16 *ptr = (u16 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = ((i+1)&((1<<(elem_size*zero_ratio))-1))+offset;
        }
    } else if(elem_size==4){
        u32 *ptr = (u32 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = ((i+1)&((1<<(elem_size*zero_ratio))-1))+offset;
        }
    }
    tpu_flush_cache(addr, ALIGN(num_elem*elem_size, tpu_cache_line_size()));
}


static const int loop = 5;
#define BEGIN(loop)                                \
  {                                                \
    u64 start_time = firmware_timer_get_time_us(); \
    int __loop = (loop);                           \
    for (int __i = 0; __i < __loop; __i++)         \
    {

#define END(info, slice, dtype)                                                                \
  }                                                                                            \
  tpu_poll();                                                                                  \
  u64 end_time = firmware_timer_get_time_us();                                                 \
  dim4 *pshape = (slice);                                                                      \
  u64 bytes = pshape->n * pshape->c * pshape->w * pshape->h * tpu_data_type_size(dtype);       \
  float avg_time = 1.0 * (end_time - start_time) / __loop;                                     \
  float bw = ((float)__loop * bytes * 1e6 / (end_time - start_time)) / (1024 * 1024 * 1024.f); \
  (void)(avg_time);(void)(bw);                                                                 \
  CORE_PRINT("Total %s time: %lldus, loop:%d, bytes=%lld, avg_time:%gus, bw=%gGB/s\n",          \
            info, (end_time - start_time), __loop, bytes, avg_time, bw);                       \
  }

int get_compressd_bytes(global_addr_t addr, int raw_data_bytes){
  tpu_invalidate_cache(addr, tpu_cache_line_size());
  int head_len = 4;
  int kmap_size = ALIGN(DIV_UP(raw_data_bytes, 16), 16);
  unsigned int* ptr = tpu_global_mem_addr(addr);
  int payload_size = (*ptr)&0xFFFFFF;
  return head_len + kmap_size + payload_size;
}

static void __nodechip_compress_normal_test(
  global_addr_t input_addr,
  global_addr_t compressed_addr,
  global_addr_t output_addr,
  dim4 *shape,
  data_type_t dtype,
  char bias0,
  char bias1,
  int zero_guard
) {

  dim4 local_stride = get_local_stride(shape, dtype, 0);
  dim4 *compress_stride = &local_stride;
  dim4 *decompress_stride = compress_stride;
  local_addr_t input_local_addr = 0;
  local_addr_t decompress_local_addr = LOCAL_MEM_SIZE/2;

  CORE_PRINT("%s input_addr=0x%llx, compressed_addr=0x%llx, output_addr=0x%llx,"
         "input_local_data=0x%x, decompress_local_data=0x%x, "
         "shape (%d, %d, %d, %d),"
         "compress_stride (%d, %d, %d, %d),"
         "decompress_stride (%d, %d, %d, %d),"
         "bias0=%d bias1=%d dtype=%d, zero_guard:%d\n",
         __func__,
         input_addr, compressed_addr, output_addr,
         input_local_addr, decompress_local_addr,
         shape->n, shape->c, shape->h, shape->w,
         compress_stride->n, compress_stride->c, compress_stride->h, compress_stride->w,
         decompress_stride->n, decompress_stride->c, decompress_stride->h, decompress_stride->w,
         bias0, bias1, dtype, zero_guard);

  int tensor_size = get_tensor_local_mem_size(shape, dtype, 1);
  ASSERT((int)(input_local_addr + tensor_size)<=(int)LOCAL_MEM_SIZE);
  ASSERT((int)(decompress_local_addr + tensor_size)<=LOCAL_MEM_SIZE);
  // 1. mv raw data to local mem
  BEGIN(loop)
  tpu_gdma_cpy_S2L(input_local_addr, input_addr, shape, compress_stride, NULL, dtype);
  END("copy_S2L", shape, dtype)
  // 2. compress local mem to compressed_addr

  BEGIN(loop)
  tpu_gdma_compress_normal_L2S(compressed_addr, input_local_addr, shape, compress_stride, dtype, bias0, bias1, zero_guard);
  END("compress", shape, dtype)
  int raw_bytes = shape->n * shape->c * shape->h * shape->w * tpu_data_type_size(dtype);
  int compressd_bytes = get_compressd_bytes(compressed_addr, raw_bytes);
  (void)(compressd_bytes);
  CORE_PRINT("  --> dsize=%d, raw_bytes=%d, compressed_bytes=%d, compress_ratio=%g\n",
            tpu_data_type_size(dtype), raw_bytes, compressd_bytes, 1.0 * compressd_bytes / raw_bytes);
  // 3. decompress compressed_addr to local mem
  BEGIN(loop)
  tpu_gdma_decompress_normal_S2L(decompress_local_addr, compressed_addr, shape, decompress_stride, dtype, bias0, bias1, zero_guard);
  END("decompress", shape, dtype)
  // 4. mv local mem data to output_data

  BEGIN(loop)
  tpu_gdma_cpy_L2S(output_addr, decompress_local_addr, shape, NULL, decompress_stride, dtype);
  END("copy_L2S", shape, dtype)
  CORE_PRINT("\n");
}

static void __nodechip_compress_RACU_test(
  global_addr_t input_addr,
  global_addr_t compressed_racu_addr,
  global_addr_t compressed_meta_addr,
  global_addr_t output_addr,
  dim4 *shape,
  data_type_t dtype,
  int split,
  char bias0,
  char bias1,
  int zero_guard
) {
  local_addr_t input_local_addr = 0;
  local_addr_t decompress_local_addr = LOCAL_MEM_SIZE/2;

  dim4 local_stride = get_local_stride(shape, dtype, 0);
  dim4 *compress_stride = &local_stride;

  dim4 *decompress_stride = compress_stride;
  dim4 racu_stride = tpu_gdma_compress_RACU_racu_stride(shape, dtype, zero_guard);
  dim4 meta_stride = tpu_gdma_compress_RACU_meta_stride(shape, dtype);

  CORE_PRINT("%s input_addr=0x%llx, compressed_racu_addr=0x%llx, compressed_meta_addr=0x%llx, output_addr=0x%llx,"
         "input_local_data=0x%x, decompress_local_data=0x%x, "
         "shape (%d, %d, %d, %d),"
         "compress_stride (%d, %d, %d, %d),"
         "decompress_stride (%d, %d, %d, %d),"
         "bias0=%d bias1=%d dtype=%d, zero_guard=%d\n",
         __func__,
         input_addr, compressed_racu_addr, compressed_meta_addr, output_addr,
         input_local_addr, decompress_local_addr,
         shape->n, shape->c, shape->h, shape->w,
         compress_stride->n, compress_stride->c, compress_stride->h, compress_stride->w,
         decompress_stride->n, decompress_stride->c, decompress_stride->h, decompress_stride->w,
         bias0, bias1, dtype, zero_guard);

  int tensor_size = get_tensor_local_mem_size(shape, dtype, 1);
  ASSERT(tensor_size<=LOCAL_MEM_SIZE);

  // 1. mv raw data to local mem
  BEGIN(loop)
  tpu_gdma_cpy_S2L(input_local_addr, input_addr, shape, compress_stride, NULL, dtype);
  END("copy_S2L", shape, dtype)

  // 2. compress local mem to compressed_addr
  BEGIN(loop)
  tpu_gdma_compress_RACU_L2S(
      compressed_racu_addr,
      compressed_meta_addr,
      input_local_addr,
      shape,
      &racu_stride,
      &meta_stride,
      compress_stride,
      dtype, bias0, bias1, zero_guard);
  END("compress", shape, dtype)
  CORE_PRINT("racu_stride (%d, %d, %d, %d), "
         "meta_stride (%d, %d, %d, %d)\n",
         racu_stride.n, racu_stride.c, racu_stride.h, racu_stride.w,
         meta_stride.n, meta_stride.c, meta_stride.h, meta_stride.w);

  // 3. decompress compressed_addr to local mem
  const int max_n = (shape->n + split-1) / split;
  const int max_h = (shape->h + split-1) / split;
  const int max_c = NPU_NUM; // must be times of NPU_NUM
  dim4 slice = {max_n, max_c, max_h, shape->w};
  dim4 index = {0, 0, 0, 0};
  while (index.n < shape->n) {
    slice.n = MIN(max_n, shape->n - index.n);
    while (index.c < shape->c) {
      slice.c = MIN(max_c, shape->c - index.c);
      while (index.h < shape->h) {
        slice.h = MIN(max_h, shape->h - index.h);

        int out_offset = index.n * decompress_stride->n +
                         (index.c / NPU_NUM) * decompress_stride->c +
                         index.h * decompress_stride->h;
        out_offset *= tpu_data_type_size(dtype);

        // racu_stride use byte as unit
        int racu_offset = index.n * racu_stride.n +
                          (index.c / NPU_NUM) * racu_stride.c +
                          index.h * racu_stride.h;
        // meta_stride use U32 as unit
        int meta_offset = index.n * meta_stride.n +
                          (index.c / NPU_NUM) * meta_stride.c +
                          index.h * meta_stride.h;
        meta_offset *= 4;

        BEGIN(loop)
        tpu_gdma_decompress_RACU_S2L(
            decompress_local_addr + out_offset,
            compressed_racu_addr + racu_offset,
            compressed_meta_addr + meta_offset,
            &slice,
            decompress_stride,
            &racu_stride,
            &meta_stride,
            dtype, bias0, bias1, zero_guard);
        END("decompress", &slice, dtype)
        CORE_PRINT("index (%d, %d, %d, %d), "
               "slice (%d, %d, %d, %d), "
               "out_offset=%d, racu_offset=%d, meta_offset=%d\n",
               index.n, index.c, index.h, index.w,
               slice.n, slice.c, slice.h, slice.w,
               out_offset, racu_offset, meta_offset);
        index.h += slice.h;
        index.w = 0;
      }
      index.c += slice.c;
      index.h = 0;
      index.w = 0;
    }
    index.n += slice.n;
    index.c = 0;
    index.h = 0;
    index.w = 0;
  }

  // 4. mv local mem data to output_data
  BEGIN(loop)
  tpu_gdma_cpy_L2S(output_addr, decompress_local_addr, shape, NULL, decompress_stride, dtype);
  END("copy_L2S", shape, dtype)
  CORE_PRINT("\n");
}


void nodechip_gdma_compress_test(
  global_addr_t input_addr,
  global_addr_t output_addr
) {

  tpu_initialize();
  int zero_guard = 0;
  int bias0 = 0;
  int bias1 = 0;
  data_type_t dtypes[] = {DT_INT8, DT_INT16};

  dim4 base_shapes[] = {
    {2, NPU_NUM, 32, 32},
    {2, NPU_NUM, 32, 128},
    {2, 2*NPU_NUM, 32, 128},
    {4, NPU_NUM, 64, 128},
    {4, NPU_NUM, 128, 64},
    {1, NPU_NUM, 128, 256},
    {1, NPU_NUM, 256, 128},
    {4, NPU_NUM, 128, 1},
    {4, NPU_NUM, 256, 1},
    {4, NPU_NUM, 1, 256},
  };
  int max_elem_num = 256*256*NPU_NUM*4*2;
    for (size_t dtype_idx = 0; dtype_idx < sizeof(dtypes) / sizeof(dtypes[0]); dtype_idx++) {
      data_type_t dtype = dtypes[dtype_idx];
      fill_data(input_addr, max_elem_num, tpu_data_type_size(dtype));
      for (size_t i = 0; i < sizeof(base_shapes) / sizeof(base_shapes[0]); i++) {
        dim4 shape = base_shapes[i];
        __nodechip_compress_normal_test(
            input_addr,
            output_addr,
            input_addr,
            &shape,
            dtype,
            bias0,
            bias1,
            zero_guard);
        __nodechip_compress_RACU_test(
            input_addr,
            output_addr,
            input_addr,
            output_addr,
            &shape,
            dtype,
            2,
            bias0,
            bias1,
            zero_guard);
      }
    }
}

// #endif
