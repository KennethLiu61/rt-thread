#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>
#include "nodechip_pld_test.h"

static inline void fill_data(u64 addr, u64 num_elem){
    const int elem_size = 4;
    void *data = tpu_global_mem_addr(addr);
    u32 *ptr = (u32 *)(data);
    for(size_t i=0; i<num_elem; i++){
        ptr[i] = i+1;
    }
    tpu_flush_cache(addr, ALIGN(num_elem*elem_size, tpu_cache_line_size()));
}

static inline void check_data(u64 addr, u64 num_elem, int is_inversed){
    const int elem_size=4;
    tpu_invalidate_cache(addr, ALIGN(num_elem*elem_size, tpu_cache_line_size()));
    void *data = tpu_global_mem_addr(addr);
    u64 count=0;
    u32 *ptr = (u32 *)(data);
    for(size_t i=0; i<num_elem; i++){
        u32 ref = (is_inversed)?(num_elem-i):(i+1);
        if(ptr[i] != ref){
            count++;
            CORE_PRINT("  => addr=0x%llx, exp=0x%08x, got=0x%08x\n", addr+i*elem_size, ref, ptr[i]);
        }
    }
    if(count==0){
        CORE_PRINT("  check mem=[0x%llx,0x%llx) PASSED!\n", addr, addr+elem_size*num_elem);
    } else {
        CORE_PRINT("  check mem=[0x%llx,0x%llx) FAILED! elem_size=%d, mismatch=%d, total=%d\n",
                  addr, addr + elem_size * num_elem, (int)elem_size, (int)count, (int)num_elem);
    }
}
void nodechip_tpu_hau_mem_rw_test(unsigned char* api_buf){
    sg_api_pld_mem_rw_test_t *p = (sg_api_pld_mem_rw_test_t *)api_buf;
    TPUKERNEL_ASSERT(p->elem_dsize == 4);
    fill_data(p->start_addr, p->elem_num);
    u64 block_size = p->elem_dsize* p->elem_num;
    u64 stride_size = p->elem_stride* p->elem_dsize;
    u64 input_addr = 0;
    u64 output_addr = p->start_addr;
    data_type_t dtype = DT_INT32;
    int is_inversed = 0;
    (void)block_size;
    do {
        input_addr = output_addr;
        output_addr += stride_size;
        is_inversed = !is_inversed;
        CORE_PRINT("READ mem=[0x%llx, 0x%llx), WRITE mem=[0x%llx, 0x%llx) descended=%d\n",
                  input_addr, input_addr + block_size,
                  output_addr, output_addr + block_size,
                  is_inversed);
        tpu_hau_sort(
            output_addr,
            input_addr,
            p->elem_num,
            p->elem_num,
            is_inversed,
            dtype
        );
        tpu_hau_poll();
        check_data(output_addr, p->elem_num, is_inversed);
    } while(output_addr+stride_size<p->end_addr);
}
