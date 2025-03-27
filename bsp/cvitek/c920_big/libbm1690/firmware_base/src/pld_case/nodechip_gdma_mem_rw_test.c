#include "firmware_timer.h"
#include "common.h"
#include "tpu_kernel.h"
#include <stdlib.h>
#include "nodechip_pld_test.h"

static inline void fill_data(u64 addr, u64 num_elem, int elem_size){
    void *data = tpu_global_mem_addr(addr);
    if(elem_size==1){
        u8 *ptr = (u8 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = i+1;
        }
    } else if(elem_size==2){
        u16 *ptr = (u16 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = i+1;
        }
    } else if(elem_size==4){
        u32 *ptr = (u32 *)(data);
        for(size_t i=0; i<num_elem; i++){
            ptr[i] = i+1;
        }
    }
    tpu_flush_cache(addr, ALIGN(num_elem*elem_size, tpu_cache_line_size()));
}

static inline void check_data(u64 addr, u64 num_elem, int elem_size){
    tpu_invalidate_cache(addr, ALIGN(num_elem*elem_size, tpu_cache_line_size()));
    void *data = tpu_global_mem_addr(addr);
    u64 count=0;
    if(elem_size==1){
        u8 *ptr = (u8 *)(data);
        for(size_t i=0; i<num_elem; i++){
            u8 ref=i+1;
            if(ptr[i] != ref){
                count++;
                CORE_PRINT("mismatch => addr=0x%llx, exp=0x%02x, got=0x%02x\n", addr+i, ref, ptr[i]);
            }
        }
    } else if(elem_size==2){
        u16 *ptr = (u16 *)(data);
        for(size_t i=0; i<num_elem; i++){
            u16 ref = i+1;
            if(ptr[i] != ref){
                count++;
                CORE_PRINT("  => addr=0x%llx, exp=0x%04x, got=0x%04x\n", addr+i*elem_size, ref, ptr[i]);
            }
        }
    } else if(elem_size==4){
        u32 *ptr = (u32 *)(data);
        for(size_t i=0; i<num_elem; i++){
            u32 ref = i+1;
            if(ptr[i] != ref){
                count++;
                CORE_PRINT("  => addr=0x%llx, exp=0x%08x, got=0x%08x\n", addr+i*elem_size, ref, ptr[i]);
            }
        }
    }
    if(count==0){
        CORE_PRINT("  check mem=[0x%llx,0x%llx) PASSED!\n", addr, addr+elem_size*num_elem);
    } else {
        CORE_PRINT("  check mem=[0x%llx,0x%llx) FAILED! elem_size=%d, mismatch=%d, total=%d\n",
                  addr, addr + elem_size * num_elem, (int)elem_size, (int)count, (int)num_elem);
    }
}
void nodechip_tpu_gdma_mem_rw_test(unsigned char* api_buf){
    sg_api_pld_mem_rw_test_t *p = (sg_api_pld_mem_rw_test_t *)api_buf;
    fill_data(p->start_addr, p->elem_num, p->elem_dsize);
    u64 block_size = p->elem_dsize* p->elem_num;
    u64 stride_size = p->elem_stride* p->elem_dsize;
    u64 input_addr = 0;
    u64 output_addr = p->start_addr;
    (void)block_size;
    data_type_t dtype = p->elem_dsize==1? DT_UINT8: (p->elem_dsize==2?DT_UINT16: DT_UINT32);
    do {
        input_addr = output_addr;
        output_addr += stride_size;
        CORE_PRINT("READ mem=[0x%llx, 0x%llx), WRITE mem=[0x%llx, 0x%llx)\n",
                  input_addr, input_addr + block_size,
                  output_addr, output_addr + block_size);
        tpu_gdma_system_cpy(
            output_addr,
            input_addr,
            p->elem_num,
            dtype
            );
        tpu_poll();
        check_data(output_addr, p->elem_num, p->elem_dsize);
    } while(output_addr+stride_size<p->end_addr);
}
