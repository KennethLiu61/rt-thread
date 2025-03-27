#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "common.h"
#include "atomic_gdma_gen_cmd.h"
#include "gdma_reg_value.h"
#include "tpu_kernel.h"
#include <stdlib.h>

void nodechip_gdma_general_test(
    unsigned long long input_addr,
    unsigned long long output_addr)
{
#define BEGIN() start_time = firmware_timer_get_time_us(); \
                for (int t = 0; t < 10; ++t) {
#define END(info, prec) \
    }                   \
    poll_all_engine_done(&pid_node);         \
    end_time = firmware_timer_get_time_us();    \
    printf("test 10 times %s gdma general %s \n", #info, prec);   \
    printf("Total %s time: %lldus\n\n", #info, (end_time - start_time))

    CMD_ID_NODE pid_node;
    resync_cmd_id(&pid_node);
    unsigned long long start_time = 0, end_time = 0;
    (void)start_time;
    (void)end_time;

    int lens[] = {256, 4096, 8192};

    data_type_t data_type[] = {DT_FP32, DT_FP16, DT_INT8};
    char *type[] = {"FP32", "FP16", "INT8"};
    for (int i = 0; i < 3; ++i) {
        int len = lens[i];
        printf("\nlen=%d\n", len);
        for (int j = 0; j < 3; ++j) {
            data_type_t dtype = data_type[j];
            char *perc = type[j];
            BEGIN()
            general_gdma_gen_cmd(
                input_addr,
                tpu_l2_sram_get_start_addr(),
                ((dtype) >> 1),
                len,
                false,
                MASTER_THREAD,
                &pid_node
            );
            END(S2L2, perc);
            BEGIN()
            general_gdma_gen_cmd(
                tpu_l2_sram_get_start_addr(),
                output_addr,
                ((dtype) >> 1),
                len,
                false,
                MASTER_THREAD,
                &pid_node
            );
            END(L22S, perc);
            BEGIN()
            general_gdma_gen_cmd(
                input_addr,
                STATIC_MEM_START_ADDR + SMEM_STATIC_END_OFFSET,
                ((dtype) >> 1),
                len,
                false,
                MASTER_THREAD,
                &pid_node
            );
            END(S2SM, perc);
            BEGIN()
            general_gdma_gen_cmd(
                STATIC_MEM_START_ADDR + SMEM_STATIC_END_OFFSET,
                output_addr,
                ((dtype) >> 1),
                len,
                false,
                MASTER_THREAD,
                &pid_node
            );
            END(SM2S, perc);
        }
    }
#undef BEGIN
#undef END
}
