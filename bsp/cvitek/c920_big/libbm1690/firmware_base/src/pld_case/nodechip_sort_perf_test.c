#include "nodechip_pld_test.h"
#include "firmware_timer.h"
#include "atomic_gen_cmd.h"

#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

void nodechip_sort_perf_test(u64 input_addr_glb) {
    u64 st, et;
    ////////////////////////////////////////////////
    // data in global memory
    ///////////////////////////////////////////////
    const int loop=10;
    printf("SORT PERF TEST (data in DDR)\n");
    CMD_ID_NODE id_node = {0};
    int glen[5] = {100, 1000, 10000, 100000, 1000000};
    int gtopk[4] = {1, 10, 100, 1000};
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            float expT = 0.9 * DIV_UP(gtopk[j], 128) + (glen[i] * DIV_UP(gtopk[j], 128) - 64 * DIV_UP(gtopk[j], 128) * DIV_UP(gtopk[j], 128)) / 8000.0;
            if (gtopk[j] > glen[i]) continue;
            for (int idx_en = 0; idx_en < 2; idx_en++) {
                expT *= (idx_en ? 2 : 1);
                for (int auto_idx = 0; auto_idx < 2; auto_idx++) {
                    if (idx_en == 0 && auto_idx == 1) continue;
                    for (int is_fp = 0; is_fp < 2; is_fp++) {
                        st = firmware_timer_get_time_us();
                        for(int l=0; l<loop; l++){
                            atomic_sort_gen_cmd(
                                    input_addr_glb,
                                    input_addr_glb + 1000000 * sizeof(float),
                                    input_addr_glb + 1000000 * sizeof(float) * 2,
                                    input_addr_glb + 1000000 * sizeof(float) * 3,
                                    !is_fp,
                                    glen[i],
                                    true,
                                    idx_en,
                                    auto_idx,
                                    gtopk[j],
                                    MASTER_THREAD,
                                    &id_node);
                        }
                        poll_hau_engine_done(&id_node);
                        et = firmware_timer_get_time_us();
                        printf("len=%d topk=%d idx_en=%d auto_idx=%d is_fp=%d time=%gus expTime=%.1fus\n",
                               glen[i], gtopk[j], idx_en, auto_idx, is_fp, 1.0*(et-st)/loop, expT);

                    }
                }
            }
        }
    }
}
