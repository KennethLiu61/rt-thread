#include "common.h"
#include "tpu_defs.h"

void __print_trace() {
   _print_trace();
}

THREAD bool check_id_node = true;
THREAD CMD_ID_NODE id_node;
THREAD CMD_ID_NODE bdc_id_node;
THREAD CMD_ID_NODE gdma_id_node;
int rsqrt_iter_num = 3;
int div_iter_num = 3;
int sfu_taylor_sin_len = 16;
int sfu_taylor_cos_len = 16;
int sfu_taylor_tan_len = 16;
int sfu_taylor_arcsin_len = 16;
