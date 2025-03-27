#include "atomic_gen_cmd.h"
#include "gen_cmd.h"
#include "bd_reg_def.h"

#ifdef USING_CMODEL
typedef struct {
  int n;
  int c;
  int h;
  int w;
} dim4_t;

#define SWAP(a, b, type)  \
  do {                    \
    type tmp = a;         \
    a = b;                \
    b = tmp;              \
  } while (0)

void assert_diff_bank(u32* addr, dim4_t* shape, int* align, PREC* prec, int num) {
  u32* start_addr = (u32*)malloc(num * sizeof(int));
  int* index = (int*)malloc(num * sizeof(int));
  for (int i = 0; i < num; i++) {
    start_addr[i] = addr[i] - get_npu_index(addr[i]) * LOCAL_MEM_SIZE;
    index[i] = i;
  }
  for (int i = 0; i < num; i++) {
    for (int j = i + 1; j < num; j++) {
      if (start_addr[j] < start_addr[i]) {
        SWAP(start_addr[i], start_addr[j], u32);
        SWAP(index[i], index[j], int);
      }
    }
  }

  int* tensor_size = (int*)malloc(num * sizeof(int));
  for (int i = 0; i < num; i++) {
    int k = index[i];
    int cstride = get_local_cstride(shape[k].h, shape[k].w, align[k], prec[k]);
    int nstride = get_local_nstride(cstride, shape[k].c, addr[k]);
    tensor_size[i] = shape[k].n * nstride * get_bytesize(prec[k]);
  }

  for (int i = 0; i < num; i++) {
    int begin = start_addr[i] / LOCAL_BANK_SIZE;
    int end = (start_addr[i] + tensor_size[i] - get_bytesize(prec[index[i]]))
              / LOCAL_BANK_SIZE;
    if (i+1 < num) {
      int next = start_addr[i+1] / LOCAL_BANK_SIZE;
      ASSERT_FS_INFO(begin != next && end != next,
                     "bank confilict: addr: %u vs. %u, "
                     "%d vs. %d, %d vs. %d\n",
                     start_addr[i], start_addr[i+1],
                     begin, next, end, next);
    }
  }
  free(start_addr);
  free(index);
  free(tensor_size);
}
#endif

/* PL_gather_d1coor :A=[N,C,1,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d1coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,1,Wr]
 * A and R is aligned in local memory, A_start_npu == R_start_npu
 * B is compacted in local mem, support U8 and U16,
 * but aligned to 16 byte, B_start_npu = 0
 */
void atomic_pl_sgd1_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr: %x, B_addr: %x, R_addr %u, A_w: %d, R_n: %d, "
         "R_c: %d, R_w: %d, if_fill_const: %d, fill_const_val: %u, "
         "limit_en: %d B_prec: %d, R_prec: %d, op: %d\n",
         __func__, tensorA_addr, tensorB_addr, tensorR_addr, tensorA_w, tensorR_n,
         tensorR_c, tensorR_w, if_fill_const, fill_const_val, limit_enable,
         B_prec, R_prec, op);

  int tensorB_c = op == PL_gather_d1coor ? tensorR_w : tensorA_w;
#ifdef USING_CMODEL
  ASSERT(op == PL_gather_d1coor || op == PL_scatter_d1coor);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorB_addr) == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(R_prec != INT4); //only 4bit not support
  ASSERT(B_prec == INT8 || B_prec == INT16);
  ASSERT(if_fill_const == 0 || if_fill_const == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PL_scatter_d1coor) {
    ASSERT(if_fill_const == 0);
  }
#endif

  u32 opd2_addr = 0;
  if (if_fill_const) opd2_addr = fill_const_val;
  SG_GET_PROFILE(tensorR_n, tensorR_c, op == PL_gather_d1coor ? tensorR_w : tensorA_w, 0, tensorR_addr, op, R_prec, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)R_prec << 8) |
             ((u64)R_prec << 11) |
             ((u64)B_prec << 14) |
             ((u64)if_fill_const << 22) |
             ((u64)limit_enable << 62);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorR_n) |
            ((u64)tensorR_c << 16) |
            ((u64)1 << 32) |
            ((u64)tensorR_w << 48);
      high[2] = ((u64)1 << 32) | ((u64)tensorA_w << 48);
      low[3] = ((u64)tensorB_c << 16) | ((u64)1 << 48);
      high[4] = ((u64)tensorR_addr) | ((u64)tensorA_addr << 32);
      low[5] = ((u64)tensorB_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)limit_enable << 50) |
            ((u64)bd_power_step() << 59) |
            ((u64)if_fill_const << 63);
      high[0] = ((u64)tensorR_n) |
             ((u64)tensorR_c << 16) |
             (1ull << 32) |
             ((u64)tensorR_w << 48);
      low[1] = (1ull) |
            ((u64)tensorA_w << 16) |
            ((u64)tensorB_c << 32) |
            (1ull << 48);
      high[1] = ((u64)R_prec) |
             ((u64)(tensorR_addr & 0x1fffffff) << 3) |
             ((u64)R_prec << 32) |
             ((u64)(tensorA_addr & 0x1fffffff) << 35);
      low[2] = ((u64)B_prec) |
            ((u64)(tensorB_addr & 0x1fffffff) << 3) |
            ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}

/* PL_gather_d2coor :A=[N,C,Ha,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d2coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,Hr,Wr]
 * A and R is aligned in local memory, A_start_npu == R_start_npu
 * B is compacted in local mem, but aligned to 16 byte
 * opd1 is uint16, but storaged as INT32 with [h, w], B_start_npu = 0
 */
void atomic_pl_sgd2_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC R_prec,
  SG_OP op,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr: %x, B_addr: %x, R_addr %u, A_H: %d, A_w: %d,"
         "R_n: %d, R_c: %d, R_h: %d, R_w: %d, if_fill_const: %d, "
         "fill_const_val: %u, limit_en: %d R_prec: %d, op: %d\n",
         __func__, tensorA_addr, tensorB_addr, tensorR_addr, tensorA_h, tensorA_w, tensorR_n,
         tensorR_c, tensorR_h, tensorR_w, if_fill_const, fill_const_val, limit_enable, R_prec, op);

  int tensorB_c = op == PL_gather_d2coor ? tensorR_w : tensorA_w;
#ifdef USING_CMODEL
  ASSERT(op == PL_gather_d2coor || op == PL_scatter_d2coor);
  if(op == PL_gather_d2coor) {
    ASSERT(tensorR_h == 1);
  } else {
    ASSERT(tensorA_h == 1);
  }
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorB_addr) == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(R_prec != INT4); //only 4bit not support
  ASSERT(if_fill_const == 0 || if_fill_const == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_h < (1 << 16) && tensorR_h > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_h < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PL_scatter_d2coor) {
    ASSERT(if_fill_const == 0);
  }
#endif

  u32 opd2_addr = 0;
  if (if_fill_const) opd2_addr = fill_const_val;
  SG_GET_PROFILE(tensorR_n, tensorR_c, op == PL_gather_d2coor ? tensorR_w : tensorA_w, 0, tensorR_addr, op, R_prec, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)R_prec << 8) |
             ((u64)R_prec << 11) |
             ((u64)INT16 << 14) |
             ((u64)if_fill_const << 22) |
             ((u64)limit_enable << 62);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorR_n) |
            ((u64)tensorR_c << 16) |
            ((u64)tensorR_h << 32) |
            ((u64)tensorR_w << 48);
      high[2] = ((u64)tensorA_h << 32) | ((u64)tensorA_w << 48);
      low[3] = ((u64)tensorB_c << 16) | ((u64)1 << 48);
      high[4] = ((u64)tensorR_addr) | ((u64)tensorA_addr << 32);
      low[5] = ((u64)tensorB_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)limit_enable << 50) |
            ((u64)bd_power_step() << 59) |
            ((u64)if_fill_const << 63);
      high[0] = ((u64)tensorR_n) |
             ((u64)tensorR_c << 16) |
             ((u64)tensorR_h << 32) |
             ((u64)tensorR_w << 48);
      low[1] = ((u64)tensorA_h) |
            ((u64)tensorA_w << 16) |
            ((u64)tensorB_c << 32) |
            (1ull << 48);
      high[1] = ((u64)R_prec) |
             ((u64)(tensorR_addr & 0x1fffffff) << 3) |
             ((u64)R_prec << 32) |
             ((u64)(tensorA_addr & 0x1fffffff) << 35);
      low[2] = ((u64)INT16) |
            ((u64)(tensorB_addr & 0x1fffffff) << 3) |
            ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}

/* PE_S_gather_d1coor: do not support bank confilict,
 * PE_S_gather_hzd: support bank confilict,
 * A=[1,C,1,A_w], B=[N,C,1,R_w],R=[N,C,1,R_w]
 * PE_S_scatter_d1coor: do not support bank confilict
 * PE_S_scatter_hzd: support bank confilict
 * A=[1,C,1,A_w], B=[N,C,1,A_w],R=[N,C,1,R_w]
 */
/* A is aligned in local memory, if A_cstride is 0
 * A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B is aligned in local memory, support U8 and U16,
 * R is aligned in local memory
 * all start_npu is same
 */
void atomic_pes_sg_d1hzd_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int A_cstride_is0,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr: %x, B_addr: %x, R_addr: %x, A_w: %d,"
         "R_n: %d, R_c: %d, R_w: %d, A_cstride_is0: %d, if_fill_const: %d,"
         "fill_const_val:%u, limit_enable:%d, B_prec:%d, R_prec:%d, op:%d\n",
         __func__, tensorA_addr, tensorB_addr, tensorR_addr, tensorA_w, tensorR_n,
         tensorR_c, tensorR_w, A_cstride_is0, if_fill_const, fill_const_val, limit_enable,
         B_prec, R_prec, op);

#ifdef USING_CMODEL
  ASSERT(B_prec == INT8 || B_prec == INT16);
  ASSERT(R_prec != INT4); //only 4bit not support
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
  ASSERT(if_fill_const < (1 << 1) && if_fill_const >= 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(op == PE_S_gather_d1coor || op == PE_S_gather_hzd ||
         op == PE_S_scatter_d1coor || op == PE_S_scatter_hzd);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  if (op == PE_S_scatter_d1coor || op == PE_S_scatter_hzd) {
    ASSERT(if_fill_const == 0);
  }
  if (op == PE_S_gather_d1coor || op == PE_S_scatter_d1coor) {
    int B_w = op == PE_S_gather_d1coor ? tensorR_w : tensorA_w;
    int A_npu = get_npu_index(tensorA_addr);
    u32 addr[3] = {tensorA_addr, tensorB_addr, tensorR_addr};
    dim4_t shape[3] = {{1, A_cstride_is0 ? (NPU_NUM - A_npu) : tensorR_c, 1, tensorA_w},
                       {tensorR_n, tensorR_c, 1, B_w},
                       {tensorR_n, tensorR_c, 1, tensorR_w}};
    PREC prec_list[3] = {R_prec, B_prec, R_prec};
    int align[3] = {1, 1, 1};
    assert_diff_bank(addr, shape, align, prec_list, 3);
  }
#endif

  u32 opd2_addr = 0;
  if (if_fill_const) opd2_addr = fill_const_val;
  int A_short_str = A_cstride_is0 ? 4 : 0;
  int B_w = tensorR_w;
  if (op == PE_S_scatter_d1coor || op == PE_S_scatter_hzd) {
    B_w = tensorA_w;
  }
  SG_GET_PROFILE(tensorR_n, tensorR_c, (op == PE_S_gather_d1coor || op == PE_S_gather_hzd) ? tensorR_w : tensorA_w,
                 0, tensorR_addr, op, R_prec, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)R_prec << 8) |
             ((u64)R_prec << 11) |
             ((u64)B_prec << 14) |
             ((u64)if_fill_const << 22) |
             ((u64)A_short_str << 26) |
             ((u64)limit_enable << 62);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorR_n) |
            ((u64)tensorR_c << 16) |
            ((u64)1 << 32) |
            ((u64)tensorR_w << 48);
      high[2] = ((u64)1 << 32) | ((u64)tensorA_w << 48);
      low[3] = ((u64)tensorR_c << 16) | ((u64)B_w << 48);
      high[4] = ((u64)tensorR_addr) | ((u64)tensorA_addr << 32);
      low[5] = ((u64)tensorB_addr) | ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)limit_enable << 50) |
            ((u64)A_short_str << 52) |
            ((u64)bd_power_step() << 59) |
            ((u64)if_fill_const << 63);
      high[0] = ((u64)tensorR_n) |
             ((u64)tensorR_c << 16) |
             ((u64)1 << 32) |
             ((u64)tensorR_w << 48);
      low[1] = ((u64)1) |
            ((u64)tensorA_w << 16) |
            ((u64)tensorR_c << 32) |
            ((u64)B_w << 48);
      high[1] = ((u64)R_prec) |
             ((u64)(tensorR_addr & 0x1fffffff) << 3) |
             ((u64)R_prec << 32) |
             ((u64)(tensorA_addr & 0x1fffffff) << 35);
      low[2] = ((u64)B_prec) |
            ((u64)(tensorB_addr & 0x1fffffff) << 3) |
            ((u64)opd2_addr << 32);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}

/* PE_S_mask_select: do not support bank confilict
 * PE_S_mask_selhzd: support bank confilict
 * A, B, R are aligned in local memory,
 * if A_cstride is 0, A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B support uint8/uint16/uint32
 * and mask_num is compacted in local mem which only support uint16
 * A=[1,C,1,A_w], B=[N, C, 1, A_w], R=[N,C,1,R_w], mask_num=[N,C,1,1]
 */
void atomic_pes_mask_sel_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_w,
  int tensorB_n,
  int tensorB_c,
  int A_cstride_is0,
  PREC B_prec,
  PREC R_prec,
  SG_OP op,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr: %x, B_addr: %x, R_addr: %x, mask_num_addr: %x,"
         "A_w: %d, B_n: %d, B_c: %d, A_cstride_is0: %d,"
         "B_prec: %d, R_prec: %d, op: %d\n",
         __func__, tensorA_addr, tensorB_addr, tensorR_addr, mask_num_addr,
         tensorA_w, tensorB_n, tensorB_c, A_cstride_is0,
         B_prec, R_prec, op);

#ifdef USING_CMODEL
  ASSERT(R_prec != INT4);  //only 4bit not support
  ASSERT(B_prec == INT8 || B_prec == INT16 || B_prec == INT32);
  ASSERT(op == PE_S_mask_select || op == PE_S_mask_selhzd);
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
  ASSERT(tensorB_n < (1 << 16) && tensorB_n > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(mask_num_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(mask_num_addr));
  if (op == PE_S_mask_select) {
    u32 addr[4] = {tensorA_addr, tensorB_addr, tensorR_addr, mask_num_addr};
    int A_npu = get_npu_index(tensorA_addr);
    dim4_t shape[4] = {{1, A_cstride_is0 ? (NPU_NUM - A_npu) : tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, 1}};
    PREC prec_list[4] = {R_prec, B_prec, R_prec, INT16};
    int align[4] = {1, 1, 1, 0};
    assert_diff_bank(addr, shape, align, prec_list, 4);
  }
#endif

  int A_short_str = A_cstride_is0 ? 4 : 0;
  SG_GET_PROFILE(tensorB_n, tensorB_c, tensorA_w, 0, tensorR_addr, op, R_prec, pid_node);
  const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)R_prec << 8) |
             ((u64)R_prec << 11) |
             ((u64)B_prec << 14) |
             ((u64)A_short_str << 26);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorB_n) |
            ((u64)tensorB_c << 16) |
            ((u64)1 << 32) |
            ((u64)tensorA_w << 48);
      high[2] = ((u64)1 << 32) | ((u64)tensorA_w << 48);
      low[3] = ((u64)tensorB_c << 16) | ((u64)tensorA_w << 48);
      high[4] = ((u64)tensorR_addr) | ((u64)tensorA_addr << 32);
      low[5] = ((u64)tensorB_addr);
      high[7] = ((u64)mask_num_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)A_short_str << 52) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)tensorB_n) |
             ((u64)tensorB_c << 16) |
             ((u64)1 << 32) |
             ((u64)tensorA_w << 48);
      low[1] = ((u64)1) |
            ((u64)tensorA_w << 16) |
            ((u64)tensorB_c << 32) |
            ((u64)tensorA_w << 48);
      high[1] = ((u64)R_prec) |
             ((u64)(tensorR_addr & 0x1fffffff) << 3) |
             ((u64)R_prec << 32) |
             ((u64)(tensorA_addr & 0x1fffffff) << 35);
      low[2] = ((u64)B_prec) |
            ((u64)(tensorB_addr & 0x1fffffff) << 3);
      high[2] = ((u64)mask_num_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}

/* PE_S_nonsero: do not support bank confilict
 * PE_S_nonzero_hzd: support bank confilict
 * A, R are aligned in local memory,
 * A support INT8/INT16/INT32, R support INT16/INT32
 * and mask_num is compacted in local mem which only support uint16
 * A=[N,C,1,W], R=[N,C,1,W],mask_num=[N,C,1,1]
 * all start_npu is same
 */
void atomic_pes_nonzero_gen_cmd(
  u32 tensorA_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_n,
  int tensorA_c,
  int tensorA_w,
  PREC A_prec,
  PREC R_prec,
  SG_OP op,
  int thread_id,
  CMD_ID_NODE* pid_node)
{
  FW_DBG("%s, A_addr: %x, B_addr:%x, mask_addr:%x,"
         "A_n:%d, A_c:%d, A_w:%d, A_prec:%d, R_prec:%d, op:%d\n",
         __func__, tensorA_addr, tensorR_addr, mask_num_addr, tensorA_n, tensorA_c,
         tensorA_w, A_prec, R_prec, op);

#ifdef USING_CMODEL
  ASSERT(A_prec == INT8 || A_prec == INT16 || A_prec == INT32);
  ASSERT(R_prec == INT16 || R_prec == INT32);
  ASSERT(op == PE_S_nonzero_hzd || op == PE_S_nonzero);
  ASSERT(tensorA_n < (1 << 16) && tensorA_n > 0);
  ASSERT(tensorA_c < (1 << 16) && tensorA_c > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(mask_num_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(mask_num_addr));
  if (op == PE_S_nonzero) {
    u32 addr[3] = {tensorA_addr, tensorR_addr, mask_num_addr};
    dim4_t shape[3] = {{tensorA_n, tensorA_c, 1, tensorA_w},
                       {tensorA_n, tensorA_c, 1, tensorA_w},
                       {tensorA_n, tensorA_c, 1, 1}};
    PREC prec_list[3] = {A_prec, R_prec, INT16};
    int align[3] = {1, 1, 0};
    assert_diff_bank(addr, shape, align, prec_list, 3);
  }
#endif
 SG_GET_PROFILE(tensorA_n, tensorA_c, tensorA_w, 0, tensorR_addr, op, R_prec, pid_node);
 const volatile u64 reg_addr = BDC_CMD_BASE_ADDR;
#ifndef FAST_GEN_CMD
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 8;
      u64 low[8] = {0}, high[8] = {0};
      low[0] = (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)R_prec << 8) |
             ((u64)A_prec << 11);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorA_n) |
            ((u64)tensorA_c << 16) |
            ((u64)1 << 32) |
            ((u64)tensorA_w << 48);
      high[2] = ((u64)1 << 32) | ((u64)tensorA_w << 48);
      high[4] = ((u64)tensorR_addr) | ((u64)tensorA_addr << 32);
      high[7] = ((u64)mask_num_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#else
  BEGIN_FAST_GEN_CMD_BD(thread_id)
      int elt = 3;
      u64 low[3] = {0}, high[3] = {0};
      low[0] = (1ull) |
            (((u64)pid_node->gdma_cmd_id & 0xfffff ) << 17) |
            ((u64)1ull << 37) |
            ((u64)SG << 41) |
            ((u64)op << 45) |
            ((u64)bd_power_step() << 59);
      high[0] = ((u64)tensorA_n) |
             ((u64)tensorA_c << 16) |
             ((u64)1 << 32) |
             ((u64)tensorA_w << 48);
      low[1] = ((u64)1) |
            ((u64)tensorA_w << 16);
      high[1] = ((u64)R_prec) |
             ((u64)(tensorR_addr & 0x1fffffff) << 3) |
             ((u64)A_prec << 32) |
             ((u64)(tensorA_addr & 0x1fffffff) << 35);
      high[2] = ((u64)mask_num_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}
