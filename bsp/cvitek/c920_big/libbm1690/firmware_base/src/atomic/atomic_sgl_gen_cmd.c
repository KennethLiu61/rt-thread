#include "atomic_gen_cmd.h"
#include "bd_reg_def.h"
#include "gen_cmd.h"

/* ha = tensorB(n, c, h, 0), and before call this func,
 * do if (tensorB(n, c, h, 0) >= tensorA_h
 *      tensorB(n,c,h,0) = tensorA_h - 1
 * if (ha != 0xff/0xffff || !limit_enable)
 *   tensorR(n,c,h,w) = tensorA(0,c,ha,w)
 * else if (if_fill_const)
 *   tensorR(n,c,h,w) = fill_const_val
 * else
 *   tensorR(n,c,h,w) = tensorR(n,c,h,w)
 */
/* Note:tensorA is aligned in local memory, and its stride is
 * wstride=1,hstride=ceil(w,EU_NUM) * EU_NUM,
 * cstride = A_cstride_is0 ? 0 : h*hstride,
 * nstride = c_per_npu * cstride.
 * And tensorR stride is wstride=1, hstride=ceil(w,EU_NUM) * EU_NUM
 * cstride=h*hstride, nstride=c_per_npu * cstride.
 * And tensorB is aligned in local memory with normal storage.
 * if (PE_S_gather_line) A=[1, C, A_h, A_w], B=[N, C, R_h, 1], R=[N, C, R_h, A_w]
 * else A=[1, C, A_h, A_w], B=[N, C, A_h, 1], R=[N, C, R_h, A_w]
 */
void atomic_sgl_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
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
  FW_DBG("%s,  A_addr: %u,  B_addr: %u, R_addr: %u,  A_h:%d,  R_n: %d,  "
         "R_c: %d, R_h: %d, R_w: %d, A_cstride_is0: %d, if_fill_const: %d"
         "fill_const_val: %u, limit_enable: %d, B_prec: %d, R_prec:%d, op: %d\n",
         __func__, tensorA_addr, tensorB_addr, tensorR_addr, tensorA_h, tensorR_n,
         tensorR_c, tensorR_h, tensorR_w, A_cstride_is0, if_fill_const, fill_const_val,
         limit_enable, B_prec, R_prec, op);

#ifdef USING_CMODEL
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(op == PE_S_scatter_line || op == PE_S_gather_line);
  ASSERT(R_prec != INT4); //only 4bit not support
  ASSERT(B_prec == INT8 || B_prec == INT16);
  ASSERT(if_fill_const < (1 << 1) && if_fill_const >= 0);
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_h < (1 << 16) && tensorR_h > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_h < (1 << 16) && tensorA_h > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PE_S_scatter_line) {
    ASSERT(if_fill_const == 0);
  }
#endif

  u32 opd2_addr = 0;
  if (if_fill_const) opd2_addr = fill_const_val;
  int opd0_short_str = A_cstride_is0 ? 4 : 3;
  SGL_GET_PROFILE(tensorR_n, tensorR_c, tensorR_w, op == PE_S_gather_line ? tensorR_h : tensorA_h, tensorA_addr, op, R_prec, pid_node);
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
             ((u64)B_prec << 14) |
             ((u64)if_fill_const << 22) |
             ((u64)opd0_short_str << 26) |
             ((u64)limit_enable << 62);
      high[1] = bd_get_lane_mask();
      low[2] = ((u64)tensorR_n) |
            ((u64)tensorR_c << 16) |
            ((u64)tensorR_h << 32) |
            ((u64)tensorR_w << 48);
      high[2] = ((u64)tensorA_h << 32);
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
            ((u64)limit_enable << 53) |
            ((u64)bd_power_step() << 59) |
            ((u64)if_fill_const << 63);
      high[0] = ((u64)R_prec) |
             ((u64)B_prec << 3) |
             ((u64)opd0_short_str << 6) |
             ((u64)tensorR_n << 16) |
             ((u64)tensorR_c << 32) |
             ((u64)tensorR_h << 48);
      low[1] = ((u64)tensorR_w) |
            ((u64)tensorA_h << 16) |
            ((u64)tensorR_addr << 32);
      high[1] = ((u64)tensorA_addr) |
             ((u64)tensorB_addr << 32);
      low[2] = ((u64)opd2_addr);
      for (int i = 0; i < elt; ++i) {
        WRITE_CMD_EX(reg_addr, i, high[i], low[i]);
      }
  END_FAST_GEN_CMD_BD(pid_node)
#endif
  profile_time_set_node(ENGINE_BD, SG,
      op, R_prec, pid_node, high, low, elt);
}
