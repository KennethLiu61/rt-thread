#ifndef ATOMIC_GEN_CMD_H
#define ATOMIC_GEN_CMD_H
#include "firmware_common.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline u64 bd_get_lane_mask() {
  u64 lane_mask = 0xffffffffffffffffull;
#if defined(USING_CMODEL) && defined(SG_TV_GEN)
  char *en_lane_mask = getenv("TV_GEN_EN_LANE_MASK");
  char *p = getenv("TV_GEN_LOG_PATH");
  char path_int[1024 * 2] = {'\0'};
  if (p) {
    strcpy(path_int, p);
  } else {
    strcpy(path_int, "");
  }
  strcat(path_int, "lane_mask_param");

  if (en_lane_mask && access(path_int, F_OK) == 0 && atoi(en_lane_mask) == 1) {
    FILE *file = fopen(path_int, "r");
    fscanf(file, "%llx\n", &lane_mask);
    fclose(file);
    printf("lane_mask:%llx\n", lane_mask);
  } else if (en_lane_mask && atoi(en_lane_mask) == 1) {
    lane_mask = 0;
    for (int i = 0; i < 64; i++) {
      lane_mask |= (rand() % 3 ? 1ull : 0ull) << i;
    }
    if (lane_mask == 0) {
      lane_mask = 1ull << (rand() % NPU_NUM);
    }

    FILE *file = fopen(path_int, "w");
    fprintf(file, "%llx\n", lane_mask);
    fclose(file);
  }
#endif
  return lane_mask;
}

static inline int bd_power_step() {
#ifdef ENABLE_POWER_CTRL
  return 0xf;
#else
  return 0x0;
#endif
}
static inline u64 gdma_get_cache_en() {
  u64 cache_en = 0;
#if defined(USING_CMODEL) && defined(SG_TV_GEN)
  char *en_lane_mask = getenv("TV_GEN_GDMA_ENABLE_CACHE");
  if (en_lane_mask && atoi(en_lane_mask) == 1) {
    cache_en |= 0b11;
  }
#endif
  return cache_en;
}

void atomic_conv_gen_cmd(
    u32 input_addr,
    u32 weight_addr,
    u32 bias_addr,
    u32 pad_ins_addr,
    u32 rescale_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int *input_stride,
    int do_relu,
    PREC input_prec,
    PREC output_prec,
    PREC bias_prec,
    int input_sign,
    int weight_sign,
    int res_sign,
    int bias_sign,
    int do_rescale,
    int rescale_is_const,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE * pid_node);

void atomic_conv_quant_gen_cmd(
    u32 input_addr,
    u32 weight_addr, // or weight const value
    u32 bias_addr, // or bias const value
    u32 pad_ins_addr, // pad const value
    u32 kzp_addr, // kzp const value
    u32 requant_addr, // multipiler const value
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kzp_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int res_sign,
    int *input_stride,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC weight_prec,
    PREC output_prec,
    PAD_MODE pad_mode,
    int thread_id,
    CMD_ID_NODE * pid_node);

void atomic_fused_linear_gen_cmd(
    u32   A_addr,
    u32   B_addr,
    u32   C_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    int   B_is_const,
    int   C_is_const,
    PREC input_prec,
    PREC output_prec,
    FP8_TYPE fp8_type,
    LIN_OP op,
    int   thread_id,
    CMD_ID_NODE * pid_node);

void atomic_fused_cmp_gen_cmd(
    u32 tensorA_addr,
    u32 tensorB_addr,
    u32 tensorC_addr,
    u32 tensorD_addr,
    u32 tensorR0_addr,
    u32 tensorR1_addr,
    int N,
    int C,
    int H,
    int W,
    int A_is_constant,
    int B_is_constant,
    int C_is_constant,
    int D_is_constant,
    int sign,
    int side,
    int bin_w,
    int A_short_str, //normal(0:align, 3:tensor)
    int B_short_str, //normal(0:align, 3:tensor)
    PREC AB_prec,
    PREC CD_prec,
    PREC RES0_dtype,
    CMP_OP op,
    int thread_id,
    CMD_ID_NODE* pid_node);

void atomic_mm_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int is_bias_const,
  int add_result,
  int do_relu,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_mm_fixed_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int L_sign,
  int R_sign,
  int bias_sign,
  int Res_sign,
  int is_bias_const,
  int add_result,
  int if_relu,
  int sym_range,
  int do_rq,
  s32 multiplier,
  s8 shift,
  s16 yzp,
  PREC Y_prec,
  PREC LR_prec,
  ROUND_MODE round_mode,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_mm2_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 Bias_addr,
  u32 RQ_addr,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_bias_const,
  int add_result,
  int do_relu,
  int do_rq,
  int is_rq_const,
  PREC LR_prec,
  PREC B_prec,
  PREC Y_prec,
  FP8_TYPE L_fp8_type,
  FP8_TYPE R_fp8_type,
  FP8_TYPE Y_fp8_type,
  int thread_id,
  int tf32_mode,
  CMD_ID_NODE* pid_node);

/* if (!L_trans) zp=[1,NPU_NUM,1,R_col_num],and its stride is[0,0,W,1]
 * if (L_trans && R_trans) zp=[1,R_col_num,1,1], and it is compacted in local memory
 */
void atomic_mm2_fixed_gen_cmd(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 rzp_addr,
  u32 Bias_addr,
  u32 RQ_addr,
  s8 shift_val,
  s16 yzp_val,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_zp_const,
  int L_sign,
  int R_sign,
  int add_result,
  int Res_sign,
  int Bias_sign,
  int is_bias_const,
  int is_rq_const,
  int do_relu,
  int sym_range,
  int do_rq,
  ROUND_MODE rshift_rd,
  PREC L_prec,
  PREC R_prec,
  PREC Y_prec,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_lane_broad_gen_cmd(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node);

void atomic_lane_copy_gen_cmd(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int C,
    int H,
    int W,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node);

void atomic_static_broad_gen_cmd(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node);

void atomic_static_distribute_gen_cmd(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    u64 lane_mask,
    PREC prec,
    int thread_id,
    CMD_ID_NODE* pid_node);

void atomic_cw_transpose_gen_cmd(
    u32   A_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    PREC dtype,
    TRAN_OP op,
    int   thread_id,
    CMD_ID_NODE * pid_node
);

void atomic_sort_gen_cmd(
    u64 src_data_addr,
    u64 src_idx_addr,
    u64 dst_data_addr,
    u64 dst_idx_addr,
    int data_type,   // 0:fp32 1:int32 2:uint32
    int row_num,
    int len,
    int is_descend,
    int idx_enable,
    int idx_auto,
    int topk,
    CMD_ID_NODE *pid_node);

void atomic_vector_correlation_gen_cmd(
        u32 A_addr,
        u32 B_addr,
        u32 R_addr,
        int A_len,
        int B_len,
        int A_w,
        int B_w,
        AR_OP op,
        PREC A_prec,
        PREC B_prec,
        PREC R_prec,
        ROUND_MODE round_mode,
        u32 select_val,
        int A_sign,
        int B_sign,
        int R_sign,
        int thread_id,
        CMD_ID_NODE* pid_node);

// scatter_gather_line
/* Note:tensorA is aligned in local memory, and its stride is
 * wstride=1,hstride=ceil(w,EU_NUM) * EU_NUM,
 * cstride = A_cstride_is0 ? 0 : h*hstride,
 * nstride = c_per_npu * cstride.
 * And tensorR stride is wstride=1, hstride=ceil(w,EU_NUM) * EU_NUM
 * cstride=h*hstride, nstride=c_per_npu * cstride.
 * And tensorB is aligned in local memory with normal storage.
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
  CMD_ID_NODE* pid_node);

/* PL_gather_d1coor :A=[N,C,1,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d1coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,1,Wr]
 * A and R is aligned in local memory, B is compacted in local mem
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
  CMD_ID_NODE* pid_node);

/* PL_gather_d2coor :A=[N,C,Ha,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d2coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,Hr,Wr]
 * A and R is aligned in local memory, B is compacted in local mem
 * opd1 is uint16, but storaged as INT32 with [h, w]
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
  CMD_ID_NODE* pid_node);

/* PE_S_gather_d1coor
 * PE_S_gather_hzd
 * PE_S_scatter_d1coor
 * PE_S_scatter_hzd
 */
/* A is aligned in local memory, if A_cstride is 0
 * A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B and R is aligned in local memory
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
  CMD_ID_NODE* pid_node);

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
  CMD_ID_NODE* pid_node);

/* PE_S_nonsero: do not support bank confilict
 * PE_S_nonzero_hzd: support bank confilict
 * A, R are aligned in local memory,
 * A support INT8/INT16/INT32, R support INT16/INT32
 * and mask_num is compacted in local mem which only support uint16
 * A=[N,C,1,W], R=[N,C,1,W],mask_num=[N,C,1,1]
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
  CMD_ID_NODE* pid_node);

/* PE_M_gather_d1coor:do not support bank confilict
 * A=[4,C,1,A_w] per batch data is same,
 * and aligned in the [LOCAL_MEM_SIZE/2, LOCAL_MEM_SIZE]
 * A_cstride=ALIGN(A_w, EU_NUM),A_nstride=bank_size/(bitwidth/8)
 * B=[1,C,1,B_w] support INT8/INT16,
 * aligned in bank0 and bank1
 * R=[1,C,1,B_w], aligned in bank2 and bank3
 */
void atomic_pem_gather_d1_gen_cmd(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_c,
  int tensorA_w,
  int tensorB_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  int A_cstride_is0,
  PREC B_prec,
  PREC R_prec,
  CMD_ID_NODE* pid_node);

/*
 param  n : length of taylor table for TAYLOR or number of iterations of Newton's algorithm for RSQRT
*/
void atomic_sfu_gen_cmd(
    u32     A_addr,
    u32     Y_addr,
    int     input_n,
    int     input_c,
    int     input_h,
    int     input_w,
    int     n,
    SFU_OP  sfu_op,
    u32     table_start_addr,
    PREC    res0_prec,
    PREC    opd0_prec,
    int     thread_id,
    CMD_ID_NODE * pid_node);

void atomic_rq_f32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  float zp_value,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode,
  ROUND_MODE f2i_mode,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_rq_i32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  int scale_val,
  char shift_val, // negative: right shift, positive: left shift
  short zp_val,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_dq_f32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  short zp_value, // S8/U8/S16/U16
  int A_sign,
  int R_sign,
  PREC A_prec,
  ROUND_MODE i2f_mode,
  int thread_id,
  CMD_ID_NODE* pid_node);

void atomic_dq_i32mode_gen_cmd(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  short zp_value, // S8/U8/S16/U16
  int scale_factor, // S8/S16/S32
  char shift_num, // negative: right shift, positive: left shift
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd,
  int thread_id,
  CMD_ID_NODE* pid_node);
#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_GEN_CMD_H */
