#ifdef SG_STAS_GEN
#define SET_GDMA_CMD_INFO(pid_node, N, C, H, W, dir, src_data_format, dst_data_format, setting)  \
    gdma_cmd_node_info_t *the_info = &pid_node->gdma_cmd_info;                                   \
    the_info->n = N;                                                                             \
    the_info->c = C;                                                                             \
    the_info->h = H;                                                                             \
    the_info->w = W;                                                                             \
    the_info->direction = dir;                                                                   \
    the_info->src_format = src_data_format;                                                      \
    the_info->dest_format = dst_data_format;                                                     \
    the_info->setted = setting
#endif

#ifdef SG_STAS_GEN
#define FUSED_LINEAR_GET_PROFILE(input_n, input_c, input_h, input_w, input_prec, output_prec, res_addr, pid_node) \
    if (pid_node != NULL)                                                                \
    {                                                                                    \
        pid_node->inst_profile = atomic_fused_linear_get_profile(                        \
            input_n, input_c, input_h, input_w, input_prec, output_prec, res_addr);                         \
        pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                           \
        strcpy(pid_node->cmd_name, "FUSED_LINEAR");                                      \
    }
#else
#define FUSED_LINEAR_GET_PROFILE(input_n, input_c, input_h, input_w, input_prec, output_prec, res_addr, pid_node)
#endif

#ifdef SG_STAS_GEN
#define FUSED_CMP_GET_PROFILE(res_n, res_c, res_h, res_w, prec, res_addr, pid_node) \
    if (pid_node != NULL)                                                                  \
    {                                                                                      \
        pid_node->inst_profile = atomic_fused_cmp_get_profile(                             \
            res_n, res_c, res_h, res_w, prec, res_addr);                                   \
            pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                         \
        strcpy(pid_node->cmd_name, "FUSED_CMP");                                           \
    }
#else
#define FUSED_CMP_GET_PROFILE(res_n, res_c, res_h, res_w, prec, res_addr, pid_node)
#endif

#ifdef SG_STAS_GEN
#define SFU_GET_PROFILE(input_n, input_c, input_h, input_w, sfu_op, prec, n, res_addr, pid_node) \
    if (pid_node != NULL)                                                                        \
    {                                                                                            \
        pid_node->inst_profile = atomic_sfu_get_profile(                                         \
            input_n, input_c, input_h, input_w, sfu_op, prec, n, res_addr);                      \
        pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                   \
        strcpy(pid_node->cmd_name, "SFU");                                                       \
    }
#else
#define SFU_GET_PROFILE(input_n, input_c, input_h, input_w, sfu_op, prec, n, res_addr, pid_node)
#endif

#ifdef SG_STAS_GEN
#define MM_GET_PROFILE(L_row_num, L_col_num, res_c, res_w, opd1_w, opd0_c, opd0_w,               \
                        prec,res_addr, have_bias, left_tran, add_result, pid_node)               \
  if (pid_node != NULL) {                                                                        \
    int res0_n, opd0_n ;                                                                         \
    if(left_tran){                                                                               \
        opd0_n = L_col_num;                                                                      \
        res0_n = opd0_w * (opd0_c - 1) + opd1_w;                                                  \
    } else {                                                                                     \
        opd0_n = L_row_num;                                                                      \
        res0_n = opd0_n;                                                                          \
    }                                                                                            \
    pid_node->inst_profile =                                                                     \
        atomic_mm_get_profile(res0_n, res_c, res_w, opd1_w, opd0_c, opd0_w, opd0_n,              \
                            prec, res_addr, have_bias, left_tran, add_result);                   \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "MM");                                                            \
  }
#else
#define MM_GET_PROFILE(L_row_num, L_col_num, res_c, res_w, opd1_w, opd0_c, opd0_w,               \
                            prec, res_addr, have_bias, left_tran, add_result, pid_node)
#endif

#ifdef SG_STAS_GEN
#define MM2_GET_PROFILE(L_row_num, L_col_num, R_col_num, prec, mm_op, add_result, pid_node)      \
  if (pid_node != NULL) {                                                                        \
     int res_c, res_w, opd1_c, opd1_w ;                                                          \
    if(mm_op == MM_TT || mm_op == MM_TT_TF32){                                                                          \
        res_c = R_col_num;                                                                       \
        res_w = L_row_num;                                                                       \
    } else {                                                                                     \
        res_c = L_row_num;                                                                       \
        res_w = R_col_num;                                                                       \
    }                                                                                            \
    if(mm_op == MM_NN || mm_op == MM_NN_TF32){                                                                          \
        opd1_c = L_col_num;                                                                      \
        opd1_w = R_col_num;                                                                      \
    } else {                                                                                     \
        opd1_c = R_col_num;                                                                      \
        opd1_w = L_col_num;                                                                      \
    }                                                                                            \
    pid_node->inst_profile =                                                                     \
        atomic_mm2_get_profile(res_c, res_w, opd1_c, opd1_w, mm_op, prec, add_result);           \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "MM2");                                                           \
  }
#else
#define MM2_GET_PROFILE(L_row_num, L_col_num, R_col_num, prec, mm_op, add_result, pid_node)
#endif

#ifdef SG_STAS_GEN
#define VEC_CORR_GET_PROFILE(res_n, res_c, res_h, res_w, res_addr, A_prec, B_prec, R_prec,       \
                             vec_corr_op, pid_node)                                              \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_vec_corr_get_profile(                                        \
        res_n, res_c, res_h, res_w, res_addr, A_prec, B_prec, R_prec, vec_corr_op);              \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "VEC_CORR");                                                      \
  }
#else
#define VEC_CORR_GET_PROFILE(res_n, res_c, res_h, res_w, res_addr, A_prec, B_prec, R_prec,       \
                             vec_corr_op, pid_node)
#endif

#ifdef SG_STAS_GEN
#define AR_GET_PROFILE(res_n, res_c, res_h, res_w, res_hstr, res_wstr, res_short_str,            \
                       opd0_hstr, opd0_wstr, opd0_short_str,                                     \
                       opd1_hstr, opd1_wstr, opd1_short_str,                                     \
                       opd0_addr, opd1_addr, res_addr,                                           \
                       opd0_prec, opd1_prec, res_prec, ar_op,                                    \
                       div_iter, opd0_is_const, opd1_is_const, pid_node)                         \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_ar_get_profile(                                              \
        res_n, res_c, res_h, res_w, res_hstr, res_wstr, res_short_str,                           \
        opd0_hstr, opd0_wstr, opd0_short_str,                                                    \
        opd1_hstr, opd1_wstr, opd1_short_str,                                                    \
        opd0_addr, opd1_addr, res_addr,                                                          \
        opd0_prec, opd1_prec, res_prec, div_iter, ar_op, opd0_is_const, opd1_is_const);          \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "AR");                                                            \
  }
#else
#define AR_GET_PROFILE(res_n, res_c, res_h, res_w, res_hstr, res_wstr, res_short_str,            \
                       opd0_hstr, opd0_wstr, opd0_short_str,                                     \
                       opd1_hstr, opd1_wstr, opd1_short_str,                                     \
                       opd0_addr, opd1_addr, res_addr,                                           \
                       opd0_prec, opd1_prec, res_prec, ar_op,                                    \
                       div_iter, opd0_is_const, opd1_is_const, pid_node)
#endif

#ifdef SG_STAS_GEN
#define PorD_GET_PROFILE(res_n, res_c, res_h, res_w, opd1_h, opd1_w,                             \
                         stride_h, stride_w, res_addr, prec,                                     \
                         op, pid_node)                                                           \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_pool_depthwise_get_profile(                                  \
        res_n, res_c, res_h, res_w, opd1_h, opd1_w, stride_h, stride_w, op, res_addr, prec);     \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "PorD");                                                          \
  }
#else
#define PorD_GET_PROFILE(res_n, res_c, res_h, res_w, opd1_h, opd1_w,                             \
                         stride_h, stride_w, res_addr, prec, op, pid_node)
#endif

#ifdef SG_STAS_GEN
#define RQDQ_GET_PROFILE(A_addr, B_addr, R_addr,                                                 \
                         N, C, H, W, B_is_const, pid_node)                                       \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_rqdq_get_profile(                                            \
            A_addr, B_addr, R_addr, N, C, H, W, B_is_const);                                     \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "RQDQ");                                                          \
  }
#else
#define RQDQ_GET_PROFILE(A_addr, B_addr, R_addr,                                                 \
                         N, C, H, W, B_is_const, pid_node)
#endif

#ifdef SG_STAS_GEN
#define CW_TRANSPOSE_GET_PROFILE(res_n, res_c, res_h, res_w, prec, tran_op, pid_node)            \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_cw_trans_get_profile(                                        \
        res_n, res_c, res_h, res_w, prec, tran_op);                                              \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "TRANS");                                                         \
  }
#else
#define CW_TRANSPOSE_GET_PROFILE(res_n, res_c, res_h, res_w, prec, tran_op, pid_node)
#endif

#ifdef SG_STAS_GEN
#define CONV_GET_PROFILE(res_n, res_c, res_h, res_w, opdo_c, opd1_h, opd1_w,                     \
                         prec, pid_node)                                                         \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_conv_get_profile(                                            \
        res_n, res_c, res_h, res_w, opdo_c, opd1_h, opd1_w, prec);                               \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "CONV");                                                          \
  }
#else
#define CONV_GET_PROFILE(res_n, res_c, res_h, res_w, opdo_c, opd1_h, opd1_w, prec, pid_node)
#endif

#ifdef SG_STAS_GEN
#define SG_GET_PROFILE(res_n, res_c, opd1_w, opd3_addr, res_addr, sg_op,                         \
                         prec, pid_node)                                                         \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_sg_get_profile(                                              \
        res_n, res_c, opd1_w, opd3_addr, sg_op, prec, res_addr);                                 \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "SG");                                                            \
  }
#else
#define SG_GET_PROFILE(res_n, res_c,  opd1_w, opd3_addr, res_addr, sg_op, prec, pid_node)
#endif

#ifdef SG_STAS_GEN
#define SGL_GET_PROFILE(res_n, res_c, res_w, opd1_h, res_addr, sg_op,                            \
                         prec, pid_node)                                                         \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_sgl_get_profile(                                             \
        res_n, res_c, res_w, opd1_h, sg_op, prec, res_addr);                                     \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    strcpy(pid_node->cmd_name, "SGL");                                                           \
  }
#else
#define SGL_GET_PROFILE(res_n, res_c, res_w, opd1_h,  res_addr, sg_op, prec, pid_node)
#endif

#ifdef SG_STAS_GEN
#define LANE_BC_GET_PROFILE(src_n, dst_c, src_h, src_w, prec, src_addr, dst_addr, pid_node) \
  if (pid_node != NULL) {                                           \
    pid_node->inst_profile = atomic_lane_bc_get_profile(            \
      src_n, 1, dst_c, src_h, src_w, prec, src_addr, dst_addr);     \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;          \
    strcpy(pid_node->cmd_name, "LANE_BC");                          \
  }
#else
#define LANE_BC_GET_PROFILE(src_n, dst_c, src_h, src_w, prec, src_addr, dst_addr, pid_node)
#endif

#ifdef SG_STAS_GEN
#define LANE_CPY_GET_PROFILE(src_n, src_c, src_h, src_w, prec, src_addr, dst_addr, pid_node)  \
  if (pid_node != NULL) {                                             \
    pid_node->inst_profile = atomic_lane_bc_get_profile(              \
      src_n, src_c, src_c, src_h, src_w, prec, src_addr, dst_addr);   \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;            \
    strcpy(pid_node->cmd_name, "LANE_COPY");                          \
  }
#else
#define LANE_CPY_GET_PROFILE(src_n, src_c, src_h, src_w, prec, src_addr, dst_addr, pid_node)
#endif

#ifdef SG_STAS_GEN
#define STATIC_BC_GET_PROFILE(dst_c, src_w, prec, pid_node) \
  if (pid_node != NULL) {                                   \
    pid_node->inst_profile = atomic_static_bc_get_profile(  \
      dst_c, src_w, prec);                                  \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;  \
    strcpy(pid_node->cmd_name, "STATIC_BC");                \
  }
#else
#define STATIC_BC_GET_PROFILE(dst_c, src_w, prec, pid_node)
#endif

#ifdef SG_STAS_GEN
#define DIS_BC_GET_PROFILE(dst_c, prec, pid_node)           \
  if (pid_node != NULL) {                                   \
    pid_node->inst_profile = atomic_dis_bc_get_profile(     \
      dst_c, prec);                                         \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;  \
    strcpy(pid_node->cmd_name, "DISTRIBUTE_BC");            \
  }
#else
#define DIS_BC_GET_PROFILE(dst_c, prec, pid_node)
#endif

#ifdef SG_STAS_GEN
#define GDMA_GET_PROFILE(src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h,    \
                         dst_w, g_addr, src_data_format, dst_data_format,      \
                         result_add, dir, transpose, is_cwtrans,               \
                         is_const_fill, src_wstride, dst_wstride,              \
                         src_hstride, dst_hstride, src_cstride, dst_cstride,   \
                         src_nstride, dst_nstride, stride_enable,              \
                         dma_type, pid_node)                                   \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_get_profile(                          \
        src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h, dst_w, g_addr,        \
        src_data_format, dst_data_format, result_add, dir, transpose,          \
        is_cwtrans, is_const_fill, src_wstride, dst_wstride,                   \
       src_hstride, dst_hstride, src_cstride, dst_cstride,                     \
        src_nstride, dst_nstride, stride_enable, dma_type);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    gdma_cmd_node_info_t *the_info = &pid_node->gdma_cmd_info;                 \
    the_info->n = src_n;                                                       \
    the_info->c = src_c;                                                       \
    the_info->h = src_h;                                                       \
    the_info->w = src_w;                                                       \
    the_info->direction = dir;                                                 \
    the_info->src_format = src_data_format;                                    \
    the_info->dest_format = dst_data_format;                                   \
    the_info->setted = true;                                                   \
    strcpy(pid_node->cmd_name, "GDMA_");                                       \
  }

#define GDMA_TENSOR_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,        \
      direction,  special_func, store_type, src_wstride, dst_wstride, src_hstride,        \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)   \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_tensor_get_profile(                          \
        src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,        \
      direction,  special_func, store_type, src_wstride, dst_wstride, src_hstride,        \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, src_n, src_c, src_h, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_TENSOR");                                       \
  }

#define GDMA_MATRIX_GET_PROFILE(row_num, col_num, sec_size, src_addr, dst_addr, data_format, direction, transpose, \
     global_row_stride, local_row_stride, local_sec_stride, stride_enable, pid_node)                       \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_matrix_get_profile(                          \
        row_num, col_num, sec_size, src_addr, dst_addr, data_format, direction, transpose, \
     global_row_stride, local_row_stride, local_sec_stride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, 1, 1, row_num, col_num, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_MATRIX");                                       \
  }

#define GDMA_CONSTANT_GET_PROFILE(dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, data_format,  \
        special_func, is_local_dst, dst_wstride,                                                 \
        dst_hstride, dst_cstride, dst_nstride, stride_enable, pid_node)                          \
  if (pid_node != NULL) {                                                                        \
    pid_node->inst_profile = atomic_gdma_constant_get_profile(                                   \
      dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, data_format,                              \
      special_func, is_local_dst,  dst_wstride,                                                  \
      dst_hstride, dst_cstride, dst_nstride, stride_enable);                                     \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                                       \
    SET_GDMA_CMD_INFO(pid_node, dst_n, dst_c, dst_h, dst_w, 0, data_format, data_format, false); \
    strcpy(pid_node->cmd_name, "GDMA_CONSTANT");                                                 \
  }

#define GDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, data_format, src_is_const, pid_node)  \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_general_get_profile(                          \
          src_count, src_addr, dst_addr, data_format, src_is_const);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, 1, 1, 1, src_count, GDMA_S2S, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_GENERAL");                                       \
  }

#define GDMA_CW_TRANS_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,         \
      direction, src_wstride, dst_wstride, src_hstride,                                                     \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)             \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_cw_trans_get_profile(                          \
    src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,                                  \
      direction, src_wstride, dst_wstride, src_hstride,                                                     \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, src_n, src_c, src_h, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_CW_TRANS");                                       \
  }

#define GDMA_MASKED_SEL_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , data_format,  direction, pid_node)     \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_masked_sel_get_profile(                          \
   src_n, src_c, src_h, src_w,  src_addr, dst_addr , data_format,  direction);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, src_n, src_c, src_h, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_MASKED_SEL");                                       \
  }

#define GDMA_NONZERO_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, direction, pid_node)   \
  if (pid_node != NULL) {                                                      \
    pid_node->inst_profile = atomic_gdma_nonzero_get_profile(                          \
  src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, direction);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, src_n, src_c, src_h, src_w, direction, src_data_format, dst_data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_NONZERO");                                       \
  }

#define GDMA_BROADCAST_GET_PROFILE(src_n, src_h, src_w, dst_c, src_addr, dst_addr, data_format, direction, src_wstride, dst_wstride, \
        src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)     \
  if (pid_node != NULL) {                                                                                      \
    pid_node->inst_profile = atomic_gdma_broadcast_get_profile(                          \
             src_n, src_h, src_w, dst_c, src_addr, dst_addr, data_format, direction, src_wstride, dst_wstride, \
             src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, src_n, 1, src_h, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_BROADCAST");                                       \
  }

#define GDMA_GATHER_GET_PROFILE(src_c, src_h, src_w, index_H, src_addr, dst_addr, data_format, \
      direction,  src_wstride, dst_wstride, src_hstride,                                           \
      dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)     \
  if (pid_node != NULL) {                                                                                      \
    pid_node->inst_profile = atomic_gdma_gather_get_profile(                          \
          src_c, src_h, src_w, index_H, src_addr, dst_addr, data_format,                                \
          direction,  src_wstride, dst_wstride, src_hstride,                                           \
      dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, 1, src_c, index_H, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_GATHER");                                       \
  }

#define GDMA_SCATTER_GET_PROFILE(src_c, src_h, src_w, dst_h, src_addr, dst_addr, data_format, \
        direction, src_C_is1, index_C_is1, src_wstride, dst_wstride, src_hstride, dst_hstride, \
        src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, inplace_add, pid_node)       \
  if (pid_node != NULL) {                                                                                      \
    pid_node->inst_profile = atomic_gdma_scatter_get_profile(src_c, src_h, src_w, dst_h, src_addr, dst_addr, data_format, \
        direction, src_C_is1, index_C_is1, src_wstride, dst_wstride, src_hstride, \
        dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable);                    \
    pid_node->cur_op_cycle = pid_node->inst_profile.cycle;                     \
    SET_GDMA_CMD_INFO(pid_node, 1, src_c, src_h, src_w, direction, data_format, data_format, true); \
    strcpy(pid_node->cmd_name, "GDMA_SCATTER");                                       \
  }

#define SDMA_TENSOR_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,        \
      special_func, store_type, src_wstride, dst_wstride, src_hstride,        \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)   \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_TENSOR");                                       \
  }

#define SDMA_CONSTANT_GET_PROFILE( dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, data_format,  \
        special_func, dst_wstride, dst_hstride,                                     \
        dst_cstride, dst_nstride, stride_enable, pid_node) \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_CONSTANT");                                       \
  }

#define SDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, data_format, src_is_const, pid_node) \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_GENERAL");                                       \
  }

#define SDMA_CW_TRANS_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,         \
      src_wstride, dst_wstride, src_hstride,                                                     \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)             \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_CW_TRANS");                                       \
  }

#define SDMA_FILTER_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , data_format,  pid_node)     \
  // TODO

#define SDMA_NONZERO_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, pid_node) \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_NONZERO");                                       \
  }

#define SDMA_GATHER_GET_PROFILE(src_c, src_h, src_w, index_H, src_addr, dst_addr, data_format, \
      src_wstride, dst_wstride, src_hstride,                                           \
      dst_hstride, src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node) \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_GAGHER");                                       \
  }

#define SDMA_SCATTER_GET_PROFILE(src_c, src_h, src_w, dst_h, src_addr, dst_addr, data_format, \
        src_C_is1, index_C_is1, src_wstride, dst_wstride, src_hstride, dst_hstride, \
        src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, inplace_add, pid_node) \
  if (pid_node != NULL) {                                                      \
    strcpy(pid_node->cmd_name, "SDMA_SCATTER");                                       \
  }

#else
#define GDMA_GET_PROFILE(src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h,      \
                         dst_w, g_addr, src_data_format, dst_data_format,      \
                         result_add, direction, transpose, is_cwtrans,         \
                         is_const_fill, src_wstride, dst_wstride,              \
                         src_hstride, dst_hstride, src_cstride, dst_cstride,   \
                         src_nstride, dst_nstride, stride_enable, dma_tpye, pid_node)

#define GDMA_TENSOR_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,  \
      direction,  special_func, store_type, src_wstride, dst_wstride, src_hstride,  \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define GDMA_MATRIX_GET_PROFILE(row_num, col_num, sec_size, src_addr, dst_addr, data_format, direction, transpose, \
     global_row_stride, local_row_stride, local_sec_stride, stride_enable, pid_node)

#define GDMA_CONSTANT_GET_PROFILE( dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, data_format,  \
        special_func, is_local_dst, dst_wstride, dst_hstride,                                     \
        dst_cstride, dst_nstride, stride_enable, pid_node)

#define GDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, data_format, src_is_const, pid_node)

#define GDMA_CW_TRANS_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,         \
      direction, src_wstride, dst_wstride, src_hstride,                                                     \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define GDMA_MASKED_SEL_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , data_format,  direction, pid_node)

#define GDMA_NONZERO_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, direction, pid_node)

#define GDMA_BROADCAST_GET_PROFILE(src_n, src_h, src_w, dst_c, src_addr, dst_addr, data_format, direction, src_wstride, dst_wstride, \
        src_hstride, dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define GDMA_GATHER_GET_PROFILE(src_c, src_h, src_w, index_H, src_addr, dst_addr, data_format, \
      direction,  src_wstride, dst_wstride, src_hstride,                                           \
      dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define GDMA_SCATTER_GET_PROFILE(src_c, src_h, src_w, dst_h, src_addr, dst_addr, data_format, \
        direction, src_C_is1, index_C_is1, src_wstride, dst_wstride, src_hstride, dst_hstride, \
        src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, inplace_add, pid_node)

#define SDMA_TENSOR_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,        \
      special_func, store_type, src_wstride, dst_wstride, src_hstride,        \
      dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)   \

#define SDMA_CONSTANT_GET_PROFILE( dst_n, dst_c, dst_h, dst_w,  src_addr, dst_addr, data_format,  \
        special_func, dst_wstride, dst_hstride,                                     \
        dst_cstride, dst_nstride, stride_enable, pid_node)

#define SDMA_GENERAL_GET_PROFILE(src_count, src_addr, dst_addr, data_format, src_is_const, pid_node)

#define SDMA_CW_TRANS_GET_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr, data_format,         \
      src_wstride, dst_wstride, src_hstride,                                                     \
     dst_hstride,src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define SDMA_FILTER_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , data_format,  pid_node)

#define SDMA_NONZERO_PROFILE(src_n, src_c, src_h, src_w,  src_addr, dst_addr , src_data_format, dst_data_format, pid_node)

#define SDMA_GATHER_GET_PROFILE(src_c, src_h, src_w, index_H, src_addr, dst_addr, data_format, \
      src_wstride, dst_wstride, src_hstride,                                           \
      dst_hstride, src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, pid_node)

#define SDMA_SCATTER_GET_PROFILE(src_c, src_h, src_w, dst_h, src_addr, dst_addr, data_format, \
        src_C_is1, index_C_is1, src_wstride, dst_wstride, src_hstride, dst_hstride, \
        src_cstride, dst_cstride, src_nstride, dst_nstride, stride_enable, inplace_add, pid_node)

#endif
