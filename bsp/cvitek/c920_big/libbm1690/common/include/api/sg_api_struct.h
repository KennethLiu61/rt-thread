#ifndef SG_API_STRUCT_H
#define SG_API_STRUCT_H

#pragma pack(push, 1)
#include "common_def.h"

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif

#ifndef MAX_ROI_ALIGN_NUM_LEVELS
#define MAX_ROI_ALIGN_NUM_LEVELS   5
#endif

#define MAX_CONCATLAYER_INPUT_NUM 10
#define MAX_PROPOSAL_LAYER_OUTPUT_ROI_NUM 1000
/*
 * The following api struct information must be compatible with bm1684
 */
typedef struct sg_api_memset {
    unsigned long long global_offset;
    unsigned int height;
    unsigned int width;
    int mode;
    int val;
} __attribute__((packed)) sg_api_memset_t;

typedef struct sg_api_memcpy {
    unsigned long long src_global_offset;
    unsigned long long dst_global_offset;
    int input_n;
    int src_nstride;
    int dst_nstride;
    int count;
} __attribute__((packed)) sg_api_memcpy_t;

typedef struct sg_api_memcpy_wstride {
    unsigned long long src_global_offset;
    unsigned long long dst_global_offset;
    int src_wstride;
    int dst_wstride;
    int count;
    int format_bytes;
} __attribute__((packed)) sg_api_memcpy_wstride_t;

typedef struct sg_api_memcpy_byte {
    unsigned long long src_global_offset;
    unsigned long long dst_global_offset;
    unsigned long long size;
} __attribute__((packed)) sg_api_memcpy_byte_t;

typedef struct sg_api_memcpy_system_local {
    unsigned long long system_addr;
    unsigned long long local_addr;
    int n;
    int c;
    int h;
    int w;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_memcpy_system_local_t;
#else
} sg_api_memcpy_system_local_t;
#endif

/*
 * Then the new api struct information can be set without consider bm168x.
 * Please add new api struct in the following.
 */

typedef struct sg_api_pooling_parallel {
    unsigned long long ifmap_offset_global;
    unsigned long long ofmap_offset_global;
    unsigned long long max_mask_offset_global;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int kh;
    int kw;
    int pad_h;
    int pad_w;
    int pad_h_after;
    int pad_w_after;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int is_avg_pooling;
    int avg_pooling_mode;
    int max_with_mask;
    int if_relu;
    float relu_upper_limit;
    sg_data_type_t   data_type;
    int is_max;
#ifndef WIN32
} __attribute__((packed)) sg_api_pooling_parallel_t;
#else
} sg_api_pooling_parallel_t;
#endif

typedef struct sg_api_pooling_fix8b_parallel {
    unsigned long long ifmap_offset_global;
    unsigned long long ofmap_offset_global;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int kh;
    int kw;
    int pad_h_top;
    int pad_h_bottom;
    int pad_w_left;
    int pad_w_right;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int is_avg_pooling;
    int avg_pooling_mode;
    int ceil_mode;
    sg_data_type_t   output_dtype;
    sg_data_type_t   input_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_pooling_fix8b_parallel_t;
#else
} sg_api_pooling_fix8b_parallel_t;
#endif

typedef struct sg_api_pooling_fp8_parallel {
    unsigned long long ifmap_offset_global;
    unsigned long long ofmap_offset_global;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int kh;
    int kw;
    int pad_h;
    int pad_w;
    int pad_h_after;
    int pad_w_after;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int is_avg_pooling;
    int avg_pooling_mode;
    int max_with_mask;
    int if_relu;
    float relu_upper_limit;
    float re_scale;
    sg_data_type_t   output_dtype;
    sg_data_type_t   input_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_pooling_fp8_parallel_t;
#else
} sg_api_pooling_fp8_parallel_t;
#endif


typedef struct {
    unsigned long long A_addr;
    unsigned long long B_addr;
    unsigned long long R_addr;
    int op;
    int n;
    int c;
    int h;
    int w;
#if defined(__mars3__) || defined(__sgtpuv8__)
    sg_data_type_t dType;
#endif
#ifndef WIN32
} __attribute__((packed)) sg_api_arithmetic_t;
#else
} sg_api_arithmetic_t;
#endif

typedef struct {
    unsigned long long src_data_addr;
    unsigned long long src_idx_addr;
    unsigned long long dst_data_addr;
    unsigned long long dst_idx_addr;
    int len;
    int topk;
    int is_descend;
    int idx_en;
    int auto_idx;
    int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_sort_t;
#else
} sg_api_sort_t;
#endif

typedef struct sg_api_scale_forward {
    unsigned long long bottom_global_addr;
    unsigned long long scale_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long top_global_addr;
    int                bottom_n;
    int                bottom_c;
    int                bottom_h;
    int                bottom_w;
    int                axis; // scale begin axis
    int                axis_num; // scale axis num
    int                has_bias;
    int                if_relu;
    float              relu_upper_limit;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_scale_forward_t;
#else
} sg_api_scale_forward_t;
#endif

typedef struct sg_api_bnscale_fix8b_forward {
    unsigned long long bottom_global_addr;
    unsigned long long scale_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long shift_global_addr;
    unsigned long long top_global_addr;
    int                bottom_n;
    int                bottom_c;
    int                bottom_h;
    int                bottom_w;
    int                input_sign;
    int                scale_sign;
    int                bias_sign;
    int                if_relu;
    int                relu_upper_limit;
    sg_round_mode_t    round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_bnscale_fix8b_forward_t;
#else
} sg_api_bnscale_fix8b_forward_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long bias_addr;
    unsigned long long Y_addr;
    int L_row_num;
    int L_col_num;
    int R_col_num;
    int transpose;
    int have_bias;
    sg_data_type_t L_dtype;
    sg_data_type_t R_dtype;
    sg_data_type_t Y_dtype;
    sg_data_type_t bias_dtype;
    int if_relu;
    float relu_upper_limit;
    int rshift_bit;
#ifndef WIN32
} __attribute__((packed)) sg_api_fc_t;
#else
} sg_api_fc_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long B_addr;
    unsigned long long Y_addr;
    unsigned long long rescale_addr;
    int batch_num;
    int L_row_num;
    int L_col_num;
    int R_col_num;
    sg_data_type_t L_dtype;
    sg_data_type_t R_dtype;
    sg_data_type_t Y_dtype;
    int if_relu;
    float relu_upper_limit;
    int use_bias;
    int do_rescale;
    int rescale_is_const;
    int rescale_const_val;
#ifndef WIN32
} __attribute__((packed)) sg_api_batch_matmul_t;
#else
} sg_api_batch_matmul_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long zp_addr;
    unsigned long long Y_addr;
    int batch_num;
    int L_row_num;
    int L_col_num;
    int R_col_num;
    sg_data_type_t L_dtype;
    sg_data_type_t R_dtype;
    int zp_is_const;
    int zp_const_val;
#ifndef WIN32
} __attribute__((packed)) sg_api_batch_matmul_fix8b_t;
#else
} sg_api_batch_matmul_fix8b_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long scale_global_addr;
  unsigned long long zp_global_addr;
  unsigned long long output_global_addr;
  int final_row_num;
  int inner_num;
  int final_col_num;
  int has_bias;
  int has_zp;
  int scale_zp_zip;
  int q_group_size;
  sg_data_type_t weight_dtype;
  sg_data_type_t bias_dtype;
  int R_trans;
  int sign;
  int weight_bits;
  sg_data_type_t out_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_a8_matmul_t;
#else
} sg_api_a8_matmul_t;
#endif

typedef struct {
    unsigned long long bottom_A_global_offset;
    unsigned long long bottom_B_global_offset;
    unsigned long long top_global_offset;
    unsigned long long mask_global_offset;
    int input_num;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    int op_code;
    int coeff_A;
    int coeff_B;
    int need_mask;
    int mask_index_A;
    int mask_index_B;
    int if_relu;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_eltwise_t;
#else
} sg_api_eltwise_t;
#endif

typedef struct {
    unsigned long long bottom_A_global_addr;
    unsigned long long bottom_B_global_addr;
    unsigned long long top_global_addr;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    int op_code;
    int scale_A;
    int scale_B;
    int sign_A;
    int sign_B;
    int rshift_A;
    int rshift_B;
    int if_relu;
    int round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_eltwise_fix8b_t;
#else
} sg_api_eltwise_fix8b_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long keys_addr;
    unsigned long long values_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long weight_out_addr;
    unsigned long long table_addr;
    unsigned long long mask_addr;
    unsigned long long Y_addr;
    int batch_num;
    int M_queries_num;
    int M_keys_num;
    int N_num;
    int dim;
    int hasbias;
    float scale;
    int has_mask;
    int quant_param[16];
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_attention_fix8b_t;
#else
} sg_api_attention_fix8b_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long keys_addr;
    unsigned long long values_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long weight_out_addr;
    unsigned long long mask_addr;
    unsigned long long Y_addr;
    int batch_num;
    int M_queries_num;
    int M_keys_num;
    int N_num;
    int dim;
    int hasbias;
    float scale;
    int has_mask;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_attention_t;
#else
} sg_api_attention_t;
#endif

typedef struct {
    unsigned long long bottom_global_offset;
    unsigned long long top_global_offset;
    unsigned long long slope_global_offset;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    int channel_shared;
    float slope_val;
    float upper_limit;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_prelu_t;
#else
} sg_api_prelu_t;
#endif

typedef struct {
    unsigned long long bottom_global_offset;
    unsigned long long top_global_offset;
    unsigned long long slope_global_offset;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    int channel_shared;
    int slope_val;
    int rshift_bit;
    int upper_limit;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_prelu_fix8b_t;
#else
} sg_api_prelu_fix8b_t;
#endif

typedef struct {
    unsigned long long bottom_global_offset;
    unsigned long long top_global_offset;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    float upper_limit;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relu_t;
#else
} sg_api_relu_t;
#endif

typedef struct {
    unsigned long bottom_global_offset;
    unsigned long top_global_offset;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    float upper_limit;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relu_local_t;
#else
} sg_api_relu_local_t;
#endif

typedef struct {
    unsigned long long bottom_global_offset;
    unsigned long long top_global_offset;
    int tensor_n;
    int tensor_c;
    int tensor_h;
    int tensor_w;
    int axis;
    int offset;
    int axis_num;
    int axis_list[FW_MAX_SHAPE_DIMS];
    int offset_list[FW_MAX_SHAPE_DIMS];
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_swap_dim_t;
#else
} sg_api_swap_dim_t;
#endif

typedef struct {
    unsigned long long bottom_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long rescale_global_addr;
    unsigned long long top_global_addr;
    int                bottom_n;
    int                bottom_c;
    int                bottom_h;
    int                bottom_w;
    int                kh;
    int                kw;
    int                stride_h;
    int                stride_w;
    int                pad_h_t;
    int                pad_h_b;
    int                pad_w_l;
    int                pad_w_r;
    int                dh;
    int                dw;
    int                has_bias;
    int                if_relu;
    float              relu_upper_limit;
    int                if_rescale;
    sg_data_type_t     dtype;
    sg_data_type_t     out_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_depthwise_normal_t;
#else
} sg_api_depthwise_normal_t;
#endif

typedef struct {
    unsigned long long bottom_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long top_global_addr;
    int                bottom_n;
    int                bottom_c;
    int                bottom_h;
    int                bottom_w;
    int                kh;
    int                kw;
    int                stride_h;
    int                stride_w;
    int                pad_h_t;
    int                pad_h_b;
    int                pad_w_l;
    int                pad_w_r;
    int                dh;
    int                dw;
    int                has_bias;
    int                if_relu;
    int                relu_upper_limit;
    int                rshift_num;
    int                input_sign;
    int                weight_sign;
    int                bias_sign;
    sg_data_type_t     output_dtype;
    sg_round_mode_t    round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_depthwise_fix8b_normal_t;
#else
} sg_api_depthwise_fix8b_normal_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int ishape[4];
    int groups;
    int output_c;
    int kernel[2];
    int stride[2];
    int dilation[2];
    int pad[4];
    int has_bias;
    int if_relu;
    int upper_limit;
    int rshift;
    sg_data_type_t idtype;
    sg_data_type_t wdtype;
    sg_data_type_t bdtype;
    sg_data_type_t odtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_quant_sym_t;
#else
} sg_api_conv_quant_sym_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
     unsigned long long requant_global_addr;
    unsigned long long output_global_addr;
    int ishape[4];
    int groups;
    int output_c;
    int kernel[2];
    int stride[2];
    int dilation[2];
    int pad[4];
    int has_bias;
    int if_relu;
    int if_requant;
    int requant_is_const;
    int rshift;
    int yzp;
    int upper_limit;
    sg_data_type_t idtype;
    sg_data_type_t wdtype;
    sg_data_type_t bdtype;
    sg_data_type_t odtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_requant_t;
#else
} sg_api_conv_requant_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long rescale_global_addr;
    unsigned long long output_global_addr;
    int ishape[4];
    int groups;
    int output_c;
    int kernel[2];
    int stride[2];
    int dilation[2];
    int pad[4];
    int has_bias;
    int if_relu;
    float upper_limit;
    int result_add;
    int has_rescale;
    sg_data_type_t idtype;
    sg_data_type_t odtype;
    int weight_is_coeff;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_float_t;
#else
} sg_api_conv_float_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long padding_insert_global_addr;
    unsigned long long kernel_grad_global_addr;
    unsigned long long nc_trans_buffer_global_addr;
    int ishape[4];
    int output_c;
    int forward_kernel[2]; //kh, kw
    int forward_output[2]; //oh, ow
    int ins[2]; //ins_h, ins_w
    int dilation[2]; //dh, dw
    int stride[2]; //stride_h, stride_w
    int pad[4]; //pad_h_t, pad_h_b, pad_w_l, pad_w_r
    unsigned int ins_const_val;
    int pad_ins_is_const;
    int pad_mode;
    sg_data_type_t input_dtype;
    sg_data_type_t grad_dtype;
    sg_data_type_t output_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_bw_float_t;
#else
} sg_api_conv_bw_float_t;
#endif

typedef struct {
    unsigned long long grad_out_global_addr;
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    unsigned long long buffer_global_addr;
    int groups;
    int input_shape[4];
    int grad_out_shape[4];
    int kernel[2]; //kh, kw
    int ins[2]; //ins_h, ins_w
    int dilation[2]; //dh, dw
    int stride[2]; //stride_h, stride_w
    int pad[4]; //pad_h_t, pad_h_b, pad_w_l, pad_w_r
    unsigned int ins_const_val;
    int pad_ins_is_const;
    int pad_mode;
    int grad_input_enable;
    int grad_weight_enable;
    int grad_bias_enable;
    sg_data_type_t input_dtype;
    sg_data_type_t grad_dtype;
    sg_data_type_t grad_weight_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_bwd_t;
#else
} sg_api_conv_bwd_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  unsigned long long buffer_global_addr;
  int input_shape[4];
  int output_shape[4];
  int groups;
  int kernel[2];
  int stride[2];
  int dilation[2];
  int pad[4];
  sg_data_type_t dtype;
  int weight_formated;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv2d_backward_t;
#else
} sg_api_conv2d_backward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int input_n;
    int input_c;
    int input_inner_dim;
    float scale_val;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_t;
#else
} sg_api_softmax_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long table_global_addr;
    int Tensor_N;
    int Tensor_C;
    int Tensor_H;
    int Tensor_W;
    int zero_point;
    float scale_val;
    sg_data_type_t input_dtype;
    sg_data_type_t output_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_tflite_fix8b_t;
#else
} sg_api_softmax_tflite_fix8b_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long buffer_global_addr;
    unsigned long long output_global_addr;
    int input_shape[8];
    int axis_list[8];
    int shape_dims;
    int axis_num;
    int method;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_reduce_t;
#else
} sg_api_reduce_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long buffer_global_addr;
    unsigned long long output_global_addr;
    int input_shape[8];
    int axis_list[8];
    int shape_dims;
    int axis_num;
    int method;
    float input_scale;
    float output_scale;
    sg_data_type_t input_dtype;
    sg_data_type_t output_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_reduce_fix8b_t;
#else
} sg_api_reduce_fix8b_t;
#endif

typedef struct {
    unsigned long long A_global_addr;
    unsigned long long B_global_addr;
    unsigned long long res_global_addr;
    int A_shape[4];
    int B_shape[4];
    int scale_A;
    int scale_B;
    int rshift_A;
    int rshift_B;
    sg_data_type_t A_dtype;
    sg_data_type_t B_dtype;
    sg_data_type_t res_dtype;
    sg_binary_type_t binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_bcbinary_fix8b_t;
#else
} sg_api_bcbinary_fix8b_t;
#endif

typedef struct {
    unsigned long long A_global_addr;
    unsigned long long B_global_addr;
    unsigned long long res_global_addr;
    int A_shape[FW_MAX_SHAPE_DIMS];
    int B_shape[FW_MAX_SHAPE_DIMS];
    int dims;
    sg_data_type_t dtype;
    sg_binary_type_t binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_bcbinary_float_t;
#else
} sg_api_bcbinary_float_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    sg_binary_type_t binary_type;
    sg_data_type_t dtype;
    float const_value;
    int is_inversed;
#ifndef WIN32
} __attribute__((packed)) sg_api_const_binary_float_t;
#else
} sg_api_const_binary_float_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    float min;
    float max;
    int if_relu;
    sg_data_type_t dtype;
    float relu_upper_limit;
#ifndef WIN32
} __attribute__((packed)) sg_api_clip_float_t;
#else
} sg_api_clip_float_t;
#endif

typedef struct {
  unsigned long long bottom_global_offset;
  unsigned long long mean_global_offset;
  unsigned long long variance_global_offset;
  unsigned long long top_global_offset;
  int                input_n;  // note this is total input_n
  int                input_c;
  int                input_h;
  int                input_w;
  int                if_relu;
  float              relu_upper_limit;
  sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm_forward_inference_parallel_t;
#else
} sg_api_batchnorm_forward_inference_parallel_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long running_mean_global_addr;
    unsigned long long running_variance_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    unsigned long long saved_mean_global_addr;
    unsigned long long saved_invstd_global_addr;
    unsigned long long running_mean_update_global_addr;
    unsigned long long running_var_update_global_addr;
    int in_shape[4];
    float momentum;
    float eps;
    int if_relu;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm_forward_train_multicore_t;
#else
} sg_api_batchnorm_forward_train_multicore_t;
#endif

typedef struct {
    unsigned long long bnbwd_grad_input_glolbal_addr;
    unsigned long long bnbwd_grad_weight_global_addr;
    unsigned long long bnbwd_grad_bias_global_addr;
    unsigned long long relu_output_global_addr;
    unsigned long long relu_grad_output_glolbal_addr;
    unsigned long long bn_input_global_addr;
    unsigned long long bn_weight_global_addr;
    unsigned long long bn_bias_global_addr;
    unsigned long long bn_running_mean_global_addr;
    unsigned long long bn_running_invstd_global_addr;
    int relu_input_shape[4];  //[n, oc, oh, ow]
    int do_recompute;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_where_bnbwd_multicore_t;
#else
} sg_api_where_bnbwd_multicore_t;
#endif

typedef struct {
    unsigned long long A_global_addr;
    unsigned long long res_global_addr;
    int A_shape[4];
    int scale_A;
    int rshift_A;
    int B_const_val;
    int inversed;
    sg_data_type_t A_dtype;
    sg_data_type_t B_dtype;
    sg_data_type_t res_dtype;
    sg_binary_type_t binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_const_binary_fix8b_t;
#else
} sg_api_const_binary_fix8b_t;
#endif

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    int shape[8];
    int shape_dim;
    sg_active_type_t active_type;
    float coef[8];
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_active_t;
#else
} sg_api_active_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long index_global_addr;
    unsigned long long output_global_addr;
    int input_shape[8];
    int shape_dims;
    int index_num;
    int axis;
    int const_val;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_index_select_t;
#else
} sg_api_index_select_t;
#endif

typedef struct {
  unsigned long long input_global_mem_addr;
  unsigned long long output_global_mem_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int order[FW_MAX_SHAPE_DIMS];
  int dims;
  unsigned long long buffer_global_mem_addr;
  sg_data_type_t     sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_transpose_t;
#else
} sg_api_transpose_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  int input_shape[4];
  int block_sizes[2];
  int in_is_nchw;
  int out_is_nchw;
  int is_inversed;
  int is_crd_mode;
  int swap_cr;
  sg_data_type_t sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_depth2space_t;
#else
} sg_api_depth2space_t;
#endif

typedef struct {
  unsigned long long input_global_mem_addr;
  unsigned long long output_global_mem_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int block_sizes[2];
  int crop_sizes[4];
  sg_data_type_t sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batch2space_t;
#else
} sg_api_batch2space_t;
#endif

typedef struct {
  unsigned long long input_global_addrs[MAX_CONCATLAYER_INPUT_NUM];
  unsigned long long output_global_addr;
  int                input_shapes[MAX_CONCATLAYER_INPUT_NUM][FW_MAX_SHAPE_DIMS];
  int                st_by_concatway[MAX_CONCATLAYER_INPUT_NUM];
  int                input_num;
  int                concat_axis;
  int                input_dims;
  sg_data_type_t     sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_concat_t;
#else
} sg_api_concat_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int size;
    int if_relu;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_upsample_t;
#else
} sg_api_upsample_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long requant_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int is_perchannel;
    int scale_value;
    int shift_value;
    int offset_value;
    sg_data_type_t bottom_dtype;
    sg_data_type_t top_dtype;
    int mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_requant_int_t;
#else
} sg_api_requant_int_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long requant_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int is_perchannel;
    float scale_value;
    float offset_value;
    sg_data_type_t bottom_dtype;
    sg_data_type_t top_dtype;
    int mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_requant_float_t;
#else
} sg_api_requant_float_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long dequant_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int is_perchannel;
    float scale_value;
    int offset_value;
    sg_data_type_t bottom_dtype;
    sg_data_type_t top_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_dequant_float_t;
#else
} sg_api_dequant_float_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long dequant_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int gsize;
    sg_data_type_t src_dtype;
    sg_data_type_t dst_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_dequant_half_prec_t;
#else
} sg_api_dequant_half_prec_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long dequant_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int is_perchannel;
    int scale_value;
    int offset_value;
    int shift_value;
    sg_data_type_t bottom_dtype;
    sg_data_type_t top_dtype;
    int mode;
    int lshift;
    sg_round_mode_t round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_dequant_int_t;
#else
} sg_api_dequant_int_t;
#endif

typedef struct {
    unsigned long long input_data_addr;
    unsigned long long input_index_addr;
    unsigned long long output_data_addr;
    unsigned long long output_index_addr;
    unsigned long long buffer_addr;
    int input_index_valid;
    int k;
    int descending;
    int batchs;
    int batch_num;
    int batch_stride;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_topk_t;
#else
} sg_api_topk_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int              input_n;
    int              input_c;
    int              input_h;
    int              input_w;
    int              output_h;
    int              output_w;
    int              pad_bag;
    int              pad_end;
    int              align_corners;
    int              half_pixel_centers;
    PLATFORM_SUPPORT platform_sp;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_interp_parallel_t;
#else
} sg_api_interp_parallel_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long Y_addr;
    int L_row_num;
    int L_col_num;
    int R_col_num;
    int transpose;
    sg_data_type_t L_dtype;
    sg_data_type_t R_dtype;
    sg_data_type_t Y_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_gemm_t;
#else
} sg_api_gemm_t;
#endif

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    sg_round_mode_t round_mode;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_round_fp_t;
#else
} sg_api_round_fp_t;
#endif

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long idx_global_addr;
    unsigned long long val_global_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                axis;
    int                method;
    int                is_index_int32;
    int                select_last_index;
    int                need_val;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_arg_t;
#else
} sg_api_arg_t;
#endif

typedef struct {
    unsigned long long input;
    unsigned long long output;
    unsigned long long scale;
    int               across_spatial;
    int               channel_share;
    int                input_n;
    int                input_c;
    int                input_h;
    int                input_w;
    float              eps;
    float              scale_val;
    int                if_relu;
#ifndef WIN32
} __attribute__((packed)) sg_api_normalize_t;
#else
} sg_api_normalize_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long index_global_addr;
    unsigned long long output_global_addr;
    int                input_shape[FW_MAX_SHAPE_DIMS];
    int                shape_dims;
    int                indices_shape[FW_MAX_SHAPE_DIMS];
    int                indices_dims;
    int                const_val;
    int                batch_dims;
    sg_data_type_t     dtype;
} sg_api_gather_nd_tf_t;

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    unsigned long long pos_num_addr;
    unsigned long long buffer_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                order;
    sg_data_type_t     sgdtype;
} sg_api_where_t;

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    unsigned long long buf_global_addr;
    int                input_n;
    int                input_c;
    int                input_h;
    int                input_w;
    int                num;
    int                classes;
    int                coords;
    int                background;
    int                softmax;
    sg_data_type_t     dtype;
    int                test_glb;
#ifndef WIN32
} __attribute__((packed)) sg_api_yolo_t;
#else
} sg_api_yolo_t;
#endif

typedef struct {
    unsigned long long data_global_addr;
    unsigned long long output_data_global_addr;
    unsigned long long output_index_global_addr;
    unsigned long long buffer_global_addr;
    int                input_shape[FW_MAX_SHAPE_DIMS];
    int                input_dims;
    int                sort_dim;
    int                is_argsort;
    int                stable;
    int                descending;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_sort_per_dim_t;
#else
} sg_api_sort_per_dim_t;
#endif

typedef struct {
    unsigned long long  out_global_addr;
    int                 shape[FW_MAX_SHAPE_DIMS];
    int                 shape_dim;
    unsigned int        filled_value;
    sg_data_type_t      filled_sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_constant_fill_t;
#else
} sg_api_constant_fill_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long rois_global_addr;
    unsigned long long output_global_addr;
    int             input_n;
    int             input_c;
    int             input_h;
    int             input_w;
    int             roi_num;
    int             roi_len;
    int             pooled_h;
    int             pooled_w;
    float           spatial_scale;
    int             position_sensitive;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_roi_pooling_t;
#else
} sg_api_roi_pooling_t;
#endif

typedef struct {
    unsigned long long ifmap_offset_global;
    unsigned long long ofmap_offset_global;
    unsigned long long rois_offset_global;
    unsigned long long rois_offset_gloabl_tmp;
    int             input_n;
    int             input_c;
    int             input_h;
    int             input_w;
    int             output_dim;
    int             group_size;
    int             roi_num;
    float           spatial_scale;
    sg_data_type_t  dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_psroi_pooling_t;
#else
} sg_api_psroi_pooling_t;
#endif

typedef struct {
  unsigned long long input_proposal_addr;
  unsigned long long output_proposal_addr;
  unsigned long long all_mask_addr;
  unsigned long long iou_addr;
  int nms_type; //0:hard nms, 1:soft nms 2:ADAPTIVE_NMS 3:SSD_NMS
  int proposal_size;
  float nms_threshold;

  //below pararemter is used for soft-nms, ADAPTIVE_NMS, ssd_nms
  float score_threshold;
  float sigma;
  int   weighting_method;
  float eta;
  int hard_nms_version; //0: v1, 1: v2
  int keep_top_k; //just for hard_nms v2
#ifndef WIN32
} __attribute__((packed)) sg_api_nms_t;
#else
} sg_api_nms_t;
#endif

typedef struct {
  unsigned long long input_dets_addr;
  unsigned long long input_scores_addr;
  unsigned long long output_indx_addr;
  int dets_num;
  int dets_dim;
  float iou_threshold;
#ifndef WIN32
} __attribute__((packed)) sg_api_rotated_nms_t;
#else
} sg_api_rotated_nms_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long input_weight0_addr;
  unsigned long long input_weight1_addr;
  int shape_dim;
  int shape[4];
  int W_shape[4];
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_rope_t;
#else
} sg_api_rope_t;
#endif

typedef struct {
  unsigned long long   loc_global_offset;
  unsigned long long   conf_global_offset;
  unsigned long long   prior_global_offset;
  unsigned long long   buffer_global_offset;
  unsigned long long   top_global_offset;
  int   batch_num;
  int   num_prior;
  int   num_classes;
  int   num_loc_classes;
  int   share_location;
  int   background_label_id;
  int   top_k;
  int   code_type;
  int   keep_top_k;
  int   variance_encoded_in_target;
  float nms_threshold;
  float conf_threshold;
  float eta;
  int onnx_nms; //1:onnx_nms, 0: normal case
#ifndef WIN32
} __attribute__((packed)) sg_api_ssd_detect_out_t;
#else
} sg_api_ssd_detect_out_t;
#endif

typedef struct {
  unsigned long long    in_global_addr;
  unsigned long long    shift_global_addr;
  unsigned long long    out_global_addr;
  int                   shape[FW_MAX_SHAPE_DIMS];
  int                   shape_dim;
  int                   is_per_channel; //0:use shift_value as const shift value; 1:per channel, use shift_global_addr as shift value which has C shift value
  int                   shift_value;
  sg_data_type_t        in_dtype;
  sg_data_type_t        shift_dtype;
  sg_data_type_t        out_dtype;
  sg_round_mode_t       sg_round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_arithmetic_shift_t;
#else
} sg_api_arithmetic_shift_t;
#endif

typedef struct {
    unsigned long long   input_global_addr;
    unsigned long long   rois_global_addr;
    unsigned long long   output_global_addr;
    int                  input_n;
    int                  input_c;
    int                  input_h;
    int                  input_w;
    int                  roi_num;
    int                  roi_len;
    int                  pooled_height;
    int                  pooled_width;
    float                spatial_scale;
    int                  sampling_ratio;
    int                  position_sensitive;
    int                  align_corners;
    int                  plat_sp;
    sg_data_type_t       dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_roi_align_t;
#else
} sg_api_roi_align_t;
#endif

typedef struct {
    unsigned long long   input_global_addrs[MAX_ROI_ALIGN_NUM_LEVELS];
    unsigned long long   rois_global_addr;
    unsigned long long   target_lvls_global_addr;
    unsigned long long   output_global_addr;
    int                  input_shapes[MAX_ROI_ALIGN_NUM_LEVELS][FW_MAX_SHAPE_DIMS];
    int                  num_levels;
    int                  input_dims;
    int                  roi_num;
    int                  roi_len;
    int                  pooled_height;
    int                  pooled_width;
    float                spatial_scales[MAX_ROI_ALIGN_NUM_LEVELS];
    int                  sampling_ratio;
    int                  position_sensitive;
    int                  align_corners;
    int                  plat_sp;
    sg_data_type_t       dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_roi_extractor_t;
#else
} sg_api_roi_extractor_t;
#endif

typedef struct sg_api_column_hash {
    unsigned long long  input_global_addr[11];
    unsigned long long  output_global_addr;
    unsigned long long  coeff_global_addr;
    unsigned int        len;
             int        init;
             int        mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_column_hash_t;
#else
} sg_api_column_hash_t;
#endif

typedef struct {
  unsigned long long      in_global_addr;
  unsigned long long      out_global_addr;
  int                     shape[FW_MAX_SHAPE_DIMS];
  int                     shape_dim;
  int                     shift_axis;
  sg_shift_type_t         shift_dir; //0:left, 1:right, 2:circle left, 3:circle right
  int                     shift_num;
  sg_data_type_t          in_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_shift_t;
#else
} sg_api_shift_t;
#endif

typedef struct sg_api_tile {
    unsigned long long  in_global_addr;
    unsigned long long  buf_global_addr;
    unsigned long long  out_global_addr;
    int                 input_shape[FW_MAX_SHAPE_DIMS];
    int                 tile_coeff[FW_MAX_SHAPE_DIMS];
    int                 input_dim;
    int                 type;
    sg_data_type_t      dtype;
    int                 test_glb;
#ifndef WIN32
} __attribute__((packed)) sg_api_tile_t;
#else
} sg_api_tile_t;
#endif

typedef struct sg_api_split {
    unsigned long long  in_global_addr;
    unsigned long long  out_global_addr[10];
    int                 input_shape[FW_MAX_SHAPE_DIMS];
    int                 input_dim;
    int                 split_axis;
    int                 split_num;
    int                 split_size[10];
    sg_data_type_t      dtype;
    int                 test_glb;
#ifndef WIN32
} __attribute__((packed)) sg_api_split_t;
#else
} sg_api_split_t;
#endif

typedef struct sg_api_yolov3_detect_out{
    unsigned long long b0_global_addr;
    unsigned long long b1_global_addr;
    unsigned long long b2_global_addr;
    unsigned long long buffer_global_offset;
    unsigned long long top_global_addr;
    unsigned long long all_mask_ddr_addr; //for to support large box, need to alloc ddr at outside
    int input_num;
    int batch_num;
    int hw_shape[3][2];
    int num_classes;
    int num_boxes;
    int mask_group_size;
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    float bias[18];
    float anchor_scale[3];
    float mask[9];
    int yolov5_flag; //0: yolov3 1:yolov5 2:yolov7
    int len_per_batch;
    int scale;//for yolov7 post hanle
    int orig_image_shape[100]; //reserved 800 bytes
    int model_h;
    int model_w;
#ifndef WIN32
} __attribute__((packed)) sg_api_yolov3_detect_out_t;
#else
} sg_api_yolov3_detect_out_t;
#endif

typedef struct {
    unsigned long long   input_global_addr;
    unsigned long long   mask_global_addr;
    unsigned long long   buffer_global_addr;
    unsigned long long   output_global_addr;
    unsigned long long   mask_count_global_addr;
    int                  input_shape[FW_MAX_SHAPE_DIMS];
    int                  mask_shape[FW_MAX_SHAPE_DIMS];
    int                  input_dims;
    int                  mask_dims;
    int                  bcast_from_begin;
    sg_data_type_t       input_dtype;
    sg_data_type_t       mask_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_masked_select_t;
#else
} sg_api_masked_select_t;
#endif

typedef struct sg_api_pad {
    unsigned long long  input_global_addr;
    unsigned long long  output_global_addr;
    int                 input_n;
    int                 input_c;
    int                 input_h;
    int                 input_w;
    int                 pad[4][2];
    int                 type;
    float               constant;
    sg_data_type_t      dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_pad_t;
#else
} sg_api_pad_t;
#endif

typedef struct sg_api_upsample_mask_forward_parallel {
  unsigned long long bottom_global_offset;
  unsigned long long bottom_mask_global_offset;
  unsigned long long top_global_offset;
  int bottom_global_N;
  int bottom_c;
  int bottom_h;
  int bottom_w;
  int size;
#ifndef WIN32
} __attribute__((packed)) sg_api_upsample_mask_forward_parallel_t;
#else
} sg_api_upsample_mask_forward_parallel_t;
#endif

typedef struct {
  unsigned long long      a_global_addr;
  unsigned long long      b_global_addr;
  unsigned long long      res_global_addr;
  int                     a_shape[FW_MAX_SHAPE_DIMS];
  int                     b_shape[FW_MAX_SHAPE_DIMS];
  int                     shape_dim;
  sg_binary_type_t        binary_op;
  int                     rshift_num;
  int                     b_is_const;
  int                     b_const_val;
  int                     a_is_coeff;
  int                     b_is_coeff;
  int                     inversed;
  sg_data_type_t          a_dtype;
  sg_data_type_t          b_dtype;
  sg_data_type_t          res_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_shift_t;
#else
} sg_api_binary_shift_t;
#endif

typedef struct {
    unsigned long long    buffer_global_addr;
    unsigned long long    output_global_addr;
    float            mins[FW_MAX_SHAPE_DIMS];
    float            maxs[FW_MAX_SHAPE_DIMS];
    float            asps[FW_MAX_SHAPE_DIMS];
    float            vars[FW_MAX_SHAPE_DIMS];
    int              min_size;
    int              max_size;
    int              asp_size;
    int              var_size;
    float            step_h;
    float            step_w;
    int              img_h;
    int              img_w;
    int              fmp_h;
    int              fmp_w;
    int              num_priors;
    float            offset;
    int              clip;
    float            thTop;
    sg_data_type_t   odtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_prior_box_t;
#else
} sg_api_prior_box_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned long long length;
  int divisor;
  int quant_mode; // 0: float(input/divisor), 1:trim((input+0.5*divisor)/divisor), 2: trim((input+sign(input)*0.5*divisor)/divisor)
  sg_data_type_t in_dtype;
  sg_data_type_t out_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_quant_div_t;
#else
} sg_api_quant_div_t;
#endif

typedef struct sg_api_strideslice {
    unsigned long long  in_global_addr;
    unsigned long long  out_global_addr;
    int                 input_shape[FW_MAX_SHAPE_DIMS];
    int                 shape_dim;
    int                 begin_mask;
    int                 end_mask;
    int                 begin_index[FW_MAX_SHAPE_DIMS];
    int                 end_index[FW_MAX_SHAPE_DIMS];
    int                 strides[FW_MAX_SHAPE_DIMS];
    sg_data_type_t      dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_strideslice_t;
#else
} sg_api_strideslice_t;
#endif


// ** used for test all instructions performance
typedef enum _InstructionMode{
    MODE_ALL = 0,
    MM2_NN = 1,
    MM2_NT = 2,
    LIN_MODE_MAC = 3,
    LIN_MODE_ADD_SQR = 4,
    LIN_MODE_SUB_SQR = 5,
    AR_MODE_MUL = 6,
    AR_MODE_ADD = 7,
    AR_MODE_SUB = 8,
    AR_MODE_COPY = 9,
    RQ_MODE_0 = 10,
    RQ_MODE_1 = 11,
    DQ_MODE_0 = 12,
    DQ_MODE_1 = 13,
    DW_MODE_DW = 14,
    DW_MODE_AP = 15,
    DW_MODE_MP = 16,
    AR_MODE_DIV = 17,
    AR_MODE_MAX = 18,
    AR_MODE_ABS = 19,
    AR_MODE_AND = 20,
    AR_MODE_CAST = 21,
    AR_MODE_NOT = 22,
    AR_MODE_SELECT_GREAT = 23,
} InsMode;

// ** used for test all instructions performance
typedef enum _InstructionType{
    INS_ALL = 0,
    INS_CONV = 1,
    INS_MAT_MUL1 = 2,
    INS_MAT_MUL2 = 3,
    INS_FUSED_COMPARE =4,
    INS_SPECIAL = 5,
    INS_FUSED_LINEAR=6,
    INS_VECTOR_CORRELATION=7,
    ARITHMETIC=8,
    INS_TENSOR_s2s = 9,
    INS_TENSOR_s2l =10,
    INS_TENSOR_l2s =11,
    INS_MATRIX_s2l=12,
    INS_MATRIX_l2s =13,
    INS_TENSOR_CW_TRANPOSE_s2l=14,
    INS_TENSOR_CW_TRANPOSE_l2s=15,
    INS_TENSOR_CW_TRANPOSE_s2s=16,
    INS_MAX_POOL=17,
    INS_AVG_POOL=18,
    INS_DEPTHWISE=19,
    INS_ROI_MAX_POOL=20,
    INS_ROI_AVG_POOL=21,
    INS_ROI_DEPTHWISE=22,
    INS_CW_TRANSPOSE=23,
    INS_REQ=24,
    INS_DEQ=25,
    INS_LANE_BCAST=26,
    INS_STATIC_BCAST=27,
    INS_LANE_COPY=28,
} InsType;

typedef struct sg_api_test_instruction{
    InsType type;
    int     loop_times;
    int     dtype;
    InsMode mode;
    int     use_multi_engine;
    int     idle_max_interleave;
    unsigned long long input_addr;
    unsigned long long output_addr;
#ifndef WIN32
} __attribute__((packed)) sg_api_test_instruction;
#else
} sg_api_test_instruction;
#endif

typedef struct {
    unsigned long long  in_global_addr;
    unsigned long long  mask_global_addr;
    unsigned long long  out_global_addr;
    int                 input_shape[FW_MAX_SHAPE_DIMS];
    int                 mask_shape[FW_MAX_SHAPE_DIMS];
    int                 input_dims;
    int                 mask_dims;
    float               value;
    sg_data_type_t      dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_masked_fill_t;
#else
} sg_api_masked_fill_t;
#endif

typedef struct {
    unsigned long long    input_global_addr;
    unsigned long long    grid_global_addr;
    unsigned long long    output_global_addr;
    int                   input_n;
    int                   input_c;
    int                   input_h;
    int                   input_w;
    int                   output_h;
    int                   output_w;
    int                   align_corners;
    GridSampleInterpMode  interp_mode;
    GridSamplePaddingMode padding_mode;
    sg_data_type_t        dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_grid_sample_t;
#else
} sg_api_grid_sample_t;
#endif

typedef struct {
    unsigned long long  in_global_addr;
    unsigned long long  out_global_addr;
    int                 shape[FW_MAX_SHAPE_DIMS];
    int                 shape_dim;
    int                 group;
    sg_data_type_t      dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_shuffle_channel_t;
#else
} sg_api_shuffle_channel_t;
#endif

typedef struct {
  unsigned long long input_global_mem_addr;
  unsigned long long output_global_mem_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int oh;
  int ow;
  int mode;
  sg_data_type_t sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_adaptive_pool_t;
#else
} sg_api_adaptive_pool_t;
#endif

typedef struct {
    unsigned long long input_data_addr;
    unsigned long long medium_data_addr;
    unsigned long long output_data_addr;
    int num_embeddings;
    int embedding_dim;
    int mode;
    int indices_size;
    int offsets_size;
    unsigned long long indices_addr;
    unsigned long long offsets_addr;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_embedding_bag_t;
#else
} sg_api_embedding_bag_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int is_upper;
  int diagonal;
  sg_data_type_t sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_triangularize_t;
#else
} sg_api_triangularize_t;
#endif

typedef struct {
    unsigned long input_local_addr;
    unsigned long output_local_addr;
    unsigned long buffer_local_addr;
    int shape[4];
    int is_upper;
    int diagonal;
    int hidx;
    int widx;
    sg_data_type_t sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_triangularize_local_t;
#else
} sg_api_triangularize_local_t;
#endif

typedef struct {
    unsigned long long input_data_addr;
    unsigned long long medium_data_addr;
    unsigned long long output_data_addr;
    int num_embeddings;
    int embedding_dim;
    int mode;
    int indices_size;
    int offsets_size;
    unsigned long long indices_addr;
    unsigned long long offsets_addr;
    unsigned long long scale_addr;
    unsigned long long bias_addr;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_embedding_bag_fix8b_t;
#else
} sg_api_embedding_bag_fix8b_t;
#endif

typedef struct sg_api_proposal {
    unsigned long long score_data_addr;
    unsigned long long box_data_addr;
    unsigned long long src_info_data_addr;
    int feat_stride;
    int min_size;
    int pre_nms_topN;
    int post_nms_topN;
    float nms_thresh;
    float score_thresh;
    int base_size;
    int scales_num;
    int ratios_num;
    int anchor_scales[5];
    float ratios[5];
    int batch_num;
    int map_height;
    int map_width;
    int distribute_fpn_proposals_flag;//for paddle distribute_fpn_proposals
    int score_out_flag;
    unsigned long long out_addr;
    unsigned long long score_out_addr;
#ifndef WIN32
} __attribute__((packed)) sg_api_proposal_t;
#else
} sg_api_proposal_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long index_addr;
    unsigned long long value_addr;
    unsigned long long output_addr;
    unsigned long long buffer_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    int indice_len;
    int mode;
    int accumulate;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_index_put_t;
#else
} sg_api_index_put_t;
#endif

typedef struct sg_api_deform_gather {
    unsigned long long input_addr;
    unsigned long long offset_addr;
    unsigned long long mask_addr;
    unsigned long long output_addr;
    unsigned long long buffer_addr;
    int           input_n;
    int           input_c;
    int           input_h;
    int           input_w;
    int           modulated;
    int           deform_groups;
    int           kh;
    int           kw;
    int           pad_h;
    int           pad_w;
    int           pad_h_after;
    int           pad_w_after;
    int           stride_h;
    int           stride_w;
    int           dilation_h;
    int           dilation_w;
    int           mode;
    int           offset_interleave;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_deform_gather_t;
#else
} sg_api_deform_gather_t;
#endif

typedef struct sg_api_bilinear_interpolate {
    unsigned long long  input_addr;
    unsigned long long  map_y_addr;
    unsigned long long  map_x_addr;
    unsigned long long  output_addr;
    int            input_c;
    int            input_h;
    int            input_w;
    int            map_yx_len;
    int            map_yx_c_need_bcast;
    int            mode;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_bilinear_interpolate_t;
#else
} sg_api_bilinear_interpolate_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long output_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                axis;
    int                group_num;
    float              eps;
    int                affine;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_group_norm_t;
#else
} sg_api_group_norm_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long output_addr;
    unsigned long long mean_addr;
    unsigned long long rstd_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                axis;
    float              eps;
    int                affine;
    int                need_mean;
    int                need_rstd;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_layer_norm_t;
#else
} sg_api_layer_norm_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight_addr;
    unsigned long long output_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    float              eps;
    int                affine;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_rms_norm_t;
#else
} sg_api_rms_norm_t;
#endif

typedef struct sg_api_sparse_conv3d {
    unsigned long long  input_features_global_addr;
    unsigned long long  input_coor_global_addr;
    unsigned long long  weight_global_addr;
    unsigned long long  origin_input_shape_global_addr;
    unsigned long long  intermedia_mem_pool_global_addr;
    unsigned long long  intermedia_mem_pool_ex_global_addr;
    unsigned long long  output_features_global_addr;
    unsigned long long  output_coor_global_addr;
    unsigned long long  origin_output_shape_global_addr;
    unsigned long long  debug_pool_global_addr;
    int                 case_num;
    int                 batch_num;
    int                 limit_active_out_num;
    int                 ndim;
    int                 output_channel;
    int                 input_channel;
    int                 kernel_size[3];
    int                 stride[3];
    int                 padding[3];
    int                 dilation[3];
    int                 has_bias;
    int                 subm;
    int                 output_padding[3];
    sg_data_type_t      feature_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_sparse_conv3d_t;
#else
} sg_api_sparse_conv3d_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long output_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    float              eps;
    int                affine;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_pixel_norm_t;
#else
} sg_api_pixel_norm_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight_addr;
    unsigned long long bias_addr;
    unsigned long long output_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    float              scale;
    float              eps;
    int                affine;
    sg_data_type_t     idtype;
    sg_data_type_t     odtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_pixel_norm_fix8b_t;
#else
} sg_api_pixel_norm_fix8b_t;
#endif

typedef struct {
    unsigned long long      col_addr;
    unsigned long long      im_adddr;
    int                     img_n;
    int                     img_c;
    int                     img_h;
    int                     img_w;
    int                     kernel_h;
    int                     kernel_w;
    int                     stride_h;
    int                     stride_w;
    int                     dilation_h;
    int                     dilation_w;
    int                     pad_left;
    int                     pad_right;
    int                     pad_top;
    int                     pad_bottom;
    sg_data_type_t          dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_col2im_t;
#else
} sg_api_col2im_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    sg_data_type_t     dtype;
    int                axis; // scale begin axis
#ifndef WIN32
} __attribute__((packed)) sg_api_reverse_t;
#else
} sg_api_reverse_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long compressed_racu_addr;
  unsigned long long compressed_meta_addr;
  unsigned long long output_addr;
  int shape[4];
  sg_data_type_t dtype;
  int bias0;
  int bias1;
  int zero_guard;
#ifndef WIN32
} __attribute__((packed)) sg_api_compress_racu_t;
#else
} sg_api_compress_racu_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long compressed_addr;
  unsigned long long output_addr;
  int shape[4];
  sg_data_type_t dtype;
  int bias0;
  int bias1;
  int zero_guard;
#ifndef WIN32
} __attribute__((packed)) sg_api_compress_normal_t;
#else
} sg_api_compress_normal_t;
#endif

typedef struct {
    unsigned long long input0_addr;
    unsigned long long input1_addr;
    unsigned long long output_addr;
    unsigned long long buffer_addr;
    int n;
    int c;
    int h;
    int w;
    int reduce_opcode;
#ifndef WIN32
} __attribute__((packed)) sg_api_lossy_compress_t;
#else
} sg_api_lossy_compress_t;
#endif


typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned long long reversed_addr;
  int shape[4];
  int axis;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_local_reverse_t;
#else
} sg_api_local_reverse_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long rzp_addr;
    unsigned long long bias_addr;
    unsigned long long requant_addr;
    unsigned long long Y_addr;
    int batch_num;
    int hsize;
    int L_row_num;
    int L_col_num;
    int R_row_num;
    int R_col_num;
    sg_data_type_t L_dtype;
    sg_data_type_t R_dtype;
    sg_data_type_t rzp_dtype;
    sg_data_type_t bias_dtype;
    sg_data_type_t Y_dtype;
    int L_trans;
    int R_trans;
    int bias_is_const;
    int bias_const_val;
    int rzp_is_const;
    int rzp_const_val;
    int izp_const_val;
    int requant_mode;
    int is_perchannel;
    int scale_val;
    int offset_val;
    int shift_val;
    sg_round_mode_t round_mode;
    int do_sym_saturate;
    int do_relu;
#ifndef WIN32
} __attribute__((packed)) sg_api_batch_matmul_fix8b_ext_t;
#else
} sg_api_batch_matmul_ext_t;
#endif

typedef struct {
    unsigned long long tpu_cmd_addr;
    unsigned long long gdma_cmd_addr;
    unsigned long long hau_cmd_addr;
    unsigned long long sdma_cmd_addr;
    unsigned long long imm_buf_addr;
    unsigned long long pio_addr;
    unsigned long long pio_addr_o;
    unsigned long long input_addr;
    unsigned long long output_addr;
    unsigned long long io_addr;
    unsigned long long input_origin_addr;
    unsigned long long output_origin_addr;
    unsigned long long io_origin_addr;
    int tpu_cmd_nums;
    int gdma_cmd_nums;
    int hau_cmd_nums;
    int sdma_cmd_nums;
    unsigned int tpu_cmd_byte_size;
    unsigned int gdma_cmd_byte_size;
    unsigned int hau_cmd_byte_size;
    unsigned int sdma_cmd_byte_size;
    unsigned int imm_buf_byte_size;
    int output_offset;
    int loop;
    int enable_pio_des_interleave;
    unsigned int input_byte_size;
    unsigned int output_byte_size;
    unsigned int io_byte_size;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_sync_t;
#else
} sg_api_msg_sync_t;
#endif

typedef struct {
    unsigned long long addr;
    unsigned int size;
    unsigned long long origin_addr;
#ifndef WIN32
} __attribute__((packed)) sg_api_io_param_t;
#else
} sg_api_io_param_t;
#endif

typedef struct
{
    unsigned long long tpu_cmd_addr;
    unsigned long long gdma_cmd_addr;
    unsigned long long hau_cmd_addr;
    unsigned long long sdma_cmd_addr;
    unsigned long long cdma_cmd_addr;
    unsigned long long imm_buf_addr;
    int                tpu_cmd_nums;
    int                gdma_cmd_nums;
    int                hau_cmd_nums;
    int                sdma_cmd_nums;
    int                cdma_cmd_nums;
    unsigned int       tpu_cmd_byte_size;
    unsigned int       gdma_cmd_byte_size;
    unsigned int       hau_cmd_byte_size;
    unsigned int       sdma_cmd_byte_size;
    unsigned int       cdma_cmd_byte_size;
    unsigned int       imm_buf_byte_size;
    int                dtype;
    unsigned long long l2mem_reduce_addr;
    int                reduce_opcde;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_sync_cdma_single_task_t;
#else
} sg_api_msg_sync_cdma_single_task_t;
#endif

#define MAX_CDMA_CASE_NUM 10
typedef struct
{
    int core_id;
    unsigned int cdma_test_loop;
    sg_api_msg_sync_cdma_single_task_t engine_param[MAX_CDMA_CASE_NUM];
    unsigned long long  pio_addr;
    unsigned int       pio_byte_size;
    int                enable_pio_des_interleave;
    unsigned long long  input_addr;
    unsigned long long  output_addr;
    unsigned long long  input_cmd_addr;
    unsigned long long  output_cmd_addr;
    unsigned int       input_byte_size;
    unsigned int       output_byte_size;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_sync_cdma_t;
#else
} sg_api_msg_sync_cdma_t;
#endif

typedef struct {
    sg_api_msg_sync_t param[8];
    int loop;
    int enable_pio_des_interleave;
    int core_num;
    sg_api_io_param_t input[20]; // Align with test_msg_sync_multi_core MAX_IO_NUM
    sg_api_io_param_t output[20]; // Align with test_msg_sync_multi_core MAX_IO_NUM
    unsigned long long total_io_size;
    int input_num;
    int output_num;
    unsigned long long placeholder_addr;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_sync_multi_core_t;
#else
} sg_api_msg_sync_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  int shape[4];
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_cwtrans_t;
#else
} sg_api_cwtrans_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  sg_active_type_t active_type;
  float coeff[FW_MAX_SHAPE_DIMS];
  sg_data_type_t dtype;
  int slice_num;
  int slice_idx;
#ifndef WIN32
} __attribute__((packed)) sg_api_active_multi_core_t;
#else
} sg_api_active_multi_core_t;
#endif

typedef struct {
    int loop;
#ifndef WIN32
} __attribute__((packed)) sg_api_llama_multi_core_t;
#else
} sg_api_llama_multi_core_t;
#endif

typedef struct {
    unsigned long long blob_A_0;
    unsigned long long blob_B_0;
    unsigned long long blob_T_0;
    unsigned long long blob_A_1;
    unsigned long long blob_B_1;
    unsigned long long blob_T_1;
    int                 op;
    int                 n;
    int                 c;
    int                 h;
    int                 w;
    int                 test_core_idx0;
    int                 test_core_idx1;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_central_multi_core_t;
#else
} sg_api_msg_central_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long weight0_addr;
  unsigned long long weight1_addr;
  unsigned long long bias0_addr;
  unsigned long long bias1_addr;
  unsigned long long output_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int w0_shape[FW_MAX_SHAPE_DIMS];
  int w1_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int w0_dims;
  int w1_dims;
  sg_data_type_t in_dtype;
  sg_data_type_t out_dtype;
  int has_bias;
  int use_fast;
#ifndef WIN32
} __attribute__((packed)) sg_api_mgm_multi_core_t;
#else
} sg_api_mgm_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long weight0_addr;
  unsigned long long mat0_addr;
  unsigned long long gelu_addr;
  unsigned long long weight1_addr;
  unsigned long long grad_output_addr;
  unsigned long long grad_input_addr;
  unsigned long long grad_weight0_addr;
  unsigned long long grad_bias0_addr;
  unsigned long long grad_mat0_addr;
  unsigned long long grad_weight1_addr;
  unsigned long long grad_bias1_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int w0_shape[FW_MAX_SHAPE_DIMS];
  int w1_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int w0_dims;
  int w1_dims;
  sg_data_type_t dtype;
  int has_bias;
  int use_fast;
#ifndef WIN32
} __attribute__((packed)) sg_api_mgm_bwd_multi_core_t;
#else
} sg_api_mgm_bwd_multi_core_t;
#endif

typedef struct {
  unsigned long long input0_addr;
  unsigned long long input1_addr;
  unsigned long long gamma_addr;
  unsigned long long beta_addr;
  unsigned long long weight_addr;
  unsigned long long bias_addr;
  unsigned long long output_addr;
  unsigned long long gelu_output_addr;
  unsigned long long norm_output_addr;
  unsigned long long norm_mean_addr;
  unsigned long long norm_rstd_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int weight_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int weight_dims;
  sg_data_type_t dtype;
  int has_bias;
  int use_fast;
  float eps;
#ifndef WIN32
} __attribute__((packed)) sg_api_mlp0_fuse_multi_core_t;
#else
} sg_api_mlp0_fuse_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long gamma_addr;
  unsigned long long beta_addr;
  unsigned long long weight_addr;
  unsigned long long bias_addr;
  unsigned long long output_addr;
  unsigned long long norm_mean_addr;
  unsigned long long norm_rstd_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int weight_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int weight_dims;
  sg_data_type_t dtype;
  int has_bias;
  float eps;
#ifndef WIN32
} __attribute__((packed)) sg_api_layernorm_matmul_fuse_multi_core_t;
#else
} sg_api_layernorm_matmul_fuse_multi_core_t;
#endif

typedef struct {
  unsigned long long grad_out_addr;
  unsigned long long softmax_out_addr;
  unsigned long long cond_addr;
  unsigned long long grad_in_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int cond_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  sg_data_type_t dtype;
  float value;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_where_bwd_fuse_multi_core_t;
#else
} sg_api_softmax_where_bwd_fuse_multi_core_t;
#endif

typedef struct {
  unsigned long long output_addr;
  unsigned long long cond_addr;
  unsigned long long self_addr;
  unsigned long long other_addr;
  int out_shape[FW_MAX_SHAPE_DIMS];
  int cond_shape[FW_MAX_SHAPE_DIMS];
  int self_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dims;
  sg_data_type_t cond_dtype;
  sg_data_type_t dtype;
  int self_is_scalar;
  int other_is_scalar;
  float self_val;
  float other_val;
#ifndef WIN32
} __attribute__((packed)) sg_api_where_multi_core_t;
#else
} sg_api_where_multi_core_t;
#endif

typedef struct {
  unsigned long long input0_addr;
  unsigned long long input1_addr;
  unsigned long long output_addr;
  int in0_shape[FW_MAX_SHAPE_DIMS];
  int in1_shape[FW_MAX_SHAPE_DIMS];
  int in0_dims;
  int in1_dims;
  int binary_type;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_multi_core_t;
#else
} sg_api_binary_multi_core_t;
#endif

typedef struct {
  unsigned long long input0_addr;
  unsigned long long input1_addr;
  unsigned long long grad_output_addr;
  unsigned long long grad_input0_addr;
  unsigned long long grad_input1_addr;
  int in0_shape[FW_MAX_SHAPE_DIMS];
  int in1_shape[FW_MAX_SHAPE_DIMS];
  int in0_dims;
  int in1_dims;
  int binary_type;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_bwd_multi_core_t;
#else
} sg_api_binary_bwd_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long grad_output_addr;
  unsigned long long grad_input_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  float const_value;
  int is_inversed;
  int binary_type;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_const_binary_float_bwd_t;
#else
} sg_api_const_binary_float_bwd_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long target_addr;
  unsigned long long weight_addr;
  unsigned long long output_addr;
  unsigned long long sum_addr;
  unsigned long long max_addr;
  int ignore_index;
  int batch_num;
  int class_num;
  int reduction;
  float label_smoothing;
  int target_is_int64;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_cross_entropy_multi_core_t;
#else
} sg_api_cross_entropy_multi_core_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long target_addr;
  unsigned long long weight_addr;
  unsigned long long grad_output_addr;
  unsigned long long grad_input_addr;
  unsigned long long sum_addr;
  unsigned long long max_addr;
  int batch_num;
  int class_num;
  int reduction;
  float label_smoothing;
  int target_is_int64;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_cross_entropy_bwd_multi_core_t;
#else
} sg_api_cross_entropy_bwd_multi_core_t;
#endif

typedef struct {
    unsigned long long left_global_addr;
    unsigned long long right_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int                L_shape[FW_MAX_SHAPE_DIMS];
    int                R_shape[FW_MAX_SHAPE_DIMS];
    int                L_dims;
    int                R_dims;
    int                L_trans;
    int                R_trans;
    sg_data_type_t     in_dtype;
    sg_data_type_t     out_dtype;
    int                slice_core_m;
    int                slice_core_n;
    int                slice_m;
    int                slice_n;
    int                slice_k;
    int                left_slice_dim;
    int                right_slice_dim;
    int                result_slice_dim;
    unsigned long long left_8ch_global_addr[8];
    unsigned long long right_8ch_global_addr[8];
    unsigned long long result_8ch_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_matmul_multi_core_t;
#else
} sg_api_matmul_multi_core_t;
#endif

typedef struct {
    unsigned long long X_global_addr;      //global address of input matrix
    unsigned long long loraA_global_addr;  //global address of loraA matrix
    unsigned long long loraB_global_addr;  //global address of loraB matrix
    unsigned long long W_global_addr;  //global address of original weight matrix
    unsigned long long output_global_addr; //global address of output matrix
    int X_shape[FW_MAX_SHAPE_DIMS];
    int loraA_shape[FW_MAX_SHAPE_DIMS];
    int loraB_shape[FW_MAX_SHAPE_DIMS];
    int X_dims;
    int loraA_dims;
    int loraB_dims;
    sg_data_type_t in_dtype;
    sg_data_type_t out_dtype;
    int do_scale;
    float scale_val;
#ifndef WIN32
} __attribute__((packed)) sg_api_lora_matmul_multicore_t;
#else
} sg_api_lora_matmul_multicore_t;
#endif

typedef struct {
    unsigned long long left_global_addr;
    unsigned long long right_global_addr;
    unsigned long long grad_out_global_addr;
    unsigned long long grad_left_global_addr;
    unsigned long long grad_right_global_addr;
    int                L_shape[FW_MAX_SHAPE_DIMS];
    int                R_shape[FW_MAX_SHAPE_DIMS];
    int                Y_shape[FW_MAX_SHAPE_DIMS];
    int                L_dims;
    int                R_dims;
    int                Y_dims;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_matmul_bwd_multi_core_t;
#else
} sg_api_matmul_bwd_multi_core_t;
#endif

typedef struct {
    unsigned long long Q_global_addr;
    unsigned long long K_global_addr;
    unsigned long long V_global_addr;
    unsigned long long Y_global_addr;
    unsigned long long where_cond_global_addr;
    float              C;
    float              where_other_val;
    int                batch;
    int                N;
    int                d;
    sg_data_type_t     dtype;
    sg_data_type_t     where_cond_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_gpt_qkv_multi_core_t;
#else
} sg_api_gpt_qkv_multi_core_t;
#endif

typedef struct {
  unsigned long long input;
  unsigned long long output;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  sg_data_type_t     dtype;
  int                input_slice_dim;
  int                output_slice_dim;
  unsigned long long input_8ch[8];
  unsigned long long output_8ch[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_gelu_forward_multi_core_t;
#else
} sg_api_gelu_forward_multi_core_t;
#endif

typedef struct {
  unsigned long long grad_input;
  unsigned long long grad_output;
  unsigned long long input;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  sg_data_type_t     dtype;
  int                input_slice_dim;
  int                output_slice_dim;
  unsigned long long grad_input_8ch[8];
  unsigned long long grad_output_8ch[8];
  unsigned long long input_8ch[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_gelu_backward_multi_core_t;
#else
} sg_api_gelu_backward_multi_core_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  int                begin_dim;
  int                end_dim;
  float              scale_val;
  sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_forward_multi_core_t;
#else
} sg_api_softmax_forward_multi_core_t;
#endif

typedef struct {
  unsigned long long grad_input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long output_global_addr;
  int                shape[4];
  int                axis;
  float              scale_val;
  sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_backward_multi_core_t;
#else
} sg_api_softmax_backward_multi_core_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long output_global_addr;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  int                axis;
  float              eps;
  int                affine;
  sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_layernorm_forward_multi_core_t;
#else
} sg_api_layernorm_forward_multi_core_t;
#endif

typedef struct
{
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    int axis;
    float partial;
    float eps;
    int with_weight;
    int with_bias;
    sg_data_type_t dtype;
    int enable_8ch;
    int input_slice_dim;
    int weight_slice_dim;
    int bias_slice_dim;
    int output_slice_dim;
    unsigned long long input_8ch_global_addr[8];
    unsigned long long weight_8ch_global_addr[8];
    unsigned long long bias_8ch_global_addr[8];
    unsigned long long output_8ch_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_rmsnorm_forward_multi_core_t;
#else
} sg_api_rmsnorm_forward_multi_core_t;
#endif

typedef struct
{
    unsigned long long input_global_addr;
    unsigned long long residule_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    int axis;
    float partial;
    float eps;
    int add_residule;
    int with_weight;
    int with_bias;
    sg_data_type_t dtype;
    int enable_8ch;
    int input_slice_dim;
    int weight_slice_dim;
    int bias_slice_dim;
    int output_slice_dim;
    unsigned long long input_8ch_global_addr[8];
    unsigned long long weight_8ch_global_addr[8];
    unsigned long long bias_8ch_global_addr[8];
    unsigned long long output_8ch_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_add_rmsnorm_forward_multi_core_t;
#else
} sg_api_add_rmsnorm_forward_multi_core_t;
#endif

typedef struct
{
    unsigned long long grad_output_global_addr;
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long rms_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    int           with_weight;
    int           with_bias;
    int           shape[FW_MAX_SHAPE_DIMS];
    int           dims;
    int           axis;
    int           requires_grad_input;
    sg_data_type_t   dtype;
    int           input_slice_dim;
    int           output_slice_dim;
    float         eps;
    unsigned long long grad_output_8ch_global_addr[8];
    unsigned long long input_8ch_global_addr[8];
    unsigned long long grad_input_8ch_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_rmsnorm_backward_multi_core_t;
#else
} sg_api_rmsnorm_backward_multi_core_t;
#endif

typedef struct {
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  unsigned long long grad_weight_reduce_buffer;
  unsigned long long grad_bias_reduce_buffer;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  int                axis;
  int                affine;
  int                requires_grad_input;
  sg_data_type_t     dtype;
  int                input_slice_dim;
  int                output_slice_dim;
  unsigned long long grad_output_8ch[8];
  unsigned long long input_8ch[8];
  unsigned long long grad_input_8ch[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_layernorm_backward_multi_core_t;
#else
} sg_api_layernorm_backward_multi_core_t;
#endif

typedef struct
{
    unsigned long long weight_out_global_addr;
    unsigned long long m_out_global_addr;
    unsigned long long v_out_global_addr;
    unsigned long long vmax_out_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long weight_in_global_addr;
    unsigned long long m_in_global_addr;
    unsigned long long v_in_global_addr;
    unsigned long long vmax_in_global_addr;
    unsigned long long t_global_addr;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int amsgrad;
    int maximize;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_adam_backward_multi_core_t;
#else
} sg_api_adam_backward_multi_core_t;
#endif

typedef struct
{
    unsigned long long input_global_addr;
    unsigned long long weight0_global_addr;
    unsigned long long weight1_global_addr;
    unsigned long long weight2_global_addr;
    unsigned long long rescale0_global_addr;
    unsigned long long rescale1_global_addr;
    unsigned long long rescale2_global_addr;
    unsigned long long output_global_addr;
    unsigned long long silu_global_addr;
    unsigned long long sigmoid_global_addr;
    unsigned long long m0_global_addr;
    int save_mid_res;
    int batch;
    int input_w;
    int middle_w;
    int do_rescale;
    int rescale_is_const;
    float scale;
    float rescale0;
    float rescale1;
    float rescale2;
    sg_data_type_t dtype;
    int enable_8ch;
    int input_slice_dim;
    int weight0_slice_dim;
    int weight1_slice_dim;
    int weight2_slice_dim;
    int output_slice_dim;
    unsigned long long input_8ch_global_addr[8];
    unsigned long long weight0_8ch_global_addr[8];
    unsigned long long weight1_8ch_global_addr[8];
    unsigned long long weight2_8ch_global_addr[8];
    unsigned long long output_8ch_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_llama_mlp_forward_multi_core_t;
#else
} sg_api_llama_mlp_forward_multi_core_t;
#endif

typedef struct {
    int core_idx;
    int core_num;
    int core_msg_id;
    int name_len;
    int api_id;
    int api_size;
    unsigned char api_data[0];
#ifndef WIN32
} __attribute__((packed)) sg_api_core_info_t;
#else
} sg_api_core_info_t;
#endif

typedef struct sg_api_yolov5_detect_out{
    unsigned long long bottom_global_addr;
    unsigned long long top_global_addr;
    int input_shape[3];
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    int agnostic_nms;
    int max_hw;
#ifndef WIN32
} __attribute__((packed)) sg_api_yolov5_detect_out_t;
#else
} sg_api_yolov5_detect_out_t;
#endif

#define MAX_YOLO_INPUT_NUM 8
#define MAX_YOLO_ANCHOR_NUM 8
typedef struct sg_api_yolov5_decode_detect_out{
    unsigned long long bottom_global_addr[MAX_YOLO_INPUT_NUM];
    unsigned long long top_global_addr;
    unsigned long long detected_num_addr;
    int input_num;
    int batch_num;
    int hw_shape[MAX_YOLO_INPUT_NUM][2];
    int num_classes;
    int num_boxes;
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    float anchors[MAX_YOLO_INPUT_NUM * MAX_YOLO_ANCHOR_NUM * 2];
    float downsample_r[MAX_YOLO_INPUT_NUM];
    int clip_box;
    int agnostic_nms;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_yolov5_decode_detect_out_t;
#else
} sg_api_yolov5_decode_detect_out_t;
#endif

typedef struct sg_api_yolov8_detect_out{
    unsigned long long bottom_global_addr;
    unsigned long long top_global_addr;
    int input_shape[3];
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    int agnostic_nms;
    int max_hw;
#ifndef WIN32
} __attribute__((packed)) sg_api_yolov8_detect_out_t;
#else
} sg_api_yolov8_detect_out_t;
#endif

typedef struct sg_api_gdma_d2d{
    unsigned long long src_global_addr;
    unsigned long long dst_global_addr;
    unsigned long long size;
#ifndef WIN32
} __attribute__((packed)) sg_api_gdma_d2d_t;
#else
} sg_api_gdma_d2d_t;
#endif

typedef struct {
    int dst_chipid;
    int src_chipid;
    unsigned long long src_addr;
    int src_n;
    int src_c;
    int src_h;
    int src_w;
    int src_n_stride;
    int src_c_stride;
    int src_h_stride;
    int opcode;
    sg_data_type_t dtype;

    int core_idx;

#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_send_multi_core_t;
#else
} sg_api_cdma_send_multi_core_t;
#endif

typedef struct {
    int src_chipid;
    int dst_chipid;
    unsigned long long dst_addr;
    int dst_n;
    int dst_c;
    int dst_h;
    int dst_w;
    int dst_n_stride;
    int dst_c_stride;
    int dst_h_stride;
    int opcode;
    sg_data_type_t dtype;

    int core_idx;

#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_recv_multi_core_t;
#else
} sg_api_cdma_recv_multi_core_t;
#endif

typedef struct {
    int root_chipid;
    unsigned long long src_addr;
    unsigned long long dst_addr;
    int n;
    int c;
    int h;
    int w;
    int n_stride;
    int c_stride;
    int h_stride;
    sg_data_type_t dtype;

    int core_idx;
#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_broadcast_multi_core_t;
#else
} sg_api_cdma_broadcast_multi_core_t;
#endif

typedef struct {
    int root_chipid;
    unsigned long long src_addr;
    unsigned long long dst_addr;
    int n;
    int c;
    int h;
    int w;
    int n_stride;
    int c_stride;
    int h_stride;
    sg_data_type_t dtype;
    sg_reduce_method_t reduce_method;

    int core_idx;
#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_reduce_multi_core_t;
#else
} sg_api_cdma_reduce_multi_core_t;
#endif

typedef struct {
    unsigned long long src_addr;
    unsigned long long dst_addr;
    int n;
    int c;
    int h;
    int w;
    int n_stride;
    int c_stride;
    int h_stride;
    sg_data_type_t dtype;

    int core_idx;
#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_all_gather_multi_core_t;
#else
} sg_api_cdma_all_gather_core_t;
#endif

#define MAX_RANKS 16
typedef struct {
    int nranks; // number of ranks
    int rank; // rank of this chip
    int use_ring;
    int chip_map[MAX_RANKS]; // available port map
} sccl_args_t;

typedef enum _c2cCommunicationType{
    INTRA_CARD = 0,
    INTER_CARD = 1,
    INTER_CHIP = 2
} C2CCommunicationType;

typedef enum _c2cDirectType{
    DIRECT_LEFT2RIGHT = 0,
    DIRECT_RIGHT2LEFT = 1,
    DIRECT_BIDIR = 2,
} C2CDirectType;

typedef enum _c2cCopyType{
    COPY_S2S = 0,
    COPY_S2L2 = 1,
    COPY_L22S = 2,
    COPY_L22L2 =3,
} C2CCopyType;
typedef struct {
    unsigned long long count;
    C2CCommunicationType comm_type;
    C2CDirectType direct_type;
    C2CCopyType copy_type;
    unsigned long long info_addr;
    int dtype;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_test_c2c_perf_t;
#else
} sg_api_test_c2c_perf_t;
#endif

typedef struct {
    unsigned long long data_addr;
    unsigned long long count;
    int dtype;
    int dst_rank;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_send_t;
#else
} sg_api_c2c_send_t;
#endif

typedef struct {
    unsigned long long data_addr;
    unsigned long long count;
    int dtype;
    int src_rank;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_recv_t;
#else
} sg_api_c2c_recv_t;
#endif

typedef struct {
    unsigned long long data_addr;
    unsigned long long count;
    int dtype;
    int root;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_broadcast_t;
#else
} sg_api_c2c_broadcast_t;
#endif

typedef struct {
    unsigned long long send_buf;
    int send_type;
    unsigned long long recv_buf;
    unsigned long long recv_count;
    int recv_type;
    unsigned long  root;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_scatter_t;
#else
} sg_api_c2c_scatter_t;
#endif

typedef struct {
    unsigned long long send_buf;
    int send_type;
    unsigned long long recv_buf;
    unsigned long long recv_count;
    int recv_type;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_alltoall_t;
#else
} sg_api_c2c_alltoall_t;
#endif

typedef struct {
    unsigned long long send_addr;
    unsigned long long recv_addr;
    unsigned long long count;
    int dtype;
    sg_reduce_method_t reduce_method;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_all_reduce_t;
#else
} sg_api_c2c_all_reduce_t;
#endif

typedef struct {
    unsigned long long reduce_data_addr;
    unsigned long long cdma_cmd_addr[3];
    unsigned long long vsdma_cmd_addr[8];
    unsigned long long gdma_cmd_addr;
    int cdma_cmd_num[3];
    int vsdma_cmd_num[8];
    int gdma_cmd_num;
    int cdma_engine_num;
    int vsdma_engine_num;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_descriptor_t;
#else
} sg_api_c2c_descriptor_t;
#endif

typedef struct {
    unsigned long long send_addr;
    unsigned long long recv_addr;
    unsigned long long count;
    int dtype;
    sg_reduce_method_t reduce_method;
    int root;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_reduce_t;
#else
} sg_api_c2c_reduce_t;
#endif

typedef struct {
    unsigned long long send_addr;
    unsigned long long send_count;
    unsigned long long recv_addr;
    unsigned long long recv_count;
    int dtype;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_all_gather_t;
#else
} sg_api_c2c_all_gather_t;
#endif

typedef struct {
    unsigned long long send_addr;
    unsigned long long send_count;
    unsigned long long recv_addr;
    unsigned long long recv_count;
    int dtype;
    int root;
    int loop;
    sccl_args_t sccl_args;
#ifndef WIN32
} __attribute__((packed)) sg_api_c2c_gather_t;
#else
} sg_api_c2c_gather_t;
#endif

typedef struct {
  unsigned long long box_addr;
  unsigned long long score_addr;
  unsigned long long output_addr;
  unsigned long long buffer_global_offset;
  int batch_num;
  int num_priors;
  int num_classes;
  float nms_threshold;
  float score_threshold;
  int top_k;
  int center_point_box;
#ifndef WIN32
} __attribute__((packed)) sg_api_onnx_nms_t;
#else
} sg_api_onnx_nms_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_addcmul_t;
#else
} sg_api_addcmul_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_addcdiv_t;
#else
} sg_api_addcdiv_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    float                drop_rate;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_dropout_multi_core_t;
#else
} sg_api_dropout_multi_core_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long buffer_global_addr;
  unsigned int       element_num;
  int                dtype;
  unsigned int       case_;
#ifndef WIN32
} __attribute__((packed)) sg_api_sdma_thread_t;
#else
} sg_api_sdma_thread_t;
#endif

typedef struct {
  unsigned long long input_A_global_addr;
  unsigned long long input_B_global_addr;
  unsigned long long mask_global_addr;
  unsigned long long output_global_addr;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_multi_engine_test_t;
#else
} sg_api_multi_engine_test_t;
#endif

typedef struct {
  unsigned long long input_addr; // local mem data
  unsigned long long output_addr;
  int loop_num;
  int dtype;
  int save_last;
  int only_tiu;
#ifndef WIN32
} __attribute__((packed)) sg_api_tiu_loop_test_t;
#else
} sg_api_tiu_loop_test_t;
#endif

typedef struct {
  unsigned long long input_addr; // local mem data
  unsigned long long output_addr;
  int loop_num;
  int shape[4];
  int mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_gdma_loop_test_t;
#else
} sg_api_gdma_loop_test_t;
#endif

typedef struct {
    unsigned long long input_addr; // local mem data
    unsigned long long output0_addr;
    unsigned long long output1_addr;
    int N;
    int C;
    int H;
    int W;
    int dtype;
  #ifndef WIN32
  } __attribute__((packed)) sg_api_gdma_base_addr_t;
  #else
  } sg_api_gdma_base_addr_t;
  #endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  int loop_num;
  int shape[4];
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_loop_test_t;
#else
} sg_api_conv_loop_test_t;
#endif

typedef struct {
  unsigned long long input_addr; // local mem data
  unsigned long long output_addr;
  unsigned long long disable_mask; // reserved
  int loop_num;
  int max_run_num;
#ifndef WIN32
} __attribute__((packed)) sg_api_tpu_full_test_t;
#else
} sg_api_tpu_full_test_t;
#endif

typedef struct {
  unsigned long long input_addr; // local mem data
  unsigned long long weight_addr;
  unsigned long long output_addr;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_sync_debug_t;
#else
} sg_api_msg_sync_debug_t;
#endif

typedef struct {
    unsigned long long left_global_addr;
    unsigned long long right_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int                op_code;
    int                L_shape[FW_MAX_SHAPE_DIMS];
    int                R_shape[FW_MAX_SHAPE_DIMS];
    int                L_dims;
    int                R_dims;
    int                L_trans;
    int                R_trans;
    sg_data_type_t     in_dtype;
    sg_data_type_t     out_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_matmul_all_reduce_multi_core_t;
#else
} sg_api_matmul_all_reduce_multi_core_t;
#endif
typedef struct
{
    unsigned long long Q_global_addr;
    unsigned long long K_global_addr;
    unsigned long long V_global_addr;
    unsigned long long Qbuffer_global_addr;
    unsigned long long Kbuffer_global_addr;
    unsigned long long Vbuffer_global_addr;
    unsigned long long Kcache_global_addr;
    unsigned long long Vcache_global_addr;
    unsigned long long RoPE_cos_global_addr;
    unsigned long long RoPE_sin_global_addr;
    unsigned long long Mask_global_addr;
    unsigned long long input_length_global_addr;
    unsigned long long save_slots_global_addr;
    unsigned long long fetch_slots_global_addr;
    unsigned long long Y_global_addr;
    int slots_size;
    float C;
    int batch;
    int mask_max;
    int hidden_size;
    int num_attention_heads;
    int num_k_v_heads;
    int embeddings;
    int attention_mode;
    int block_size;
    sg_data_type_t dtype;
    int qkv_packed;
    int page_kv_cache_layout;
    char data[];  // dynamic data here
                  // input_length[batch]
#ifndef WIN32
} __attribute__((packed)) sg_api_llama2_qkv_multi_core_t;
#else
} sg_api_llama2_qkv_multi_core_t;
#endif

typedef struct
{
    unsigned long long Q_global_addr;
    unsigned long long K_global_addr;
    unsigned long long V_global_addr;
    unsigned long long Qbuffer_global_addr;
    unsigned long long Kbuffer_global_addr;
    unsigned long long Vbuffer_global_addr;
    unsigned long long RoPE_cos_global_addr;
    unsigned long long RoPE_sin_global_addr;
    unsigned long long Mask_global_addr;
    unsigned long long Y_global_addr;
    unsigned long long input_length_global_addr;
    unsigned long long Softmax_lse_global_addr;
    float C;
    float dropout_rate;
    int batch;
    int mask_max;
    int hidden_size;
    int num_attention_heads;
    int num_k_v_heads;
    sg_data_type_t dtype;
    int qkv_packed;
    int return_softmax;
    int disable_RoPE;
    int mask_batch;
    int disable_mask;
    char data[];  // dynamic data here
                  // input_length[batch]
#ifndef WIN32
} __attribute__((packed)) sg_api_llama_attention_forward_multi_core_t;
#else
} sg_api_llama_attention_forward_multi_core_t;
#endif

typedef struct {
  unsigned long long Q_global_addr;
  unsigned long long K_global_addr;
  unsigned long long V_global_addr;
  unsigned long long O_global_addr;
  unsigned long long dO_global_addr;
  unsigned long long l_global_addr;
  unsigned long long dQ_global_addr;
  unsigned long long dK_global_addr;
  unsigned long long dV_global_addr;
  unsigned long long cos_global_addr;
  unsigned long long sin_global_addr;
  unsigned long long mask_global_addr;
  unsigned long long input_lengths_global_addr;
  float C;
  int batch;
  int mask_max;
  int q_heads;
  int kv_heads;
  int hidden_size;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_llama2_qkv_backward_multi_core_t;
#else
} sg_api_llama2_qkv_backward_multi_core_t;
#endif

typedef struct
{
    unsigned long long Q_global_addr;
    unsigned long long K_global_addr;
    unsigned long long V_global_addr;
    unsigned long long Kcache_global_addr;
    unsigned long long Vcache_global_addr;
    unsigned long long Mask_global_addr;
    unsigned long long Y_global_addr;
    unsigned long long Input_length_global_addr;
    unsigned long long Save_slots_global_addr;
    unsigned long long Fetch_slots_global_addr;
    int slots_size;
    float C;
    int batch;
    int mask_max;
    int head_size;
    int num_attention_heads;
    int num_k_v_heads;
    int attention_mode;
    int block_size;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_llama2_attention_t;
#else
} sg_api_llama2_attention_t;
#endif

typedef struct
{
    unsigned long long start_offset; // start ddr offset
    unsigned long long stride; //stride of each copy
    unsigned long long data_len;  // num of data
    unsigned long long end_offset; // end ddr offset of the last copy
#ifndef WIN32
} __attribute__((packed)) sg_api_ddr_stress_t;
#else
} sg_api_ddr_stress_t;
#endif

// TODO: add more engines for sg2260
#define PROFILE_ENGINE_MCU 0
#define PROFILE_ENGINE_GDMA 1
#define PROFILE_ENGINE_TIU 2
#define PROFILE_ENGINE_CDMA 3
#define PROFILE_ENGINE_TGS 4 // for sg2260 TIU GDMA SDMA
#define PROFILE_PAUSE 5

typedef struct {
    int engine;
    unsigned long long addr;
    unsigned long long size;
#ifndef WIN32
} __attribute__((packed)) sg_api_engine_profile_param_t;
#else
} sg_api_engine_profile_param_t;
#endif

typedef struct {
    int enable;  // 0: disable, 1: enable. for 1686,sg2260, each bit controls one engine
#ifndef WIN32
} __attribute__((packed)) sg_api_set_profile_t;
#else
} sg_api_set_profile_t;
#endif

typedef struct {
    unsigned long long reserved;
    unsigned long long output_global_addr;
    unsigned int output_size;
    unsigned int byte_offset;
    unsigned int data_category;  // 0: profile_data, 1: profile extra data
#ifndef WIN32
} __attribute__((packed))  sg_api_get_profile_data_t;
#else
} sg_api_get_profile_data_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
#ifndef WIN32
} __attribute__((packed))  sg_api_depend_id_wraparound_data_t;
#else
} sg_api_depend_id_wraparound_data_t;
#endif

typedef struct {
  unsigned long long gdma_src_offset;
  unsigned long long gdma_dst_offset;
  unsigned long long gdma_reduce_src_offset[8];
  unsigned long long gdma_reduce_dst_offset[8];
  unsigned long long sdma_src_offset[8];
  unsigned long long sdma_dst_offset[8];
  unsigned long long cdma_src_offset[8];
  unsigned long long cdma_dst_offset[8];
  unsigned int gdma_shape[4];
  unsigned int gdma_reduce_shape[4];
  unsigned int sdma_shape[4];
  unsigned int sdma_reduce_shape[4];
  unsigned int cdma_shape[4];
  sg_data_type_t gdma_sg_dtype;
  sg_data_type_t gdma_reduce_sg_dtype;
  sg_data_type_t sdma_sg_dtype;
  sg_data_type_t sdma_reduce_sg_dtype;
  sg_data_type_t cdma_sg_dtype;
  sg_reduce_method_t gdma_sg_reduce_method;
  sg_reduce_method_t sdma_sg_reduce_method;
#ifndef WIN32
} __attribute__((packed)) sg_api_dma_k2k_stress_multi_core_t;
#else
} sg_api_dma_k2k_stress_multi_core_t;
#endif
typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
#ifndef WIN32
} __attribute__((packed))  sg_api_l2m_test_t;
#else
} sg_api_l2m_test_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long seq_len_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relative_position_t;
#else
} sg_api_relative_position_t;
#endif

typedef struct {
    unsigned long long dummy_value;
#ifndef WIN32
} __attribute__((packed)) sg_api_cdma_msg_central_test_multi_core_t;
#else
} sg_api_cdma_msg_central_test_multi_core_t;
#endif

typedef struct {
    int loop;
#ifndef WIN32
} __attribute__((packed)) sg_api_msg_central_stress_test_t;
#else
} sg_api_msg_central_stress_test_t;
#endif

typedef struct {
    unsigned long long input_global_addrs[5];
    unsigned long long output_global_addrs[5];
    unsigned long long buffer_global_addr;
    unsigned long long weight_bit32_gloabl_addr;
    unsigned long long weight_bit16_gloabl_addr;
    unsigned long long weight_bit8_gloabl_addr;
    int device_loop_times;
#ifndef WIN32
} __attribute__((packed)) sg_api_power_stress_t;
#else
} sg_api_power_stress_t;
#endif

typedef struct {
  unsigned long long spectral_embeddings_global_addr;
  unsigned long long num_spks_global_addr;
  unsigned long long eignvalue_global_addr;
  unsigned long long eignvector_global_addr;
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  int InShape[FW_MAX_SHAPE_DIMS];
  int  dims;
  int  num_iter_QR;
  int  num_spks;
  int  max_num_spks;
  int  min_num_spks;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_qr_householder_t;
#else
} sg_api_qr_householder_t;
#endif

typedef struct {
  unsigned long long centroids_global_addr;
  unsigned long long labels_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long buffer_global_addr;
  int  Shape_Input[FW_MAX_SHAPE_DIMS];
  int  Shape_Weight[FW_MAX_SHAPE_DIMS];
  int  dims_Input;
  int  dims_Weight;
  int  k;
  int  num_iter;
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_knn_naive_t;
#else
} sg_api_knn_naive_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long output_mask_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int kh;
    int kw;
    int top_pad_h;
    int left_pad_w;
    int bottom_pad_h;
    int right_pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int is_avg_pooling;
    int avg_pooling_mode;
    int if_relu;
    float relu_upper_limit;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_max_pooling_with_mask_forward_multi_core_t;
#else
} sg_api_max_pooling_with_mask_forward_multi_core_t;
#endif

typedef struct {
  unsigned long long  global_addr;
  int shape[4];
  int dst_stride[4];
  int src_stride[4];
  int dtype; // data_type_t
#ifndef WIN32
} __attribute__((packed)) sg_api_debug_lmem_t;
#else
} sg_api_debug_lmem_t;
#endif

typedef struct {
    unsigned long long output;
    int num;
#ifndef WIN32
} __attribute__((packed)) sg_api_reg_latency_t;
#else
} sg_api_reg_latency_t;
#endif

typedef struct
{
#ifdef DUMP_CACHED_CMD
    unsigned bytes;
#endif
    unsigned num;
    unsigned long long addr;
} sg_api_cmd_descriptor;

typedef struct
{
#ifdef USING_LLM_TICK_TOCK_PROFILE
    unsigned id;
#endif
    int addr_num;
    unsigned long long base_addr[31];
    sg_api_cmd_descriptor cmds[0];
} sg_api_launch_cache_multicore_t;

typedef struct
{
    int num;
    unsigned length[4];
    sccl_args_t sccl_args;
    sg_api_launch_cache_multicore_t param[];
} sg_api_cache_launch_batch_t;

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long target_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int ignore_index;
  int batch;
  int class_;
  int reduction;
  float label_smoothing;
  int dtype;
  int is_target_int64;
#ifndef WIN32
} __attribute__((packed)) sg_api_cross_entropy_loss_backward_t;
#else
} sg_api_cross_entropy_loss_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relu_backward_t;
#else
} sg_api_relu_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_tanh_backward_t;
#else
} sg_api_tanh_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_sigmoid_backward_t;
#else
} sg_api_sigmoid_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_silu_backward_t;
#else
} sg_api_silu_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_dtype;
  int output_dtype;
  int is_bool;
#ifndef WIN32
} __attribute__((packed)) sg_api_dtype_convert_t;
#else
} sg_api_dtype_convert_t;
#endif

typedef struct
{
  unsigned long long all_input_ids_global_addr;
  unsigned long long next_ids_global_addr;
  int n_input_lengths;
  int n_accept_ids;
  int max_input_length;
  sg_data_type_t dtype;
  char input_lengths[0];
#ifndef WIN32
} __attribute__((packed)) sg_api_tgi_input_ids_update_t;
#else
} sg_api_tgi_input_ids_update_t;
#endif

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long saved_mean_global_addr;
  unsigned long long saved_invstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  int shape[4];
  sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm2d_backward_t;
#else
} sg_api_batchnorm2d_backward_t;
#endif

typedef struct {
    unsigned long long param_global_addr;
    unsigned long long indices_global_addr;
    unsigned long long output_global_addr;
    int param_shape[FW_MAX_SHAPE_DIMS];
    int output_shape[FW_MAX_SHAPE_DIMS];
    int kernel[FW_MAX_SHAPE_DIMS];
    int stride[FW_MAX_SHAPE_DIMS];
    int padding[FW_MAX_SHAPE_DIMS];
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_max_pooling_indices_bwd_t;
#else
}  sg_api_max_pooling_indices_bwd_t;
#endif

typedef struct
{
  unsigned long long io_global_addr;
  unsigned long long indices_global_addr;
  unsigned long long src_global_addr;
  int dim;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int indices_shape[FW_MAX_SHAPE_DIMS];
  int src_shape[FW_MAX_SHAPE_DIMS];
  int input_dims;
  int indices_dims;
  int src_dims;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_scatter_add_t;
#else
} sg_api_scatter_add_t;
#endif

typedef struct
{
    unsigned long long io_global_addr;
    unsigned long long add_global_addr;
    unsigned long long index_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    int axis;
    int index_num;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_index_add_multi_core_t;
#else
} sg_api_index_add_multi_core_t;
#endif

// for test performance
typedef struct {
    /*left mat*/
    int L_C;
    int L_W;
    int L_transposed;
    /*right mat*/
    int R_C;
    int R_W;
    int R_transposed;
#ifndef WIN32
} __attribute__((packed)) sg_api_mm_perf_t;
#else
} sg_api_mm_perf_t;
#endif

typedef struct {
    int N;
    int C;
    int H;
    int W;
    unsigned long long ddr_8ch[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_gdma_perf_t;
#else
} sg_api_gdma_perf_t;
#endif

typedef struct {
    InsType ins_type;
    int loop_times;
    sg_api_mm_perf_t mm_param;
    sg_api_gdma_perf_t gdma_param;
    sg_data_type_t dtype1;
    sg_data_type_t dtype2;
#ifndef WIN32
} __attribute__((packed)) sg_api_test_tpu_perf_t;
#else
} sg_api_test_tpu_perf_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
  int binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_t_v2;
#else
} sg_api_binary_t_v2;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
  int binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_bcast_t_v2;
#else
} sg_api_binary_bcast_t_v2;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
  int binary_type;
  int inversed;
#ifndef WIN32
} __attribute__((packed)) sg_api_binary_c_t;
#else
} sg_api_binary_c_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long scale_global_addr;
  unsigned long long zp_global_addr;
  unsigned long long output_global_addr;
  int final_row_num;
  int inner_num;
  int final_col_num;
  int has_bias;
  int has_zp;
  int q_group_size;
  int weight_dtype;
  int bias_dtype;
  int R_trans;
  int sign;
  int weight_bits;
  int io_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_a16_matmul_t;
#else
} sg_api_a16_matmul_t;
#endif

typedef struct {
  unsigned long long input_addr;
  unsigned long long weight0_addr;
  unsigned long long zp0_addr;
  unsigned long long scale0_addr;
  unsigned long long weight1_addr;
  unsigned long long zp1_addr;
  unsigned long long scale1_addr;
  unsigned long long weight2_addr;
  unsigned long long zp2_addr;
  unsigned long long scale2_addr;
  unsigned long long output_addr;
  unsigned long long silu_addr;
  unsigned long long sigmoid_addr;
  unsigned long long m0_addr;
  int save_mid_res;
  int batch;
  int input_w;
  int middle_w;
  int dtype;
  int quantized;
  int group_size;
  int weight_bits;
#ifndef WIN32
} __attribute__((packed)) sg_api_llama_mlp_multi_core_t;
#else
} sg_api_llama_mlp_multi_core_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long weight0_addr;
    unsigned long long scale0_addr;
    unsigned long long weight1_addr;
    unsigned long long scale1_addr;
    unsigned long long weight2_addr;
    unsigned long long scale2_addr;
    unsigned long long output_addr;
    int batch;
    int input_w;
    int middle_w;
    int blocksize;
    int input_dtype;
    int weight_dtype;
    int quantized;
  #ifndef WIN32
  } __attribute__((packed)) sg_api_mlp_w8a16_dq_multi_core_t;
  #else
  } sg_api_deepseek_mlp_multi_core_t;
  #endif

typedef struct
{
  unsigned long long left_global_addr;
  unsigned long long right_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int                L_shape[FW_MAX_SHAPE_DIMS];
  int                R_shape[FW_MAX_SHAPE_DIMS];
  int                L_dims;
  int                R_dims;
  int                L_trans;
  int                R_trans;
  int                in_dtype;
  int                out_dtype;
  int                slice_core_m;
  int                slice_core_n;
  int                slice_m;
  int                slice_n;
  int                slice_k;
  int                slyt_num;
  int                left_slyt_fmt; // 0:vertial, 1:horizontal
  int                right_slyt_fmt; // 0:vertial, 1:horizontal
  int                result_slyt_fmt; // 0:vertial, 1:horizontal
  int                left_slyt_buf_size;
  int                right_slyt_buf_size;
  int                result_slyt_buf_size;
  unsigned long long left_slyt_global_addr[8];
  unsigned long long right_slyt_global_addr[8];
  unsigned long long result_slyt_global_addr[8];
#ifndef WIN32
} __attribute__((packed)) sg_api_matmul_multi_core_v2_t;
#else
} sg_api_matmul_multi_core_v2_t;
#endif

#pragma pack(pop)
#endif  // SG_API_STRUCT_H
