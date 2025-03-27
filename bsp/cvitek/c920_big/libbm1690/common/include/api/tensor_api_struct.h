#pragma once

#include "sg_api_struct.h"

#pragma pack(push, 1)

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif
#ifndef FW_MAX_CONCAT_NUM
#define FW_MAX_CONCAT_NUM     16
#endif

#ifndef WIN32
#define WITH_PLATFORM(x) __attribute__ ((packed)) x
#else
#define WITH_PLATFORM(x) x
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
} WITH_PLATFORM(sg_api_binary_t);

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
} WITH_PLATFORM(sg_api_binary_bcast_t);

typedef struct
{
  unsigned long long left_global_addr;
  unsigned long long right_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int batch;
  int left_row;
  int left_column;
  int right_column;
  int is_left_transposed;
  int is_right_transposed;
  int dtype;
} WITH_PLATFORM(tensor_api_batch_matmul_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(sg_api_strided_copy_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long src_global_addr;
  unsigned long long indices_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int src_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_slice_scatter_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(tensor_api_softmax_t);

typedef struct
{
  unsigned long long output_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(tensor_api_softmax_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(tensor_api_log_softmax_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  tensor_log_type_t log_type; // 0 for log, 1 for log1p, 2 for log2, 10 for log10
} WITH_PLATFORM(tesnor_api_log_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int tile_axis;
  int tile_num;
  int dtype;
} WITH_PLATFORM(tensor_api_squeeze_t);

typedef struct
{
  unsigned long long output_addr;
  unsigned long long cond_addr;
  unsigned long long self_addr;
  unsigned long long other_addr;
  int out_shape[FW_MAX_SHAPE_DIMS];
  int cond_shape[FW_MAX_SHAPE_DIMS];
  int self_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int cond_dtype;
  int dtype;
  int self_is_scalar;
  int other_is_scalar;
  float self_val;
  float other_val;
} WITH_PLATFORM(tensor_api_where_multi_core_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long buffer_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_norm2_multi_core_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_neg_t);

typedef struct
{
  int start;
  int end;
  int step;
  unsigned long long output_global_addr;
  int dtype;
  int isint64;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];
} WITH_PLATFORM(tensor_api_arange_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int repeat_times[FW_MAX_SHAPE_DIMS];
  int dim;
  int repeat_dim;
  int dtype;
} WITH_PLATFORM(tensor_api_repeat_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(tensor_api_strided_copy_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_dtype;
  int output_dtype;
  int is_bool;
} WITH_PLATFORM(tensor_api_dtype_convert_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long index_global_addr;
  unsigned long long num_global_addr;
  unsigned long long num_buffer_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_nonzero_multi_core_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  float min;
  float max;
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_clamp_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int group_num;
  float eps;
  int affine;
  int dtype;
} WITH_PLATFORM(tensor_api_native_group_norm_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(tensor_api_addcdiv_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int tensor1_shape[FW_MAX_SHAPE_DIMS];
  int tensor2_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int tensor1_dim;
  int tensor2_dim;
  float value;
  int dtype;
} WITH_PLATFORM(tensor_api_bcast_addcmul_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(tensor_api_addcmul_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_bitwise_not_t);


typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_cbrt_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_logical_and_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_logical_not_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_logical_or_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long target_global_addr;
  unsigned long long output_global_addr;
  int batch;
  int class_;
  int reduction;
  float label_smoothing;
  int dtype;
  int is_target_int64;
} WITH_PLATFORM(tensor_api_cross_entropy_loss_t);

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
} WITH_PLATFORM(tensor_api_cross_entropy_loss_backward_t);
typedef struct
{
  unsigned long long input1_global_addr;
  unsigned long long input2_global_addr;
  unsigned long long output_global_addr;
  int reduction;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_mse_loss_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int dtype;
  int do_relu;
  TPUDNN_PoolingDescriptor_t pooling_desc;
} WITH_PLATFORM(tensor_api_pooling_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mask_global_addr;
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
  int dtype;
  TPUDNN_PoolingDescriptor_t pooling_desc;
} WITH_PLATFORM(tensor_api_max_pooling_with_mask_t);

typedef struct
{
  unsigned long long param_global_addr;
  unsigned long long indices_global_addr;
  unsigned long long output_global_addr;
  int param_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int kernel[FW_MAX_SHAPE_DIMS];
  int stride[FW_MAX_SHAPE_DIMS];
  int padding[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(tensor_api_max_pooling_indices_bwd_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int reduction_dim[FW_MAX_SHAPE_DIMS];
  int reduction_dim_length;
  int dim;
  int mode;
  int dtype;
} WITH_PLATFORM(tensor_api_reduce_max_or_min_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long mul_global_addr;
  unsigned long long sum_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int reduce_list[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int output_dim;
  int reduce_dim;
  int correction;
  int keepdim;
  int dtype;
} WITH_PLATFORM(tensor_api_reduce_var_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int start_dim;
  int end_dim;
  int dtype;
  int mode;
} WITH_PLATFORM(tensor_api_reduce_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(tensor_api_reduce_prod_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long index_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int index_num;
  int axis;
  int dtype;
  int is_index_int64;
} WITH_PLATFORM(tensor_api_index_select_t);

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long index_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long sorted_index_global_addr;
  unsigned long long sorted_index_index_global_addr;
  unsigned long long from_index_global_addr;
  unsigned long long to_index_global_addr;
  int grad_output_shape[FW_MAX_SHAPE_DIMS];
  int grad_output_dim;
  int index_shape[FW_MAX_SHAPE_DIMS];
  int index_dim;
  int grad_input_shape[FW_MAX_SHAPE_DIMS];
  int grad_input_dim;
  int window_size;
  int grad_output_dtype;
  int is_index_int64;
} WITH_PLATFORM(tensor_api_embedding_backward_t);

typedef struct
{
  unsigned long long input_global_addrs[FW_MAX_CONCAT_NUM];
  unsigned long long output_global_addr;
  int input_shapes[FW_MAX_CONCAT_NUM][FW_MAX_SHAPE_DIMS];
  int dim;
  int input_num;
  int axis;
  int dtype;
} WITH_PLATFORM(tensor_api_concat_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long index_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int index_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
  int is_index_int64;
} WITH_PLATFORM(tensor_api_gather_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long values_global_addr;
  unsigned long long indices_global_addr;
  unsigned long long buffer_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int mode;
  int dtype;
} WITH_PLATFORM(tensor_api_reduce_arg_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long value_global_addr;
  unsigned long long index_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int k;
  int axis;
  int largest;
  int sorted;
  int dtype;
} WITH_PLATFORM(tensor_api_topk_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_relu_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_gelu_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_gelu_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float negative_slope;
  int dtype;
} WITH_PLATFORM(tensor_api_leakyrelu_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float min_value;
  float max_value;
} WITH_PLATFORM(tensor_api_hardtanh_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  tensor_active_type_t active_type;
} WITH_PLATFORM(tensor_api_active_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(tensor_api_real_t);

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
  int dtype;
} WITH_PLATFORM(tensor_api_batchnorm2d_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long running_mean_global_addr;
  unsigned long long running_var_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long saved_mean_global_addr;
  unsigned long long saved_invstd_global_addr;
  unsigned long long output_global_addr;
  int shape[4];
  float momentum;
  float eps;
  int dtype;
} WITH_PLATFORM(tensor_api_batchnorm2d_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  float eps;
  int dtype;
} WITH_PLATFORM(tensor_api_layernorm_t);

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
  int requires_grad_input;
} WITH_PLATFORM(tensor_api_layernorm_backward_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_signbit_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[4];
  int mode;
} WITH_PLATFORM(tensor_api_conv_weight_reorder_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];
  int stride[2];
  int dilation[2];
  int pad[4];
  int dtype;
} WITH_PLATFORM(tensor_api_conv2d_t);

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
  int dtype;
  int weight_formated;
} WITH_PLATFORM(tensor_api_conv2d_backward_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long buffer_addr;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];
  int out_shape[FW_MAX_SHAPE_DIMS];
  int pad_bag;
  int pad_end;
  int half_pixel_centers;
  int align_corners;
  int if_getting_buffer_size;
  int dtype;
  unsigned long long* buffer_size_ptr;
  // PYTORCH_SUPPORT for **bilinear**; PYTORCH_NEAREST for nearest
  PLATFORM_SUPPORT platform_sp;
} WITH_PLATFORM(tensor_api_upsampling2d_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int dtype;
  int do_relu;
  TPUDNN_PoolingDescriptor_t pooling_desc;
  int scalar;
} WITH_PLATFORM(tensor_api_upsample2d_backward_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int axis;
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_flip_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int is_upper;
  int diagonal;
  int dtype;
#ifndef WIN32
} __attribute__((packed)) tensor_api_triangularize_t;
#else
} tensor_api_triangularize_t;
#endif

typedef struct
{
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  unsigned int value;
  int dtype;
} WITH_PLATFORM(tensor_api_constant_fill_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long mask_global_addr;
  unsigned long long out_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int mask_shape[FW_MAX_SHAPE_DIMS];
  int input_dims;
  int mask_dims;
  float value;
  int dtype;
} WITH_PLATFORM(tensor_api_masked_fill_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long found_inf_global_addr;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];
  float inv_scale;
  int idtype;
  int found_inf_dtype;
} WITH_PLATFORM(tensor_api_inf_check_unscale_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long found_inf_global_addr;
  unsigned long long found_inf_buffer_global_addr;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];
  float inv_scale;
  int idtype;
  int found_inf_dtype;
  int need_clear_found_inf;
} WITH_PLATFORM(tensor_api_inf_check_unscale_multi_core_t);

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
    int dtype;
} WITH_PLATFORM(tensor_api_adam_backward_multi_core_t);

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
  int dtype;
} WITH_PLATFORM(tensor_api_cross_entropy_multi_core_t);

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
} WITH_PLATFORM(tensor_api_llama_mlp_multi_core_t);

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
} WITH_PLATFORM(tensor_api_deepseek_mlp_multi_core_t);

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
  int in_dtype;
  int out_dtype;
  int bias_dtype;
  int has_bias;
  int use_fast;
} WITH_PLATFORM(tensor_api_llava_mlp_multi_core_t);

typedef struct
{
    unsigned long long input_addr;
    unsigned long long output_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    float drop_rate;
    int dtype;
} WITH_PLATFORM(tensor_api_dropout_multi_core_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_tanh_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_sigmoid_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(tensor_api_silu_backward_t);

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
} WITH_PLATFORM(tensor_api_a16_matmul_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int in_dim;
  int other_shape[FW_MAX_SHAPE_DIMS];
  int other_dim;
  float value;
  int dtype;
  int binary_type;
  int input_format;
  int other_format;
} WITH_PLATFORM(sg_api_weight_update_t);

#pragma pack(pop)
