#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

typedef enum {
  NO_NEED_BCAST,  // both of inputs do not need to broadcast
  UNI_NEED_BCAST, // one of inputs do not need to broadcast
  DBL_NEED_BCAST, // both of inputs do not need to broadcast
  NOT_BCASTABLE,  // both of inputs are not broadcastable
} bcast_type_t;

/**
 * @author: shunrong.qian
 * @brief: use one pass algorithm to get broadcast type of two inputs
 **/
int get_bcast_type(
  const int* shape1, int dims1,
  const int* shape2, int dims2,
  bool from_beg) {
  int flag = 0;
  const int min_dims = MIN(dims1, dims2);
  const int max_dims = MAX(dims1, dims2);
  for (int i = 0; i < min_dims; ++i) {
    const int i1 = from_beg ? i : dims1 - 1 - i;
    const int i2 = from_beg ? i : dims2 - 1 - i;
    if (shape1[i1] == shape2[i2]) {
      ;
    } else if (shape1[i1] == 1) {
      flag |= 1;
    } else if (shape2[i2] == 1) {
      flag |= 2;
    } else {
      return NOT_BCASTABLE;
    }
  }
  const int* shape = dims1 > dims2 ? shape1 : shape2;
  const int f = dims1 < dims2 ? 1 : 2;
  for (int i = min_dims; i < max_dims; ++i) {
    const int j = from_beg ? i : max_dims - 1 - i;
    if (shape[j] > 1) {
      flag |= f;
    }
  }
  switch (flag) {
    case 1:
    case 2:
      return UNI_NEED_BCAST;
    case 3:
      return DBL_NEED_BCAST;
    default:
      return NO_NEED_BCAST;
  }
}

/**
 * @author: shunrong.qian
 * @brief: use one pass algorithm to get compressed shape of two inputs
 *          in bcastable sense, e.g.
 *           provided that shape1=[1,2,3,4], shape2=[2,1,3,4,1], from_beg=true,
 *           result is compr_shape1=[1,2,12],compr_shape1=[2,1,12]
 *
 **/
int get_bcast_compressed_shape(
  const int* shape1, int dims1,
  const int* shape2, int dims2,
  int* compr_shape1,
  int* compr_shape2,
  bool from_beg) {
  const int min_dims = MIN(dims1, dims2);
  const int max_dims = MAX(dims1, dims2);
  const int more_dims = max_dims - min_dims;
  int dims = -1;
  int pre_flag = -1;
  if (!from_beg && max_dims != min_dims) {
    const int* max_shape = dims1 > dims2 ? shape1 : shape2;
    int* max_compr_shape = dims1 > dims2 ? compr_shape1 : compr_shape2;
    int* min_compr_shape = dims1 < dims2 ? compr_shape1 : compr_shape2;
    int num_remain = 1;
    for (int i = 0; i < more_dims; ++i) {
      num_remain *= max_shape[i];
    }
    ++dims;
    max_compr_shape[dims] = num_remain;
    min_compr_shape[dims] = 1;
  }
  const int* _shape1 = shape1;
  const int* _shape2 = shape2;
  if (!from_beg) {
    if (dims1 > dims2) {
      _shape1 += more_dims;
    } else {
      _shape2 += more_dims;
    }
  }
  for (int i = 0; i < min_dims; ++i) {
    int cur_flag = -1;
    if (_shape1[i] == _shape2[i]) {
      cur_flag = 0;
    } else if (_shape1[i] == 1) {
      cur_flag = 1;
    } else if (_shape2[i] == 1) {
      cur_flag = 2;
    } else {
      return -1;
    }
    if (cur_flag != pre_flag) {
      pre_flag = cur_flag;
      ++dims;
      compr_shape1[dims] = _shape1[i];
      compr_shape2[dims] = _shape2[i];
    } else {
      compr_shape1[dims] *= _shape1[i];
      compr_shape2[dims] *= _shape2[i];
    }
  }
  if (from_beg && max_dims != min_dims) {
    const int* max_shape = dims1 > dims2 ? shape1 : shape2;
    int* max_compr_shape = dims1 > dims2 ? compr_shape1 : compr_shape2;
    int* min_compr_shape = dims1 < dims2 ? compr_shape1 : compr_shape2;
    int num_remain = 1;
    for (int i = min_dims; i < max_dims; ++i) {
      num_remain *= max_shape[i];
    }
    const int f = dims1 < dims2 ? 1 : 2;
    const int cur_flag = num_remain == 1 ? 0 : f;
    if (cur_flag != pre_flag) {
      ++dims;
      max_compr_shape[dims] = num_remain;
      min_compr_shape[dims] = 1;
    } else {
      max_compr_shape[dims] *= num_remain;
    }
  }
  return dims + 1;
}