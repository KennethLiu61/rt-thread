#ifndef ARG_UTIL_H
#define ARG_UTIL_H
#include <vector>
#include <functional>
#include <numeric>

typedef enum {
    ARG_MAX,
    ARG_MIN,
} arg_mode_t;

template<typename T>
std::function<bool(T, T)> get_compare_op(arg_mode_t mode, bool select_last) {
    if (select_last) {
        if (mode == ARG_MAX) {
            return std::greater_equal<T>();
        } else {
            return std::less_equal<T>();
        }
    } else {
        if (mode == ARG_MAX) {
            return std::greater<T>();
        } else {
            return std::less<T>();
        }
    }
}

template <typename T>
void native_arg(
    const T* in,
    void* out_idx,
    T* out_val,
    const std::vector<int>& input_shape,
    int axis,
    arg_mode_t method,
    bool is_index_int32,
    bool select_last_index,
    bool need_val)
{
    const int input_dims = (int)input_shape.size();
    if (axis < 0) {
        axis += input_dims;
    }
    int outer_dims =
        std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                        std::multiplies<int64_t>());
    int inner_dims =
        std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1,
                        std::multiplies<int64_t>());
    int axis_dims = input_shape[axis];
    const auto cmp_op = get_compare_op<T>(method, select_last_index);
    int num_iter = outer_dims * inner_dims;
    for (int n = 0; n < num_iter; n++) {
        const int o = n / inner_dims;
        const int i = n % inner_dims;
        const T* in_n = in + o * axis_dims * inner_dims + i;
        int target_idx = 0;
        T target_val = in_n[0];
        for (int a = 1; a < axis_dims; a++) {
            const auto v = in_n[a * inner_dims];
            if (cmp_op(v, target_val)) {
                target_val = v;
                target_idx = a;
            }
        }
        if (is_index_int32)
            *((int*)out_idx + n) = target_idx;
        else
            *((float*)out_idx + n) = target_idx;
        if (need_val) {
            out_val[n] = target_val;
        }
    }
}

#endif