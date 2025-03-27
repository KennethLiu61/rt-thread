#ifndef STRIDESLICE_UTIL_H
#define STRIDESLICE_UTIL_H
#include "macros.h"
#include "string.h"

static void get_strideslice_output_shape(
    const int* in_shape,
    int* out_shape,
    int dims,
    const int* begin_index_orig,
    const int* end_index_orig,
    const int* strides_orig,
    int begin_mask,
    int end_mask,
    int* begin_index,
    int* end_index,
    int* strides)
{
    for (int i = 0; i < dims; ++i) {
        ASSERT(in_shape[i] >= 0);
        ASSERT(strides_orig[i] > 0);
    }

    #define MASK_BIT(x) (1 << (x))
    #ifndef DIV_UP
    #define DIV_UP(a, b) ((a) == 0 ? 0 : ((a) - 1) / (b) + 1)
    #endif
    #ifndef MIN
    #define MIN(a, b) ((a < b) ? (a) : (b))
    #endif
    for (int i = 0; i < dims; i ++) {
        int begin, end;
        if (!(begin_mask & MASK_BIT(i))) {
            if(begin_index_orig[i] < 0) {
                begin = begin_index_orig[i] + in_shape[i];
            } else {
                begin = begin_index_orig[i];
            }
        } else {
            begin = 0;
        }
        begin_index[i] = begin;

        if (!(end_mask & MASK_BIT(i))) {
            if (end_index_orig[i] <= 0) {
                end = end_index_orig[i] + in_shape[i];
            } else {
                end = end_index_orig[i];
            }
        } else {
            end = in_shape[i];
        }
        end_index[i] = end;

        ASSERT(end >= begin);
    }

    for (int i = 0; i < dims; ++i) {
        ASSERT(end_index[i] >= begin_index[i]);
        out_shape[i] = DIV_UP(end_index[i] - begin_index[i], strides_orig[i]);
    }

    // try to simplify stride to 1
    memcpy(strides, strides_orig, sizeof(int) * dims);
    for (int i = 0; i < dims; ++i) {
        if (out_shape[i] == 1) {
            strides[i] = 1;
            end_index[i] = begin_index[i] + 1;
        }
    }
}

#endif