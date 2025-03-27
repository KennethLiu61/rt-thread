#ifndef DEPTH2SPACE_UTIL_H
#define DEPTH2SPACE_UTIL_H

static inline void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void get_depth2space_output_shape(
    const int in_shape[4],
    int out_shape[4],
    const int block_sizes[2],
    bool in_is_nchw,
    bool out_is_nchw,
    bool is_inversed,
    bool is_crd_mode,
    bool swap_cr)
{
    int bh = block_sizes[0];
    int bw = block_sizes[1];
    int oc, oh, ow;
    if (!is_inversed) {
        if (in_is_nchw) {
            oc = in_shape[1]/(bh*bw);
            oh = in_shape[2]*bh;
            ow = in_shape[3]*bw;
        } else {
            oh = in_shape[1]*bh;
            ow = in_shape[2]*bw;
            oc = in_shape[3]/(bh*bw);
        }
    } else {
        if (in_is_nchw) {
            oc = in_shape[1]*(bh*bw);
            oh = in_shape[2]/bh;
            ow = in_shape[3]/bw;
        } else {
            oh = in_shape[1]/bh;
            ow = in_shape[2]/bw;
            oc = in_shape[3]*(bh*bw);
        }
    }
    out_shape[0] = in_shape[0];
    if(out_is_nchw){
        out_shape[1] = oc;
        out_shape[2] = oh;
        out_shape[3] = ow;
    } else {
        out_shape[1] = oh;
        out_shape[2] = ow;
        out_shape[3] = oc;
    }
}

#endif