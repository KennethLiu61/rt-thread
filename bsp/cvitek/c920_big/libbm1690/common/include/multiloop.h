#ifndef MULTILOOP_H
#define MULTILOOP_H

static bool multiloop_incr_idxes_by_1(int* idxes, const int* fulls, int rank) {
    for (int j = rank - 1; j >= 0; --j) {
        idxes[j] += 1;
        if (idxes[j] < fulls[j])
            return true;
        if (j == 0) return false;
        idxes[j] = 0;
    }
    return false;
}

static bool multiloop_incr_idxes(int* idxes, const int* slices,
    const int* fulls, int rank) {
    for (int j = rank - 1; j >= 0; --j) {
        idxes[j] += slices[j];
        if (idxes[j] < fulls[j])
            return true;
        if (j == 0) return false;
        idxes[j] = 0;
    }
    return false;
}

#endif