#include "tpu_kernel.h"

/**
 * Created by @shunrong.qian
 *
 * This file supplies a template for constructing
 *   a LOAD-COMPUTE-STORE pipeline.
 * Given a code reading as:
 * ```c
 *   for (...; ...; ...) {
 *      load(...);
 *      compute(...);
 *      store(...);
 *   }
 * ```
 * where
 *   `load` func only contains gdma operations,
 *   `store` func only contains gdma operations,
 *   `compute` func does NOT contain any gdma operations.
 * Then the above code could be written as the following
 *   pipeline form:
 * ```c
 *   PIPELINE_PINGPONG_ARRAY_DECLARE(...)
 *   PIPELINE_ARRAY_DECLARE(...)
 *   ...
 *   PIPELINE_BEGIN(idxes, secs)
 *       PIPELINE_PROC_BEGIN
 *           PIPELINE_STORE_BEGIN
 *               store($STAGE, $PINGPONG, ...)
 *           PIPELINE_STORE_END
 *           PIPELINE_COMPUTE_BEGIN
 *               compute($STAGE, $PINGPONG, ...)
 *           PIPELINE_COMPUTE_END
 *           PIPELINE_LOAD_BEGIN
 *               load($STAGE, $PINGPONG, ...)
 *           PIPELINE_LOAD_END
 *       PIPELINE_PROC_END
 *       ...
 *       PIPELINE_TRANSITION_BEGIN
 *           PIPELINE_ARRAY_INC(...)
 *           ...
 *       PIPELINE_TRANSITION_END
 *   PIPELINE_END
 * ```
 **/

typedef enum {
    PIPELINE_LOAD = 0,
    PIPELINE_COMPUTE = 1,
    PIPELINE_STORE = 2,
} PIPELINE_STAGE;

#define PIPELINE_PINGPONG_ARRAY_DECLARE(array, start_addr, mem_size) \
    const local_addr_t array[2] = {start_addr, start_addr + mem_size};
#define PIPELINE_ARRAY_DECLARE(type, array) \
    type array[3] = {0};
#define PIPELINE_BEGIN(idxes, idx_max)         \
    int stage_idx = 0, draning_idx = 0;        \
    while (idxes[PIPELINE_STORE] < idx_max) {
#define PIPELINE_END }

#define PIPELINE_PROC_BEGIN \
    tpu_parallel_start();
#define PIPELINE_PROC_END \
    tpu_parallel_end();

#define PIPELINE_LOAD_BEGIN                  \
    if (draning_idx < 1) {                   \
        const int $STAGE = PIPELINE_LOAD;    \
        const int $PINGPONG = stage_idx & 1; \
        UNUSED($STAGE); UNUSED($PINGPONG);
#define PIPELINE_LOAD_END }

#define PIPELINE_COMPUTE_BEGIN                     \
    if (stage_idx > 0 && draning_idx < 2) {        \
        const int $STAGE = PIPELINE_COMPUTE;       \
        const int $PINGPONG = (stage_idx - 1) & 1; \
        UNUSED($STAGE); UNUSED($PINGPONG);
#define PIPELINE_COMPUTE_END }

#define PIPELINE_STORE_BEGIN                 \
    if (stage_idx > 1) {                     \
        const int $STAGE = PIPELINE_STORE;   \
        const int $PINGPONG = stage_idx & 1; \
        UNUSED($STAGE); UNUSED($PINGPONG);
#define PIPELINE_STORE_END }

#define PIPELINE_ARRAY_MOVE(array)  \
    for (int i = 2; i > 0; i--) {   \
        array[i] = array[i - 1];    \
    }
#define PIPELINE_ARRAY_INC(idxes, inc, idx_max)  \
    idxes[0] += inc;                             \
    if (idxes[0] < idx_max) continue;

#define PIPELINE_TRANSITION_BEGIN   \
    ++stage_idx;                    \
    if (draning_idx < 1) {
#define PIPELINE_TRANSITION_END     \
    }                               \
    ++draning_idx;
