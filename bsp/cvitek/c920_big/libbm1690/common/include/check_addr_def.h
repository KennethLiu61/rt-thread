#ifndef _CHECK_ADDR_DEF_H_
#define _CHECK_ADDR_DEF_H_

#ifdef __cplusplus
extern "C" {
#endif

enum checkResult
{
    RESULT_SUCCESS = 0,
    RESULT_DDR_OVERFLOW = 1,
    RESULT_STRIDE_ERR = 2,
    RESULT_INSTRUCTION_ERR = 3,
    RESULT_L2MEM_OVERFLOW = 4,
    RESULT_LMEM_OVERFLOW = 5,
    RESULT_MEM_UNKNOW = 6,
};

enum AddrMode
{
    BASIC = 0,
    IO_ALONE = 1,
    IO_TAG = 2,
    IO_TAG_FUSE = 3,
};

typedef struct cmd_check_param
{
    uint64_t neuron_base_addr;
    uint64_t neuron_size;
    uint64_t coeff_base_addr;
    uint64_t coeff_size;
    uint64_t io_addr;
    uint64_t io_size;
    int      addr_mode;
} cmd_check_param_t;

#ifdef __cplusplus
}
#endif
#endif