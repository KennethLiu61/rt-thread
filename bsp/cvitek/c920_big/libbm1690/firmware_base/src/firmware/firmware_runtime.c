#include "sg_api_struct.h"
#include "firmware_common.h"
#include "firmware_runtime.h"
#include "firmware_timer.h"
#include "message.h"
#include <setjmp.h>
#include "atomic_sys_gen_cmd.h"
#include "tpu_kernel.h"
#ifdef USING_CMODEL
#include "cmodel_common.h"
#include "cmodel_runtime.h"
#include "pthread.h"
#include <sys/prctl.h>
#include "stddef.h"
pthread_mutex_t msg_done_mutex[MAX_TPU_CORE_NUM];
pthread_cond_t msg_done_cond[MAX_TPU_CORE_NUM];
u32 msg_done[MAX_TPU_CORE_NUM];

static pthread_mutex_t api_msg_mutex[MAX_TPU_CORE_NUM];
static pthread_cond_t api_msg_cond[MAX_TPU_CORE_NUM];
#endif
#ifdef USING_PLD_TEST
#include "nodechip_pld_test.h"
#endif
#if defined(USING_PLD_TEST) && !defined(USING_CMODEL)
#include "firmware_pmu.h"
void timer_udelay(uint32_t us);
#endif
#include "firmware_pmu.h"

void firmware_lock(int index) {
    ASSERT(index < 16);
#if !defined(USING_CMODEL) && !defined(USING_EDA)
    while (READ_REG(TPU_SYS_DEVICE_LOCK + index * 4))
    {
#if USING_PLD_TEST
        timer_udelay(1);
#endif
    }

#endif
}

void firmware_unlock(int index) {
    ASSERT(index < 16);
#if !defined(USING_CMODEL) && !defined(USING_EDA)
    WRITE_REG(TPU_SYS_DEVICE_LOCK + index * 4, 0, NODECHIP_REG);
#endif
}

sg_fw_status_t __attribute__((weak)) api_route(int api_id, unsigned char* api_buf, int bytes);

#define MAX_MSG_WORD 32768

#define CASE_API(ID, func)          \
  case ID:                          \
    ret = func(api_buf, byte_size); \
    break;

jmp_buf error_stat;

#if defined(USING_CMODEL) || defined(USING_FAKE_DDR_MODE)
INLINE static sg_fw_status_t copy_message_from_sharemem(u32 *dst_msg_buf, u32 *rp, u32 *size, u32 *api_id) {
    CORE_PRINT("I'm in copy_message_from_sharemem!\n");
    u32 cur_rp = READ_SHARE_REG(SHARE_REG_MESSAGE_RP);
    *rp = cur_rp;
    u32 *src_msg_word = GET_SHARE_MEM_ADDR(cur_rp & SHAREMEM_MASK);
    CORE_PRINT("src_msg_word=%p, src_msg_word[0]=0x%x, core id:%d\n", src_msg_word, *src_msg_word, CORE_ID);
    *api_id = *src_msg_word;
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, 1, SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    CORE_PRINT("src_msg_word=%p, src_msg_word[0]=0x%x, \n", src_msg_word, *src_msg_word);
    *size = *src_msg_word;
    if (*api_id == SG_API_QUIT)
        return SG_FW_SUCCESS;
    if (!((*size + 2) < MAX_MSG_WORD)) {
        FW_ERR("bmfw: api size  = 0x%x is too large than max size!\n", (*size + 2));
        return SG_FW_ERR_DATA;
    }
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, 2, SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    CORE_PRINT("src_msg_word=%p, src_msg_word[0]=0x%x, \n", src_msg_word, *src_msg_word);
    u32 left = (1 << SHAREMEM_SIZE_BIT) - cur_rp - 2;
    if (*size <= left) {
        memcpy(dst_msg_buf, src_msg_word, *size * sizeof(u32));
    } else {
        memcpy(dst_msg_buf, src_msg_word, left * sizeof(u32));
        memcpy(dst_msg_buf + left, GET_SHARE_MEM_ADDR(0), (*size - left) * sizeof(u32));
    }
    dst_msg_buf += (*size);

    return SG_FW_SUCCESS;
}
#else
// #define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
INLINE static sg_fw_status_t copy_message_from_sharemem(u32 *dst_msg_buf, u32 *rp, sg_kapi_header_t *api_header) {
    u32 cur_rp = READ_SHARE_REG(SHARE_REG_MESSAGE_RP);
    *rp = cur_rp;
    u32 *src_msg_word = GET_SHARE_MEM_ADDR(cur_rp & SHAREMEM_MASK);
    api_header->api_id= *src_msg_word;
    if (api_header->api_id == SG_API_QUIT)
        return SG_FW_SUCCESS;
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, offsetof(sg_kapi_header_t, api_size)/sizeof(u32), SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    api_header->api_size = *src_msg_word;
    if(!((api_header->api_size + sizeof(sg_kapi_header_t)/sizeof(u32)) < MAX_MSG_WORD)) {
        FW_ERR("bmfw: api size  = %x is too large than max size!\n", (u32)(api_header->api_size + sizeof(sg_kapi_header_t)/sizeof(u32)));
        return SG_FW_ERR_DATA;
    }
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, offsetof(sg_kapi_header_t, api_handle) / sizeof(u32),
                                      SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    api_header->api_handle = *src_msg_word;
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, offsetof(sg_kapi_header_t, api_handle) / sizeof(u32) + 1,
                                      SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    api_header->api_handle |= (u64)(*src_msg_word) << 32;
    src_msg_word = GET_SHARE_MEM_ADDR(pointer_wrap_around(cur_rp, sizeof(sg_kapi_header_t) / sizeof(u32), SHAREMEM_SIZE_BIT) & SHAREMEM_MASK);
    u32 left = (1 << SHAREMEM_SIZE_BIT) - cur_rp - sizeof(sg_kapi_header_t) / sizeof(u32);
    if (api_header->api_size <= left) {
        memcpy(dst_msg_buf, src_msg_word, api_header->api_size * sizeof(u32));
    } else {
        memcpy(dst_msg_buf, src_msg_word, left * sizeof(u32));
        memcpy(dst_msg_buf + left, GET_SHARE_MEM_ADDR(0), (api_header->api_size - left) * sizeof(u32));
    }
    dst_msg_buf += (api_header->api_size);
    return SG_FW_SUCCESS;
}
#endif

//#include "spi.h"
#define CONFIG_FLASH_ADDR 0x6000000
#define CONFIG_FLASH_BIN_SIZE 0xc8000

INLINE static sg_fw_status_t bm_api_flash_update(
    firmware_runtime_param_t  *param,
    unsigned char             *api_buf,
    int                        size) {
    UNUSED(param);
    UNUSED(api_buf);
    UNUSED(size);
    /*
    bm_spi_init(SPIF_CTRL_BASE_ADDR);
    bm_spi_flash_program((u8 *)0x80f00000,
                         CONFIG_FLASH_ADDR,
                         CONFIG_FLASH_BIN_SIZE
                        );
    */
    return SG_FW_SUCCESS;
}

INLINE static sg_fw_status_t fw_handle_api_msg(firmware_runtime_param_t *param) {
    CORE_PRINT("I'm in fw_handle_api_msg!\n");
    volatile sg_fw_status_t ret = SG_FW_SUCCESS;
    u32 api_id;
    u32 byte_size;
    u32 cur_rp = 0;
    u32 next_rp = 0;
#ifndef USING_FW_SIMULATION
    u8 *api_buf = (u8 *)malloc(MAX_MSG_WORD * sizeof(u32));
    memset(api_buf, 0, MAX_MSG_WORD * sizeof(u32));
#else
    u8  api_buf[MAX_MSG_WORD * sizeof(u32)];
#endif
#if defined(USING_CMODEL) || defined(USING_FAKE_DDR_MODE)
#else
    volatile u32* smem_addr = NULL;
    volatile u64 api_start_time  = 0ULL;
    volatile u64 api_end_time = 0ULL;
#endif
#if defined(USING_CMODEL) || defined(USING_FAKE_DDR_MODE)
    u32 word_size = 0;
    ret = copy_message_from_sharemem((u32 *)api_buf, &cur_rp, &word_size, &api_id);
    if (ret) {
        FW_ERR("bmfw: api = %d copy_message_from_sharemem fail\n", api_id);
        goto api_continue;
    }
    byte_size = word_size * sizeof(u32);
#else
    sg_kapi_header_t api_header;
    memset(&api_header, 0, sizeof(api_header));
    ret = copy_message_from_sharemem( (u32 *)api_buf, &cur_rp, &api_header);
    if (ret) {
        FW_ERR("bmfw: api = %d copy_message_from_sharemem fail\n", api_header.api_id);
        goto api_continue;
    }
    byte_size = api_header.api_size*sizeof(u32);
    api_id = api_header.api_id;
#endif
    CORE_PRINT("api_id is %d\n", api_id);
    CORE_PRINT("byte_size is %d\n", byte_size);
#ifdef SG_STAS_GEN
    sg_stas_info_insert(api_id);
#endif
#if defined(USING_CMODEL) || defined(USING_FAKE_DDR_MODE)
#else
    api_start_time = firmware_timer_get_time_us();
    if (setjmp(error_stat)) {
        FW_ERR("api = %d assert fail\n", api_id);
        ret = SG_FW_ERR_ASSERT;
        goto api_continue;
    }
    if (api_header.api_handle == 0) {
        FW_INFO("bmfw: api = %d handle = 0x%llx is canceled\n", api_id, api_header.api_handle);
        goto api_continue;
    }
#endif

    if (!api_route) {
        ASSERT(0 && "firmware_route is not implemented");
        ret = SG_FW_ERR_NOFEATURE;
    }
#if defined(USING_PLD_TEST) && !defined(USING_CMODEL) && !defined(USING_EDA)
// #if (defined(USING_PLD_TEST) || defined(USING_EDA)) && !defined(USING_CMODEL)
    invalidate_cache(PLD_MESSAGE_START_ADDR, CACHE_LINE_SIZE);
    // configured by SOC
    if (true || READ_REG(PLD_MESSAGE_START_ADDR))
        enable_tpu_perf_monitor();
#endif

    ret = api_route(api_id, api_buf, byte_size);

api_continue:
    CORE_PRINT("finish api, nd %d, api_id %d, result = %s\n", param->nodechip_idx, api_id, ret ? "failed" : "success");
#if defined(USING_CMODEL) || defined(USING_FAKE_DDR_MODE)
    next_rp = pointer_wrap_around(cur_rp, word_size + 2, SHAREMEM_SIZE_BIT);
#else
    api_end_time = firmware_timer_get_time_us();
#ifdef USING_FW_API_PERF
    FW_INFO("finish api, nd %d, api_id %d start time: %lld\n", param->nodechip_idx, api_id, api_start_time);
    FW_INFO("finish api, nd %d, api_id %d end   time: %lld\n", param->nodechip_idx, api_id, api_end_time);
    FW_INFO("finish api, nd %d, api_id %d process time: %lld\n", param->nodechip_idx, api_id, (api_end_time-api_start_time));
#endif
    next_rp = pointer_wrap_around(cur_rp, offsetof(sg_kapi_header_t, result)/sizeof(u32), SHAREMEM_SIZE_BIT) & SHAREMEM_MASK;
    smem_addr = GET_SHARE_MEM_ADDR(next_rp & SHAREMEM_MASK);
    *smem_addr = ret;
    next_rp = pointer_wrap_around(cur_rp, offsetof(sg_kapi_header_t, duration)/sizeof(u32), SHAREMEM_SIZE_BIT) & SHAREMEM_MASK;
    smem_addr = GET_SHARE_MEM_ADDR(next_rp & SHAREMEM_MASK);
    *smem_addr = (u32)(api_end_time - api_start_time);
    next_rp = pointer_wrap_around(cur_rp, api_header.api_size + sizeof(sg_kapi_header_t)/sizeof(u32), SHAREMEM_SIZE_BIT);
#endif
    WRITE_SHARE_REG(SHARE_REG_MESSAGE_RP, next_rp);
    if (api_id == SG_API_QUIT) {
#ifndef USING_FW_SIMULATION
        free(api_buf);
#endif
    CORE_PRINT("I'm out fw_handle_api_msg!\n");
        return ret;
    }
#if defined(USING_INT_MSGFIFO) && !defined(USING_FAKE_DDR_MODE)
    send_msg_done_interrupt();
#endif
#if defined(USING_PLD_TEST) && !defined(USING_CMODEL) && !defined(USING_EDA)
// #if (defined(USING_PLD_TEST) || defined(USING_EDA)) && !defined(USING_CMODEL)
    // configured by SOC
    if (true || READ_REG(PLD_MESSAGE_START_ADDR)) {
        // int gdma_max_num = 20000; // READ_REG(PLD_MESSAGE_START_ADDR + 0x04);
        // int tiu_max_num = 20000; // READ_REG(PLD_MESSAGE_START_ADDR + 0x08);
        // int sdma_max_num = 20000; // READ_REG(PLD_MESSAGE_START_ADDR + 0x08);
        // int cdma_max_num = 20000; // READ_REG(PLD_MESSAGE_START_ADDR + 0x08);

        disable_tpu_perf_monitor();
        // if (gdma_max_num || tiu_max_num) {
        //     show_tpu_perf_data(gdma_max_num, sdma_max_num, tiu_max_num, cdma_max_num);
        // }
    }
#endif
#ifdef USING_CMODEL
    // add conditinal variable to simulate interrupt
    pthread_mutex_lock(&msg_done_mutex[param->nodechip_idx]);
    msg_done[param->nodechip_idx]++;
    pthread_mutex_unlock(&msg_done_mutex[param->nodechip_idx]);
    pthread_cond_signal(&msg_done_cond[param->nodechip_idx]);
#endif
    UNUSED(param);
#ifndef USING_FW_SIMULATION
    free(api_buf);
#endif
    CORE_PRINT("I'm out fw_handle_api_msg!\n");
    return ret;
}

#ifdef USING_CMODEL
void api_poll(int dev)
{
    pthread_mutex_lock(&msg_done_mutex[dev]);
    while (0 == msg_done[dev])
        pthread_cond_wait(
            &msg_done_cond[dev],
            &msg_done_mutex[dev]);
    msg_done[dev]--;
    pthread_mutex_unlock(&msg_done_mutex[dev]);
}

void api_signal_begin(int dev)
{
    pthread_mutex_lock(&api_msg_mutex[dev]);
}

void api_signal(int dev)
{
    pthread_mutex_unlock(&api_msg_mutex[dev]);
    pthread_cond_signal(&api_msg_cond[dev]);
}
#endif

INLINE static bool poll_message() {
    u32 wp = READ_SHARE_REG(SHARE_REG_MESSAGE_WP);
    u32 rp = READ_SHARE_REG(SHARE_REG_MESSAGE_RP);
    if (wp == rp)
        return false;
    else
        return true;
}

INLINE static void print_mem(const void* src, size_t size, const char* info){
    CORE_PRINT("%s: addr=%p, size=%d\n  [ ", info, src, (int)size);
    for(int i=0; i<(int)size; i+=1){
        CORE_PRINT("0x%08x ", ((int*)src)[i]);
    }
    CORE_PRINT("]\n");
}

INLINE static sg_fw_event_t fw_wait_event(firmware_runtime_param_t *param) {
    const int message_max_size=16*1024;
    (void)message_max_size;
#ifdef USING_FAKE_DDR_MODE
    invalidate_cache(PLD_MESSAGE_START_ADDR, message_max_size);
    memcpy(GET_SHARE_MEM_ADDR(0), GET_GLOBAL_ADDR(PLD_MESSAGE_START_ADDR), message_max_size);
#endif
    // check whether there is a message
#if USING_CMODEL
    bool has_message;
    pthread_mutex_lock(&api_msg_mutex[param->nodechip_idx]);
    while (true)
    {
        has_message = poll_message();
        if (has_message) break;
        pthread_cond_wait(
            &api_msg_cond[param->nodechip_idx],
            &api_msg_mutex[param->nodechip_idx]);
    }
    pthread_mutex_unlock(&api_msg_mutex[param->nodechip_idx]);
#else
    bool has_message = poll_message();
#endif
    if (has_message) {
        #if defined(USING_CMODEL) && defined(SG_TV_GEN)
        memcpy(GET_SHARE_MEM_ADDR(0), GET_GLOBAL_ADDR(PLD_MESSAGE_START_ADDR), message_max_size);
        #endif
#ifndef USING_FAKE_DDR_MODE
        u32 rp = READ_SHARE_REG(SHARE_REG_MESSAGE_RP);
        volatile u32 *msg_buf = GET_SHARE_MEM_ADDR(rp & SHAREMEM_MASK);
        // If host notify firmware to quit
        if (msg_buf[0] == SG_API_QUIT)
            param->terminated = 1;
#else
        param->terminated = 1;
#endif
        return SG_FW_EVENT_MSG;
    }
#if defined(USING_FAKE_DDR_MODE) && !defined(USING_CMODEL)
    // For real firmware running on PLD, message should have been prepared in ddr.
    // No message means no case need to be tested, but terminated signal should be set
    param->terminated = 1;
#endif
    return SG_FW_EVENT_NULL;
}

void load_lookup_tables() {
#include "tables.h"
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + EXP_TAYLOR_OFFSET),
           EXP_COEFF,
           sizeof(EXP_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + EXP_FP16_TAYLOR_OFFSET),
           EXP_FP16_COEFF,
           sizeof(EXP_FP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + EXP_BF16_TAYLOR_OFFSET),
           EXP_BF16_COEFF,
           sizeof(EXP_BF16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + LOG_TAYLOR_OFFSET),
           LOG_COEFF,
           sizeof(LOG_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + ERF_TAYLOR_OFFSET),
           ERF_COEFF,
           sizeof(ERF_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + ERF_FP16_TAYLOR_OFFSET),
           ERF_FP16_COEFF,
           sizeof(ERF_FP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + ERF_BF16_TAYLOR_OFFSET),
           ERF_BF16_COEFF,
           sizeof(ERF_BF16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + SERIAL_NUMBER_OFFSET),
           SEQ_COEFF,
           sizeof(SEQ_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + SIN_TAYLOR_OFFSET),
           SIN_COEFF,
           sizeof(SIN_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + COS_TAYLOR_OFFSET),
           COS_COEFF,
           sizeof(COS_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + ARCSIN_TAYLOR_OFFSET),
           ARCSIN_COEFF,
           sizeof(ARCSIN_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + TAN_TAYLOR_OFFSET),
           TAN_COEFF,
           sizeof(TAN_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + LOG_FP16_TAYLOR_OFFSET),
           LOG_FP16_COEFF,
           sizeof(LOG_FP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + LOG_BF16_TAYLOR_OFFSET),
           LOG_BF16_COEFF,
           sizeof(LOG_BF16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + SIN_FP16_TAYLOR_OFFSET),
           SIN_FP16_COEFF,
           sizeof(SIN_FP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + COS_FP16_TAYLOR_OFFSET),
            COS_FP16_COEFF,
            sizeof(COS_FP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + SIN_BFP16_TAYLOR_OFFSET),
           SIN_BFP16_COEFF,
           sizeof(SIN_BFP16_COEFF));
    memcpy(GET_SMEM_ADDR(STATIC_MEM_START_ADDR + COS_BFP16_TAYLOR_OFFSET),
            COS_BFP16_COEFF,
            sizeof(COS_BFP16_COEFF));
}

static void enable_l2m() {
  // enable ai mode
  uint32_t reg = READ_REG(TPU_SYS_L2M_CFG_CTRL);
  WRITE_REG(TPU_SYS_L2M_CFG_CTRL, reg | 0x1, NODECHIP_REG);
  reg = READ_REG(TPU_SYS_HNF_L2M_CTRL);
  WRITE_REG(TPU_SYS_HNF_L2M_CTRL, reg | 0xe, NODECHIP_REG);
#ifdef USING_FAKE_DDR_MODE
  for (int i = 0; i < MAX_TPU_CORE_NUM; ++i) {
    u32 clk = 0;
    clk = READ_REG(0x6908050000UL + i * CORE_OFFSET);
    WRITE_REG(0x6908050000UL + i * CORE_OFFSET, (clk | (0x7f | (1 << 20))),
              NODECHIP_REG);
    u32 reset = 0;
    reset = READ_REG(0x6908050000UL + 0x4 + i * CORE_OFFSET);
    WRITE_REG(0x6908050000UL + 0x4 + i * CORE_OFFSET, (reset & 0x0),
              NODECHIP_REG);
    reg = READ_REG(0x6908050000UL + i * CORE_OFFSET + 0x260);
    WRITE_REG(0x6908050000UL + i * CORE_OFFSET + 0x260, reg | 0x1,
              NODECHIP_REG);
    reg = READ_REG(0x6908050000UL + i * CORE_OFFSET + 0x27c);
    WRITE_REG(0x6908050000UL + i * CORE_OFFSET + 0x27c, reg | 0xe,
              NODECHIP_REG);
  }
//   firmware_lock(0);
//   printf("[%d] core %d: I'm in firmware_main!\n", CORE_ID,
//          READ_REG(CLINT_REG_MHART_ID));
//   firmware_unlock(0);
#endif
}
static void set_base_addr() {
  {
    // set local mem and smem base ddr
    int base_idx[] = {LMEM_TAG};
    u64 base_addr[] = {LOCAL_MEM_START_ADDR};
    atomic_set_base_ddr(base_idx, base_addr, 1, ENGINE_GDMA);
    int start_phy_core = tpu_start_physical_core_id();
    int l2_idx[] = {L2M_TAG};
    u64 l2_addr[] = {0x1000000 * start_phy_core};
    atomic_set_base_ddr(l2_idx, l2_addr, 1, ENGINE_GDMA);
    atomic_set_base_ddr(l2_idx, l2_addr, 1, ENGINE_SDMA);
  }

#ifdef USING_FAKE_DDR_MODE
  {
    int base_idx[] = {0, LMEM_TAG};
    u64 _base_addr = PLD_BASE_ADDR + (GLOBAL_MEM_START_ADDR - 0x0);
    u64 base_addr[] = {_base_addr, LOCAL_MEM_START_ADDR};
    atomic_set_base_ddr(base_idx, base_addr, 2, ENGINE_GDMA);
    atomic_set_base_ddr(base_idx, base_addr, 2, ENGINE_SDMA);
    atomic_set_base_ddr(base_idx, base_addr, 2, ENGINE_HAU);
#ifndef DISABLE_CDMA
    atomic_set_base_ddr(base_idx, base_addr, 2, ENGINE_CDMA);
#endif
    CORE_PRINT("set base_idx=%d, base_addr=0x%llx\n", base_idx[0],
               base_addr[0]);
    CORE_PRINT("set base_idx=%d, base_addr=0x%llx\n", base_idx[1],
               base_addr[1]);
  }
#endif
}

static void enable_cdma() {
  CORE_PRINT_CORE(0, "PLD_K2K_CDMA_TEST_PORT/MAX_CDMA_NUM = %d/%d\n",
                  (int)PLD_K2K_CDMA_TEST_PORT, (int)MAX_CDMA_NUM);
  ASSERT_INFO(0 <= PLD_K2K_CDMA_TEST_PORT &&
                  PLD_K2K_CDMA_TEST_PORT < CONFIG_MAX_CDMA_NUM,
              "PLD_K2K_CDMA_TEST_PORT(%d) should in range [0, %d)",
              (int)PLD_K2K_CDMA_TEST_PORT, (int)CONFIG_MAX_CDMA_NUM);
  for (int c2c_sys_id = 0; c2c_sys_id < 2; c2c_sys_id++) {
    WRITE_REG(C2C_TOP_BASE_ADDR(c2c_sys_id) + 0x24,
              C2C_CFG_START_ADDR(c2c_sys_id) & 0xffffffff, NODECHIP_REG);
    WRITE_REG(C2C_TOP_BASE_ADDR(c2c_sys_id) + 0x28,
              C2C_CFG_START_ADDR(c2c_sys_id) >> 32, NODECHIP_REG);
    WRITE_REG(C2C_TOP_BASE_ADDR(c2c_sys_id) + 0x2c,
              C2C_CFG_END_ADDR(c2c_sys_id) & 0xffffffff, NODECHIP_REG);
    WRITE_REG(C2C_TOP_BASE_ADDR(c2c_sys_id) + 0x30,
              C2C_CFG_END_ADDR(c2c_sys_id) >> 32, NODECHIP_REG);
  }
  // CXP CDMA INIT
  WRITE_REG(CXP_TOP_BASE_ADDR + 0x24, CXP_CFG_START_ADDR & 0xffffffff,
            NODECHIP_REG);
  WRITE_REG(CXP_TOP_BASE_ADDR + 0x28, CXP_CFG_START_ADDR >> 32, NODECHIP_REG);
  WRITE_REG(CXP_TOP_BASE_ADDR + 0x2c, CXP_CFG_END_ADDR & 0xffffffff,
            NODECHIP_REG);
  WRITE_REG(CXP_TOP_BASE_ADDR + 0x30, CXP_CFG_END_ADDR >> 32, NODECHIP_REG);
  // write reg_allreduce_enable
  for (size_t i = 0; i < MAX_CDMA_NUM; i++) {
    u32 cdma_csr0_value =
        READ_REG(CDMA_ENGINE_MAIN_CTRL(i));
    cdma_csr0_value = (cdma_csr0_value | (0x1 << 9));
    WRITE_REG(CDMA_ENGINE_MAIN_CTRL(i), cdma_csr0_value,
              NODECHIP_REG);
    CORE_PRINT_CORE(0, "WRITE_REG addr=%llx, value=%x\n",
               (u64)(CDMA_ENGINE_MAIN_CTRL(i)),
               cdma_csr0_value);
    const int cdma_a4s_src_id[11] = {
        59, 60, 61, 62, 63, 64, 65, 66, 56, 57, 58,
    };
    u32 cdma_a4s_value = READ_REG(CDMA_CSR_REG_A4S(i));
    cdma_a4s_value =
        (cdma_a4s_value & 0xffffff00) + cdma_a4s_src_id[i];
    WRITE_REG(CDMA_CSR_REG_A4S(i), cdma_a4s_value,
              NODECHIP_REG);
    CORE_PRINT_CORE(0,
                    "PLD_K2K_CDMA_TEST_PORT = %d, a4s "
                    "src_id=%d, csr_h120=%x, write to %llx\n",
                    (int)i,
                    cdma_a4s_src_id[i], READ_REG(CDMA_CSR_REG_A4S(i)),
                    (u64)(CDMA_CSR_REG_A4S(i)));
  }
  u32 p2p_mask = ((1 << 6) | (1 << 7));
  u32 p2p_enable_reg =
      READ_REG(CDMA_ENGINE_MAIN_CTRL(8));
  p2p_enable_reg = (p2p_enable_reg | p2p_mask);
  WRITE_REG((CDMA_ENGINE_MAIN_CTRL(8)), p2p_enable_reg,
            NODECHIP_REG);
  CORE_PRINT_CORE(
      0, "p2p port, enable p2p, CSR0 WRITE_REG addr=%llx, value=%x\n",
      (u64)(CDMA_ENGINE_MAIN_CTRL(8)),
      (u32)(READ_REG(CDMA_ENGINE_MAIN_CTRL(8))));
}

int g_core_id = -1;
void __attribute__((weak)) set_scaler_scheduler_mode(int);

#ifdef RECORD_FIFO_DEPTH
static uint32_t fifo_history_length[2];
static uint32_t *fifo_depth_history[2];

void record_fifo_depth(int index, uint32_t v)
{
    uint32_t *history  = fifo_depth_history[index];
    ++history[0];
    history[history[0] * 2] = firmware_timer_get_cycle() / 20;
    history[history[0] * 2 + 1] = v;
}
#else
void record_fifo_depth(int index, uint32_t v) {}
#endif

void store_fifo_depth_records()
{
#ifdef RECORD_FIFO_DEPTH
  if (tpu_workitem_index())
    return;

  // Fifo history
  uint32_t *f = (uint32_t *)GET_GLOBAL_ADDR(0x38000000);
  for (size_t i = 0; i < sizeof(fifo_depth_history) / sizeof(uint32_t *); ++i)
  {

    // 4MiB for each engine
    f += i * 0x100000;
    uint32_t num = fifo_depth_history[i][0];
    uint32_t *poffset = fifo_history_length + i;
    memcpy(f + 2 * (*poffset + 1), fifo_depth_history[i] + 2, fifo_depth_history[i][0] * 2 * sizeof(uint32_t));
    *poffset += num;
    *f = *poffset;

    fifo_depth_history[i][0] = 0;
  }
#endif
}

#ifdef USING_LLM_TICK_TOCK_PROFILE

static uint32_t kernel_profile_index;
static uint64_t kernel_profile_start;
void firmware_kernel_tick()
{
  if (tpu_workitem_index())
    return;

  kernel_profile_start = firmware_timer_get_cycle();
  //kernel_profile_start = read_bd_command_id();
}

void firmware_kernel_tock(int tag)
{
  if (tpu_workitem_index())
    return;
#ifdef RECORD_FIFO_DEPTH
  for (size_t i = 0; i < sizeof(fifo_depth_history) / sizeof(uint32_t *); ++i)
    record_fifo_depth(i, -1);
#endif

  uint64_t *p = (uint64_t *)GET_GLOBAL_ADDR(0x40000000);
  uint64_t v = tag;
  v = (v << 60) | (firmware_timer_get_cycle() - kernel_profile_start);
  //v = (v << 60) | ((read_bd_command_id() - kernel_profile_start) & 0xffffffff);

  ++kernel_profile_index;

  *(p + kernel_profile_index * 2) = v;
  *(p + kernel_profile_index * 2 + 1) = kernel_profile_start;

  *p = kernel_profile_index;
}

#else

void firmware_kernel_tick() {}
void firmware_kernel_tock(int tag) {}

#endif

void tpu_kernel_init(int core_idx)
{
  if (tpu_workitem_index() == 0)
  {
#ifdef RECORD_FIFO_DEPTH
  for (size_t i = 0; i < sizeof(fifo_depth_history) / sizeof(fifo_depth_history[0]); ++i)
  {
    fifo_history_length[i] = 0;
    fifo_depth_history[i] = malloc(0x400000);
    fifo_depth_history[i][0] = 0;
  }
#endif
#ifdef USING_LLM_TICK_TOCK_PROFILE
  kernel_profile_index = 0;
#endif
  }

#ifdef USING_CMODEL
  set_cur_nodechip_idx(core_idx);
  if (core_idx >= MAX_TPU_CORE_NUM)
      return;
#endif

  g_core_id = core_idx;
  load_lookup_tables();
  set_base_addr();

#if USING_APTP_PARALLEL
    set_scaler_scheduler_mode(1);
#endif

#ifndef DISABLE_CDMA
  enable_cdma();
#endif

  {
    // turn on this filed can get better peformance for TPU instructions
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_FB_PXL_EN, 1);
    // lower energy
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_EU_CLK_GATE_EN, 1);
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_CUBE_CLK_GATE_EN, 1);
    FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_LMEM_CLK_GATE_EN, 1);
  }

  {
    // enable the registers for msg-central to debug incorrect use of msg-ids.
    const uint64_t msg_addr = TPU_SYS_MSG_REG_ADDR + (uint64_t)core_idx * CORE_OFFSET;
    // clear status of the registers
    WRITE_REG(msg_addr + 0x41c, 0, NODECHIP_REG);
    WRITE_REG(msg_addr + 0x420, 0, NODECHIP_REG);
    WRITE_REG(msg_addr + 0x424, 0, NODECHIP_REG);
    WRITE_REG(msg_addr + 0x428, 0, NODECHIP_REG);
    // enable the interrupts
    WRITE_REG(msg_addr + 0x464, 0, NODECHIP_REG);
  }

#ifdef ENABLE_POWER_CTRL
  // power step setting
  {
    const reg_id_t reg = BD_ID_CFG_POWER_CTL;
    const uint64_t addr = BD_ENGINE_MAIN_CTRL + (reg.where / 8);
    uint32_t value = (1) | (1 << 1) | (1 << 2) | (1 << 4) |
                     (0xf << 8) | (0xf << 16);
    WRITE_REG(addr, value, NODECHIP_REG);
    value = 0xf;
    WRITE_REG(addr + 4, value, NODECHIP_REG);
  }
#endif
}

int reg_latency_test(void *args, int size)
{
    tpu_initialize();

    sg_api_reg_latency_t api;
    memcpy(&api, args, size);

    uint64_t start, durations[api.num];

    for (int i = 0; i < api.num; ++i)
    {
        start = firmware_timer_get_cycle();

#if 1
        // turn on this filed can get better peformance for TPU instructions
        FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_FB_PXL_EN, 1);
        // lower energy
        FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_EU_CLK_GATE_EN, 1);
        FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_CUBE_CLK_GATE_EN, 1);
        FW_REG_ID_WRITE(BD_ENGINE_MAIN_CTRL, BD_ID_CFG_LMEM_CLK_GATE_EN, 1);
#endif

        durations[i] = firmware_timer_get_cycle() - start;

        local_addr_t dst = 0;
        local_addr_t src = LOCAL_MEM_SIZE / 4;
        dim4 shape = {.n = 1, .c = 64, .h = 1, .w = 128};
        tpu_bdc_relu(dst, src, &shape, NULL, NULL, DT_FP32);
        tpu_poll();
    }

    memcpy(GET_GLOBAL_ADDR(api.output), durations, sizeof(durations));

    return 0;
}

void *firmware_main(void *p) {
    // gating enable
    enable_clk(0x7f | (1 << 20));
#ifdef USING_EDA
    deassert_reset(0xffffffff);
#endif
    enable_l2m();

    firmware_runtime_param_t *param = (firmware_runtime_param_t *)p;
#ifdef USING_CMODEL
    set_cur_nodechip_idx(param->nodechip_idx);
    msg_done[param->nodechip_idx] = 0;
    pthread_mutex_init(&msg_done_mutex[param->nodechip_idx], NULL);
    pthread_cond_init(&msg_done_cond[param->nodechip_idx], NULL);

    pthread_mutex_init(&api_msg_mutex[param->nodechip_idx], NULL);
    pthread_cond_init(&api_msg_cond[param->nodechip_idx], NULL);
    char name[16];
    snprintf(name, sizeof(name), "tpu_scalar%d", param->nodechip_idx);
    // pthread_setname_np(pthread_self(), name);
    prctl(PR_SET_NAME, name, 0, 0, 0);
#endif

    // Enable TPU
    u32 value = READ_REG(BD_ENGINE_MAIN_CTRL_AHB);
    (void)value;
    WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB, (value | 0x1), NODECHIP_REG);
    set_base_addr();

#ifndef DISABLE_CDMA
    enable_cdma();
#endif

    param->terminated = 0;
    sg_fw_event_t event;
    u32 status_back = 0x0;
    u32 last_val = 0x0;
    // Load tables to static memory
#ifdef USING_CMODEL
    if (param->nodechip_idx < MAX_TPU_CORE_NUM) {
        CORE_PRINT("load_lookup_tables\n");
        load_lookup_tables();
    }
#else
    CORE_PRINT("load_lookup_tables\n");
    load_lookup_tables();
#endif
    status_back = READ_SHARE_REG(SHARE_REG_FW_STATUS);
    last_val = (LAST_INI_REG_VAL) & ~(0xf << 28);
    last_val =  (last_val | (status_back & (0xf << 28)));

#ifdef USING_CMODEL
    WRITE_SHARE_REG(SHARE_REG_FW_STATUS, LAST_INI_REG_VAL);
#else
    for (int idx = 0; idx < SHARE_REG_CNT - 1; idx++)
        WRITE_SHARE_REG(idx + SHARE_REG_CNT * CORE_ID, 0);
    WRITE_SHARE_REG(SHARE_REG_FW_STATUS, last_val);
#endif
    FW_INFO("enter nodechip_idx %d\n", param->nodechip_idx);
    while (!param->terminated) {
        event = fw_wait_event(param);
        switch (event) {
        case SG_FW_EVENT_NULL:
            break;
        case SG_FW_EVENT_MSG:
            fw_handle_api_msg(param);
            break;
        default:
            /* not support event */
            FW_ERR("fw event = %d is not supported\n", event);
            break;
        }
// busy waiting sleep 200ms, reduce cpu usage
#if defined(USING_CMODEL) && !defined(USING_MULTI_THREAD_ENGINE)
        usleep(200000);
#endif
    }
    FW_INFO("terminate nodechip_idx %d\n", param->nodechip_idx);

#ifdef USING_FAKE_DDR_MODE
    // give terminate signal to hardware
    value = READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x04);
    WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x04, (value | 0x00200000), NODECHIP_REG);
    firmware_lock(0);
    printf("[%d]:I'm out firmware_main!\n", CORE_ID);
    firmware_unlock(0);
#endif

#ifdef USING_CMODEL
    // Work around gcc 4.8 bug
    // Deconstruct thread context object
    runtime_deinit();
#endif

    return NULL;
}
