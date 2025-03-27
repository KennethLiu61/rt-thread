#include "firmware_pmu.h"
#include "firmware_runtime.h"
#include "firmware_common_macro.h"
#include "tpu_kernel.h"
#include "sg_api_struct.h"

#define MAX_PRINT 1000

static inline void PMU_WRITE_REG(u64 addr, u32 data)
{
    // CORE_PRINT("  writing reg addr = 0x%llx, value = 0x%x\n", addr, data);
    WRITE_REG(addr, data, NODECHIP_REG);
}

static inline u32 PMU_READ_REG(u64 addr)
{
    u32 value = READ_REG(addr);
    // CORE_PRINT("  reading reg addr = 0x%llx, value = 0x%x\n", addr, value);
    return value;
}

#define PMU_PRINT(fmt, ...) FW_DBG(fmt, ##__VA_ARGS__)

#define BINARY_N_ONES(N) ((N >= 0 && N < sizeof(unsigned long long) * 8) ? ((1ULL << (N)) - 1) : 0)
void enable_tpu_perf_monitor()
{
    // config tiu
    // according to SG2260_TPU_TIU_Reg0.12
    {
        int size = PLD_PMU_TIU_MAX_SIZE;
        u64 start_addr = PLD_PMU_TIU_START_ADDR;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);

        PMU_PRINT("enable tpu perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                  size, start_addr, end_addr);
        // CFG7 0X70 : 0-5[6] resvd; 6 cfg_monitor_enable; 7-31[25] cfg_result_start_addr low 25bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x70);
        value &= ~(BINARY_N_ONES(25) << 7);
        value |= (start_addr & BINARY_N_ONES(25)) << 7;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x70, value);
        // CFG7 0X74 : 0-14[15] cfg_result_start_addr high 15bit; 15-31[17] cfg_result_end_addr low 17bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74);
        value &= ~(BINARY_N_ONES(15));
        value |= (start_addr >> 25 & BINARY_N_ONES(15)) << 0;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74, value);
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74);
        value &= ~(BINARY_N_ONES(17) << 15);
        value |= (end_addr & BINARY_N_ONES(17)) << 15;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74, value);
        // CFG7 0X78 : 0-22[23] cfg_result_end_addr high 23bit; 23 cfg_cmpt_en{1}; 24-31[8] cfg_cmpt_val{1} low 8bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78);
        value &= ~(BINARY_N_ONES(23) << 0);
        value |= (end_addr >> 17 & BINARY_N_ONES(23)) << 0;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78, value);
        // write cfg_cmpt_en{1} and cfg_cmpt_val{1} low 8bit; together 9 bit, write b'11
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78);
        value &= ~(BINARY_N_ONES(9) << 23);
        value |= (3 & BINARY_N_ONES(9)) << 23; //b'11
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78, value);
        // CFG7 0X7c : 0-7[8] cfg_cmpt_val{1} high 8bit; 8 cfg_rd_instr_en{1}; 9 cfg_rd_instr_stall_en{1}; 10 cfg_wr_instr_en{1}
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x7c);
        // cfg_cmpt_val{1} high 8bit, all zero; with cfg_rd_instr_en/cfg_rd_instr_stall_en/cfg_wr_instr_en all 11 bit
        value &= ~(BINARY_N_ONES(11) << 0);
        // cfg_rd_instr_en{1}/cfg_rd_instr_stall_en{1}/cfg_wr_instr_en{1} b'111
        value |= 0x7 << 8;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x7c, value);
    }

    // config gdma
    // according to GDMA_SG2260_DES_REG rev 0.68
    {
        u32 size = PLD_PMU_GDMA_MAX_SIZE;
        u64 start_addr = PLD_PMU_GDMA_START_ADDR;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);

        PMU_PRINT("enable gdma perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                  size, start_addr, end_addr);
        // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
        value = (start_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x14, value);
        // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
        value = 0;
        value = (start_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x18, value);
        // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
        value = (end_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x1c, value);
        // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
        value = 0;
        value = (end_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x20, value);
    }

    // config sdma
    // according to GDMA_SG2260_DES_REG rev 0.68
    {
        u32 size = PLD_PMU_SDMA_MAX_SIZE;
        u64 start_addr = PLD_PMU_SDMA_START_ADDR;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);

        PMU_PRINT("enable sdma perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                  size, start_addr, end_addr);
        // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
        value = (start_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x14, value);
        // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
        value = 0;
        value = (start_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x18, value);
        // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
        value = (end_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x1c, value);
        // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
        value = 0;
        value = (end_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x20, value);
    }
    // enable ccsys reg_monitor_en: tpu gdma sdma monitor en
    // according to tpsys_reg
    // h250 : reg_monitor_en : reg_monitor_en{1} : 1'b1
        // 0-3 use cc_sys
    u32 sys_val = PMU_READ_REG(TPU_SYS_PMU_ENABLE);
    sys_val |= 0x1;
    PMU_WRITE_REG(TPU_SYS_PMU_ENABLE, sys_val);
    if (CORE_ID >= 4) {
        // 4-7 use vc_sys
        u32 sys_val = PMU_READ_REG(VC_SYS_PMU_ENABLE);
        sys_val |= (1 << 31);
        PMU_WRITE_REG(VC_SYS_PMU_ENABLE, sys_val);
    }

    {
        // flush cache for cdma
        for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) {
          u32 size = PLD_PMU_CDMA_MAX_SIZE;
          u64 start_addr = PLD_PMU_CDMA_START_ADDR + port * size;
          memset(GET_GLOBAL_ADDR(start_addr), 0, size);
          flush_cache(start_addr, size);
        }
    }
    // config cdma
    // according to CDMA_2260_DES_REG_v5.1
    {
        // multi write will cause pmu not work
        if(tpu_is_last_workitem()) {
            for (int port = 0; port < 8; port++) {
                u32 size = PLD_PMU_CDMA_MAX_SIZE;
                u64 start_addr = PLD_PMU_CDMA_START_ADDR + port * size;
                u64 end_addr = start_addr + size;
                u32 value = 0;
                memset(GET_GLOBAL_ADDR(start_addr), 0, size);
                flush_cache(start_addr, size);
                PMU_PRINT("enable cdma %d perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                            port, size, start_addr, end_addr);
                // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
                value = (start_addr >> 7) & BINARY_N_ONES(32);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x34, value);
                // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
                value = 0;
                value = (start_addr >> 39) & BINARY_N_ONES(1);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x38, value);
                // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
                value = (end_addr >> 7) & BINARY_N_ONES(32);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x3c, value);
                // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
                value = 0;
                value = (end_addr >> 39) & BINARY_N_ONES(1);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x30, value);
                // h240 : reg_des_write_addr_h8 : intra_des_read_addr_high_8bit [8:15]
                // config 0x80 for AXI_RN
                value = PMU_READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x240);
                value = (value & 0xffff00ff) | (0x80 << 8);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x240, value);

                // enable CDMA pmu
                // h0 : reg_perf_monitor_enable [3:3]
                value = PMU_READ_REG(CDMA_ENGINE_MAIN_CTRL(port)) | (1 << 3);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port), value);
            }
        }
    }
}

void disable_tpu_perf_monitor()
{
    // enable ccsys reg_monitor_en: tpu gdma sdma monitor en
    // according to tpsys_reg
    // h250 : reg_monitor_en : reg_monitor_en{0} : 1'b1
    u32 sys_val = PMU_READ_REG(TPU_SYS_PMU_ENABLE);
    sys_val &= ~(0x1);
    PMU_WRITE_REG(TPU_SYS_PMU_ENABLE, sys_val);
    if (CORE_ID >= 4) {
        u32 sys_val = PMU_READ_REG(VC_SYS_PMU_ENABLE);
        sys_val &= 0x7fffffff;
        PMU_WRITE_REG(VC_SYS_PMU_ENABLE, sys_val);
    }
    if(tpu_is_last_workitem()) {
        for (int port = 0; port < 8; port++) {
            u32 sys_val = PMU_READ_REG(CDMA_ENGINE_MAIN_CTRL(port)) & ~(1 << 3);
            PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port), sys_val);
        }
    }
    PMU_PRINT("%s\n", __func__);
}

#pragma pack(1)
typedef struct {
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    // lower 1 bit: thread_id; higher 31 bits: bank_conflict
    unsigned int thread_id_and_bank_conflict;
} tiu_pmu_item_t;

#define REAL_TIU_ITEM_SIZE 32
typedef struct {
    tiu_pmu_item_t valid_data;
    // unsigned char reserved[REAL_TIU_ITEM_SIZE-sizeof(tiu_pmu_item_t)];
} tiu_pmu_item_ext_t;

typedef struct {
    // H0
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    uint32_t thread_id: 1;
    uint32_t ar_latency_cnt: 19;
    uint32_t rip_valid_latency: 12;
    // H1
    unsigned int gif_wr_rd_stall_cntr;
    unsigned int axi_d0_w_cntr;
    unsigned int axi_d0_ar_cntr;
    unsigned int axi_d0_aw_cntr;
    // H2
    unsigned int axi_d0_wr_stall_cntr;
    unsigned int axi_d0_rd_stall_cntr;
    unsigned int gif_mem_w_cntr;
    unsigned int gif_mem_ar_cntr;
    // H3
    unsigned int axi_d0_wr_vaild_cntr;
    unsigned int axi_d0_rd_vaild_cntr;
    unsigned int gif_wr_valid_cntr;
    unsigned int gif_rd_valid_cntr;
} gdma_pmu_item_t;

// #define REAL_GDMA_ITEM_SIZE 256
typedef struct {
    gdma_pmu_item_t valid_data;
    // unsigned char reserved[REAL_GDMA_ITEM_SIZE-sizeof(gdma_pmu_item_t)];
} gdma_pmu_item_ext_t;

typedef gdma_pmu_item_t sdma_pmu_item_t;

typedef struct {
    sdma_pmu_item_t valid_data;
    // unsigned char reserved[REAL_GDMA_ITEM_SIZE-sizeof(sdma_pmu_item_t)];
} sdma_pmu_item_ext_t;

typedef struct {
  // H0
  unsigned int inst_start_time;
  unsigned int inst_end_time;
  unsigned int inst_id; // bit 0-23 inst_id; bit 24 thread id
  unsigned int reserved0;
  // H1
  unsigned int m0_data_aw_cntr;
  unsigned int m0_data_w_cntr;
  unsigned int m0_data_ar_cntr;
  unsigned int reserved1;
  // H2
  unsigned int m0_data_wr_valid_cntr;
  unsigned int m0_data_wr_stall_cntr;
  unsigned int m0_data_rd_valid_cntr;
  unsigned int m0_data_rd_stall_cntr;
  // H3
  unsigned int ati_data_valid_cntr;
  unsigned int ati_data_stall_cntr;
  unsigned int ari_data_valid_cntr;
  unsigned int ari_data_stall_cntr;
  // H4
  unsigned int ati_txfifo_stall_cntr;
  unsigned int replay_number;
  unsigned int m0_data_b_st;
  unsigned int m0_data_b_end;
  // H5
  unsigned int m0_data_ar_st;
  unsigned int m0_data_ar_end;
  unsigned int m0_data_aw_st;
  unsigned int m0_data_aw_end;
  // H6
  unsigned int m0_data_rd_st;
  unsigned int m0_data_rd_end;
  unsigned int m0_data_wr_st;
  unsigned int m0_data_wr_end;
  // H7
  unsigned int ati_data_st;
  unsigned int ati_data_end;
  unsigned int ari_data_st;
  unsigned int ari_data_end;
} cdma_pmu_item_t;

typedef struct {
  cdma_pmu_item_t valid_data;
  // unsigned char reserved[REAL_CDMA_ITEM_SIZE-sizeof(cdma_pmu_item_t)];
} cdma_pmu_item_ext_t;
#pragma pack()


void show_pmu_item_gdma(const gdma_pmu_item_t *p, int i, float period) {
  float total_time = (p->inst_end_time - p->inst_start_time) * period;
  (void)(total_time);
  PMU_PRINT(
      "---> gdma record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
      "start=%.3fus, end=%.3fus, interval=%.3fus, ar_latency_cnt=%d, "
      "rip_valid_latency=%d, gif_wr_rd_stall_cntr= %d\n"
      "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
      "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
      "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
      "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
      i, p->inst_id, p->thread_id, p->inst_end_time - p->inst_start_time,
      p->inst_start_time * period, p->inst_end_time * period, total_time,
      p->ar_latency_cnt, p->rip_valid_latency,
      p->gif_wr_rd_stall_cntr, p->axi_d0_w_cntr, p->axi_d0_ar_cntr,
      p->axi_d0_aw_cntr, p->axi_d0_wr_stall_cntr, p->axi_d0_rd_stall_cntr,
      p->gif_mem_w_cntr, p->gif_mem_ar_cntr, p->axi_d0_wr_vaild_cntr,
      p->axi_d0_rd_vaild_cntr, p->gif_wr_valid_cntr, p->gif_rd_valid_cntr);
}

void show_pmu_item_sdma(const sdma_pmu_item_t *p, int i, float period) {
  float total_time = (p->inst_end_time - p->inst_start_time) * period;
  (void)(total_time);
  PMU_PRINT(
      "---> sdma record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
      "start=%.3fus, end=%.3fus, interval=%.3fus, ar_latency_cnt=%d, "
      "rip_valid_latency=%d, gif_wr_rd_stall_cntr= %d\n"
      "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
      "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
      "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
      "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
      i, p->inst_id, p->thread_id, p->inst_end_time - p->inst_start_time,
      p->inst_start_time * period, p->inst_end_time * period, total_time,
      p->ar_latency_cnt, p->rip_valid_latency,
      p->gif_wr_rd_stall_cntr, p->axi_d0_w_cntr, p->axi_d0_ar_cntr,
      p->axi_d0_aw_cntr, p->axi_d0_wr_stall_cntr, p->axi_d0_rd_stall_cntr,
      p->gif_mem_w_cntr, p->gif_mem_ar_cntr, p->axi_d0_wr_vaild_cntr,
      p->axi_d0_rd_vaild_cntr, p->gif_wr_valid_cntr, p->gif_rd_valid_cntr);
}

void show_pmu_item_tiu(const tiu_pmu_item_t *p, int i, float period) {
  PMU_PRINT("---> tiu record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
               "start=%.3fus, end=%.3fus, interval=%.3fus, bank_conflict=%d\n",
               i, (int)(p->inst_id),
               (int)(p->thread_id_and_bank_conflict & 0x1),
               (int)(p->inst_end_time - p->inst_start_time),
               p->inst_start_time * period, p->inst_end_time * period,
               (p->inst_end_time - p->inst_start_time) * period,
               (int)(p->thread_id_and_bank_conflict >> 1));
}

void show_pmu_item_cdma(const cdma_pmu_item_t *p, int i, float period) {
  float total_time = (p->inst_end_time - p->inst_start_time) * period;
  (void)(total_time);
  PMU_PRINT(
      "---> sdma record #%d, inst_id=%d, cycle=%d, "
      "start=%.3fus, end=%.3fus, interval=%.3fus, m0_data_aw_cntr= %d\n"
      "m0_data_w_cntr=%d, m0_data_ar_cntr=%d, m0_data_wr_valid_cntr=%d, "
      "m0_data_wr_stall_cntr=%d, m0_data_rd_valid_cntr=%d, "
      "m0_data_rd_stall_cntr=%d\n"
      "ati_data_valid_cntr=%d, ati_data_stall_cntr=%d, ari_data_valid_cntr=%d, "
      "ari_data_stall_cntr=%d, ati_txfifo_stall_cntr=%d, replay_number=%d\n"
      "m0_data_b_st=%d, m0_data_b_end=%d, m0_data_ar_st=%d, m0_data_ar_end=%d, "
      "m0_data_aw_st=%d, m0_data_aw_end=%d, m0_data_rd_st=%d, "
      "m0_data_rd_end=%d\n"
      "m0_data_wr_st=%d, m0_data_wr_end=%d, ati_data_st=%d, ati_data_end=%d, "
      "ari_data_st=%d, ari_data_end=%d\n",
      i, p->inst_id, p->inst_end_time - p->inst_start_time,
      p->inst_start_time * period, p->inst_end_time * period, total_time,
      p->m0_data_aw_cntr, p->m0_data_w_cntr, p->m0_data_ar_cntr,
      p->m0_data_wr_valid_cntr, p->m0_data_wr_stall_cntr,
      p->m0_data_rd_valid_cntr, p->m0_data_rd_stall_cntr,
      p->ati_data_valid_cntr, p->ati_data_stall_cntr, p->ari_data_valid_cntr,
      p->ari_data_stall_cntr, p->ati_txfifo_stall_cntr, p->replay_number,
      p->m0_data_b_st, p->m0_data_b_end, p->m0_data_ar_st, p->m0_data_ar_end,
      p->m0_data_aw_st, p->m0_data_aw_end, p->m0_data_rd_st, p->m0_data_rd_end,
      p->m0_data_wr_st, p->m0_data_wr_end, p->ati_data_st, p->ati_data_end,
      p->ari_data_st, p->ari_data_end);
}

int all_zero(const unsigned char* data, int len){
    for(int i=0; i<len; i++){
        if(data[i]!=0) return 0;
    }
    return 1;
}

void show_tpu_perf_data(int gdma_max_num, int sdma_max_num, int tiu_max_num, int cdma_max_num) {
    int gdma_num=0;
    int sdma_num=0;
    int tiu_num=0;
    int cdma_num=0;
    {
        float freq_MHz = 1000;
        float time_offset = 0;
        float period = 1/freq_MHz;
        (void)(time_offset);
        PMU_PRINT("Note: gdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
        int max_len = PLD_PMU_GDMA_MAX_SIZE/ sizeof(gdma_pmu_item_ext_t);
        u64 item_addr = PLD_PMU_GDMA_START_ADDR;
        gdma_pmu_item_ext_t* item_data = (gdma_pmu_item_ext_t*)GET_GLOBAL_ADDR(item_addr);
        const int item_size = sizeof(gdma_pmu_item_ext_t);
        const int item_cache_size = ALIGN(item_size, CACHE_LINE_SIZE);
        while (gdma_num< max_len) {
            invalidate_cache(item_addr, item_cache_size);
            if (all_zero((const unsigned char*)&item_data[gdma_num], sizeof(gdma_pmu_item_ext_t))) break;
            if (gdma_max_num > 0 && gdma_num < gdma_max_num) {
                show_pmu_item_gdma(&item_data[gdma_num].valid_data, gdma_num, period);
            }
            gdma_num++;
            item_addr += item_size;
        }
    }
    {
        float freq_MHz = 1000;
        float time_offset = 0;
        float period = 1/freq_MHz;
        (void)(time_offset);
        PMU_PRINT("Note: sdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
        int max_len = PLD_PMU_SDMA_MAX_SIZE/ sizeof(sdma_pmu_item_ext_t);
        u64 item_addr = PLD_PMU_SDMA_START_ADDR;
        sdma_pmu_item_ext_t* item_data = (sdma_pmu_item_ext_t*)GET_GLOBAL_ADDR(item_addr);
        const int item_size = sizeof(sdma_pmu_item_ext_t);
        const int item_cache_size = ALIGN(item_size, CACHE_LINE_SIZE);
        while (sdma_num< max_len) {
            invalidate_cache(item_addr, item_cache_size);
            if (all_zero((const unsigned char*)&item_data[sdma_num], sizeof(sdma_pmu_item_ext_t))) break;
            if (sdma_max_num > 0 && sdma_num < sdma_max_num) {
                show_pmu_item_sdma(&item_data[sdma_num].valid_data, sdma_num, period);
            }
            sdma_num++;
            item_addr += item_size;
        }
    }
    {
        float freq_MHz = 1000;
        float time_offset = 0;
        float period = 1 / freq_MHz;
        (void)(time_offset);
        PMU_PRINT("Note: tiu record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
        int max_len = PLD_PMU_TIU_MAX_SIZE/ sizeof(tiu_pmu_item_ext_t);
        u64 item_addr = PLD_PMU_TIU_START_ADDR;
        tiu_pmu_item_ext_t *item_data = (tiu_pmu_item_ext_t *)GET_GLOBAL_ADDR(item_addr);
        const int item_size = sizeof(tiu_pmu_item_ext_t);
        const int item_cache_size = ALIGN(item_size, CACHE_LINE_SIZE);
        int i = 0;
        while (tiu_num < max_len) {
            // cache is 64-byte aligned
            if ((i++ % 4) == 0) {
                invalidate_cache(item_addr, item_cache_size);
            }
            if (all_zero((const unsigned char*)&item_data[tiu_num], sizeof(tiu_pmu_item_ext_t))) break;
            if (tiu_max_num > 0 && tiu_num < tiu_max_num) {
                show_pmu_item_tiu(&item_data[tiu_num].valid_data, tiu_num, period);
            }
            tiu_num++;
            item_addr += item_size;
        }
    }
    {
      float freq_MHz = 1000;
      float time_offset = 0;
      float period = 1 / freq_MHz;
      (void)(time_offset);
      PMU_PRINT(
          "Note: cdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n",
          time_offset, freq_MHz, period);
      int max_len = PLD_PMU_CDMA_MAX_SIZE / sizeof(cdma_pmu_item_ext_t);
      u64 item_addr = PLD_PMU_CDMA_START_ADDR;
      cdma_pmu_item_ext_t *item_data =
          (cdma_pmu_item_ext_t *)GET_GLOBAL_ADDR(item_addr);
      const int item_size = sizeof(cdma_pmu_item_ext_t);
      const int item_cache_size = ALIGN(item_size, CACHE_LINE_SIZE);
      int i = 0;
      while (cdma_num < max_len) {
        // cache is 64-byte aligned
        if ((i++ % 4) == 0) {
          invalidate_cache(item_addr, item_cache_size);
        }
        if (all_zero((const unsigned char *)&item_data[cdma_num],
                     sizeof(cdma_pmu_item_ext_t)))
          break;
        if (cdma_max_num > 0 && cdma_num < cdma_max_num) {
          show_pmu_item_cdma(&item_data[cdma_num].valid_data, cdma_num, period);
        }
        cdma_num++;
        item_addr += item_size;
      }
    }
    PMU_PRINT("gdma record_num=%d, sdma record_num=%d, tiu_record_num=%d, "
              "cdma_recore_num=%d\n",
              (int)gdma_num, (int)sdma_num, (int)tiu_num, (int) cdma_num);
}

static bool cdma_ports_act[CONFIG_MAX_CDMA_NUM];

bool cdma_port_is_valid(int port) {
    return cdma_ports_act[port];
}

void enable_pmu(unsigned enable_bits, void(*before_func)(int), int param) {
    // enable mcu timer
    if(before_func) before_func(param);
    // enable ccsys reg_monitor_en: tpu gdma sdma monitor en
    // according to tpsys_reg
    // h250 : reg_monitor_en : reg_monitor_en{1} : 1'b1
        // 0-3 use cc_sys
    u32 sys_val = PMU_READ_REG(TPU_SYS_PMU_ENABLE);
    sys_val &= ~(0x1);
    sys_val |= ((enable_bits >> 1) & 0x1);
    PMU_WRITE_REG(TPU_SYS_PMU_ENABLE, sys_val);
    sys_val = PMU_READ_REG(TPU_SYS_PMU_ENABLE);
    if (CORE_ID >= 4) {
        // 4-7 use vc_sys
        sys_val = PMU_READ_REG(VC_SYS_PMU_ENABLE);
        sys_val &= ~(0x1 << 31);
        sys_val |= ((enable_bits >> 1) & 0x1) << 31;
        PMU_WRITE_REG(VC_SYS_PMU_ENABLE, sys_val);
        sys_val = PMU_READ_REG(VC_SYS_PMU_ENABLE);
    }
#ifndef DISABLE_CDMA
    if(tpu_is_last_workitem()) {
        // enable CDMA pmu
        // h0 : reg_perf_monitor_enable [3:3]
        // u32 core_id = tpu_workitem_index();
        for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) {
            if (!cdma_ports_act[port])
                continue;
            sys_val = PMU_READ_REG(CDMA_ENGINE_MAIN_CTRL(port));
            sys_val &= ~(0x1 << 3);
            sys_val |= ((enable_bits) & 0x1) << 3;
            PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port), sys_val);
        }
    }
#endif
}

void set_pmu_param(void* api) {
    // multi enable cause pmu not work, so make sure pmu is disabled for eatch init 
    disable_tpu_perf_monitor();
    sg_api_engine_profile_param_t* info = (sg_api_engine_profile_param_t*)api;
#ifndef DISABLE_CDMA
    // config cdma
    // according to CDMA_2260_DES_REG_v5.1
    {
        if(tpu_is_last_workitem()) {
            for (int port = 0; port < CONFIG_MAX_CDMA_NUM; port++) 
            {
                u32 size = info[port].size;
                if (size == 0) {
                    cdma_ports_act[port] = false;
                    continue;
                }
                cdma_ports_act[port] = true;
                u64 start_addr = info[port].addr;
                memset(GET_GLOBAL_ADDR(start_addr), 0, size);
                flush_cache(start_addr, size);
                u32 value = 0;
                u64 end_addr = start_addr + size;
                // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
                value = (start_addr >> 7) & BINARY_N_ONES(32);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x34, value);
                // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
                value = 0;
                value = (start_addr >> 39) & BINARY_N_ONES(1);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x38, value);
                // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
                value = (end_addr >> 7) & BINARY_N_ONES(32);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x3c, value);
                // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
                value = 0;
                value = (end_addr >> 39) & BINARY_N_ONES(1);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x30, value);
                // h240 : reg_des_write_addr_h8 : intra_des_read_addr_high_8bit [8:15]
                // config 0x80 for AXI_RN
                value = PMU_READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x240);
                value = (value & 0xffff00ff) | (0x80 << 8);
                PMU_WRITE_REG(CDMA_ENGINE_MAIN_CTRL(port) + 0x240, value);
            }
        }
    }
#endif
    info += CONFIG_MAX_CDMA_NUM + CORE_ID * 3; // 3 means tiu gdma vsdma
    // config tiu
    // according to SG2260_TPU_TIU_Reg0.12
    {
        u32 size = info[0].size;
        u64 start_addr = info[0].addr;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);
        // CFG7 0X70 : 0-5[6] resvd; 6 cfg_monitor_enable; 7-31[25] cfg_result_start_addr low 25bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x70);
        value &= ~(BINARY_N_ONES(25) << 7);
        value |= (start_addr & BINARY_N_ONES(25)) << 7;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x70, value);
        // CFG7 0X74 : 0-14[15] cfg_result_start_addr high 15bit; 15-31[17] cfg_result_end_addr low 17bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74);
        value &= ~(BINARY_N_ONES(15));
        value |= (start_addr >> 25 & BINARY_N_ONES(15)) << 0;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74, value);
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74);
        value &= ~(BINARY_N_ONES(17) << 15);
        value |= (end_addr & BINARY_N_ONES(17)) << 15;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x74, value);
        // CFG7 0X78 : 0-22[23] cfg_result_end_addr high 23bit; 23 cfg_cmpt_en{1}; 24-31[8] cfg_cmpt_val{1} low 8bit
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78);
        value &= ~(BINARY_N_ONES(23) << 0);
        value |= (end_addr >> 17 & BINARY_N_ONES(23)) << 0;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78, value);
        // write cfg_cmpt_en{1} and cfg_cmpt_val{1} low 8bit; together 9 bit, write b'11
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78);
        value &= ~(BINARY_N_ONES(9) << 23);
        value |= (3 & BINARY_N_ONES(9)) << 23; //b'11
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x78, value);
        // CFG7 0X7c : 0-7[8] cfg_cmpt_val{1} high 8bit; 8 cfg_rd_instr_en{1}; 9 cfg_rd_instr_stall_en{1}; 10 cfg_wr_instr_en{1}
        value = PMU_READ_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x7c);
        // cfg_cmpt_val{1} high 8bit, all zero; with cfg_rd_instr_en/cfg_rd_instr_stall_en/cfg_wr_instr_en all 11 bit
        value &= ~(BINARY_N_ONES(11) << 0);
        // cfg_rd_instr_en{1}/cfg_rd_instr_stall_en{1}/cfg_wr_instr_en{1} b'111
        value |= 0x7 << 8;
        PMU_WRITE_REG(BD_ENGINE_MAIN_CTRL_AHB + 0x7c, value);
    }

    // config gdma
    // according to GDMA_SG2260_DES_REG rev 0.68
    {
        u32 size = info[1].size;
        u64 start_addr = info[1].addr;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);

        PMU_PRINT("enable gdma perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                  size, start_addr, end_addr);
        // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
        value = (start_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x14, value);
        // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
        value = 0;
        value = (start_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x18, value);
        // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
        value = (end_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x1c, value);
        // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
        value = 0;
        value = (end_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(GDMA_ENGINE_MAIN_CTRL_AHB + 0x20, value);
    }

    // config sdma
    // according to GDMA_SG2260_DES_REG rev 0.68
    {
        u32 size = info[2].size;
        u64 start_addr = info[2].addr;
        u64 end_addr = start_addr + size;
        u32 value = 0;
        memset(GET_GLOBAL_ADDR(start_addr), 0, size);
        flush_cache(start_addr, size);

        PMU_PRINT("enable sdma perf monitor size = 0x%x, start addr = 0x%llx, end addr = 0x%llx\n",
                  size, start_addr, end_addr);
        // h14 : perf_monitor_res_start_addr_l32 : ddr start addr [38:7]
        value = (start_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x14, value);
        // h18 : perf_monitor_res_start_addr_h1 : ddr start addr [39]
        value = 0;
        value = (start_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x18, value);
        // h1c : perf_monitor_res_end_addr_l32 : ddr end addr [38:7]
        value = (end_addr >> 7) & BINARY_N_ONES(32);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x1c, value);
        // h20 : perf_monitor_res_end_addr_h1 : ddr end addr [38:7]
        value = 0;
        value = (end_addr >> 39) & BINARY_N_ONES(1);
        PMU_WRITE_REG(SDMA_ENGINE_MAIN_CTRL + 0x20, value);
    }
}