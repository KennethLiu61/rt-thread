#include <rtthread.h>
#include "config.h"
#include "tpu_kernel_internel.h"
#include "tpu_kernel.h"

/*
1. 0~23, 48~71, ... private use in one TPU core
2. 24~31, 72~79, ... sync between all TPU cores in a task, all the cores will be used
3. 32~39, 80~87, ... global use for different engines, and some cores may be unused
4. 384~511: msg ids for cores barrier between two tasks, you can not change these ids because tpuv7-runtime has used them.
*/
#define ENTRY_PER_MSG_CENTRAL 48
#define LOCAL_MSG_ID_NUM (24)
#define GLOBAL_MSG_ID_NUM (8)
#define CORE_SYNC_MSG_ID_NUM (8)
#define C2C_MSG_ID_NUM 6
#define SCHEDULE_MSG_ID_NUM 2
#define MSG_CENTRAL_NUM 8
#define MAX_CHIP 16
#define MAX_C2C_PORT 6

typedef struct {
   int workitem_id;
   int workitem_num;
   int group_id;
   int group_num;
   int base_msg_id;
   int core_msg_id;
   int start_physical_core_id;
   int tpuv7_env;
   struct {
    int chip_id;
    int chip_num;
    int rank;
    int chip_map[MAX_CHIP];
    int used_c2c_port_num;
    int c2c_ports[MAX_C2C_PORT];
    int sccl_inited;
    int use_ring;
   } sccl_args;
   struct {
    int local_id;
    int global_id;
    int core_sync_id;
    int c2c_id;
    int schedule_id;
   } msg;
} tpu_kernel_core_context_t;

THREAD tpu_kernel_core_context_t g_core_context = {0};

void tpu_set_base_msg_id(int base_msg_id) {
  atomic_set_base_msg_id(base_msg_id, ENGINE_BD);
  atomic_set_base_msg_id(base_msg_id, ENGINE_GDMA);
  atomic_set_base_msg_id(base_msg_id, ENGINE_SDMA);
  atomic_set_base_msg_id(base_msg_id, ENGINE_VSDMA);
  atomic_set_base_msg_id(base_msg_id, ENGINE_HAU);
  atomic_set_base_msg_id(base_msg_id, ENGINE_CDMA);
}

static int calc_base_msg_id(int core_msg_id) {
  for (int i = 0; i < MAX_TPU_CORE_NUM; i++) {
    if (core_msg_id & (1<<i)) {
      return i * ENTRY_PER_MSG_CENTRAL;
    }
  }
  return 0;
}

void tpu_core_context_setup(int core_idx, int core_num, int core_msg_id) {
  g_core_context.workitem_id = core_idx;
  g_core_context.workitem_num = core_num;
  g_core_context.group_id = 0;
  g_core_context.group_num = 1;
  g_core_context.core_msg_id = core_msg_id;

  g_core_context.msg.local_id = 0;
  g_core_context.msg.global_id = 0;
  g_core_context.msg.c2c_id = 0;
  g_core_context.msg.schedule_id = 0;

  int base_msg_id = calc_base_msg_id(g_core_context.core_msg_id);
  g_core_context.base_msg_id = base_msg_id;
}

/*
  TODO optimize get_core_msg perf
*/
static inline int tpu_get_core_msg_id(tpu_groupset_info_t *tpu_groupset) {
  TPUKERNEL_ASSERT(tpu_groupset->physical_core_id >= tpu_groupset->workitem_id);
  int start_physical_core_id = tpu_groupset->physical_core_id - (tpu_groupset->workitem_id);
  g_core_context.start_physical_core_id = start_physical_core_id;
  TPUKERNEL_ASSERT(start_physical_core_id >= 0);
  TPUKERNEL_ASSERT(start_physical_core_id + (tpu_groupset->workitem_num * tpu_groupset->group_num) <= MAX_TPU_CORE_NUM);
  int core_msg_id = 0;
  for (int core_id = start_physical_core_id; core_id < (int)(tpu_groupset->workitem_num * tpu_groupset->group_num); ++core_id) {
    core_msg_id |= ( 1 << core_id );
  }
  return core_msg_id;
}

int tpu_start_physical_core_id() {
  return g_core_context.start_physical_core_id;
}

int tpu_get_base_msg_id() {
  return g_core_context.base_msg_id;
}

void set_tpu_groupset_info(tpu_groupset_info_t *tpu_groupset) {
  g_core_context.workitem_id = (int)tpu_groupset->workitem_id;
  g_core_context.workitem_num = (int)tpu_groupset->workitem_num;
  g_core_context.group_id = (int)tpu_groupset->group_id;
  g_core_context.group_num = (int)tpu_groupset->group_num;
  g_core_context.core_msg_id = tpu_get_core_msg_id(tpu_groupset);
  g_core_context.tpuv7_env = 1;

  g_core_context.msg.local_id = 0;
  g_core_context.msg.global_id = 0;
  g_core_context.msg.core_sync_id = 0;
  g_core_context.msg.c2c_id = 0;
  int base_msg_id = calc_base_msg_id(g_core_context.core_msg_id);
  g_core_context.base_msg_id = base_msg_id;
}
RTM_EXPORT(set_tpu_groupset_info);

int tpu_core_num() {
  return tpu_workitem_num();
}

int tpu_core_index() {
#ifdef USING_EDA
  return CORE_ID;
#else
  return tpu_workitem_index();
#endif
}

int tpu_workitem_num() {
  return g_core_context.workitem_num;
}

int tpu_workitem_index() {
  return g_core_context.workitem_id;
}
RTM_EXPORT(tpu_workitem_index);

int tpu_group_num() {
  return g_core_context.group_num;
}

int tpu_group_index() {
  return g_core_context.group_id;
}

int tpu_is_last_workitem() {
  return tpu_workitem_index() == tpu_group_num() * tpu_workitem_num() - 1;
}

int tpu_tpuv7_env() {
  return g_core_context.tpuv7_env;
}

int tpu_chip_id() {
#ifdef SG_TV_GEN
  char *rank = getenv("OMPI_COMM_WORLD_RANK");
  if (rank == NULL) {
    rank = getenv("LOCAL_RANK");
    if (rank == NULL) {
      return 0;
    }
  }
  return atoi(rank);
#else
  if (!g_core_context.sccl_args.sccl_inited) {
      TPUKERNEL_ASSERT_INFO(0, "chip id has not been setted");
  }
  return g_core_context.sccl_args.chip_id;
#endif
}

int tpu_chip_num() {
#ifdef SG_TV_GEN
  char *size = getenv("OMPI_COMM_WORLD_SIZE");
  if (size == NULL) {
    size = getenv("LOCAL_WORLD_SIZE");
    if (size == NULL) {
      return 1;
    }
  }
  return atoi(size);
#else
  return g_core_context.sccl_args.chip_num;
#endif
}

int tpu_rank() {
#ifdef SG_TV_GEN
  char *rank = getenv("OMPI_COMM_WORLD_RANK");
  if (rank == NULL) {
    rank = getenv("LOCAL_RANK");
    if (rank == NULL) {
      return 0;
    }
  }
  return atoi(rank);
#else
  return g_core_context.sccl_args.rank;
#endif
}

int tpu_cdma_get_used_port_num() {
  return g_core_context.sccl_args.used_c2c_port_num;
}

int* tpu_cdma_get_used_ports() {
  int world_size = tpu_chip_num();
  if(!world_size || world_size == 1) {
    return NULL;
  }
  return g_core_context.sccl_args.c2c_ports;
}

int* tpu_chip_map() {
  if(!g_core_context.sccl_args.sccl_inited) {
    TPUKERNEL_ASSERT_INFO(0, "chip map has not been setted");
  }
  return g_core_context.sccl_args.chip_map;
}

int tpu_use_ring() {
  if(!g_core_context.sccl_args.sccl_inited) {
    TPUKERNEL_ASSERT_INFO(0, "use_ring has not been setted");
  }
  return g_core_context.sccl_args.use_ring;
}

void tpu_sccl_init(int chip_num, int rank, int* chip_map, int use_ring) {
  g_core_context.sccl_args.chip_num = chip_num;
  g_core_context.sccl_args.rank = rank;
  g_core_context.sccl_args.use_ring = use_ring;
  for (int i = 0; i < chip_num; i++) {
#ifdef USING_CMODEL
    g_core_context.sccl_args.chip_map[i] = i;
    chip_map[i] = i;
#else
    g_core_context.sccl_args.chip_map[i] = chip_map[i];
#endif
  }
  g_core_context.sccl_args.chip_id = chip_map[rank];
  for(int i = 0; i < MAX_C2C_PORT; i++) {
    g_core_context.sccl_args.c2c_ports[i] = -1;
  }
  if(chip_num == 1) {
    g_core_context.sccl_args.used_c2c_port_num = 0;
    return;
  }
  int idx = 0;
  int recv_port = 0;
  int send_port = 0;
  int cur_chip_id = chip_map[rank];
  if(chip_num == 2) {
    int peer_rank = (rank + 1) % chip_num;
    int peer_id = chip_map[peer_rank];
    recv_port = tpu_cdma_get_port(cur_chip_id, peer_id, C2C_RECV);
    send_port = tpu_cdma_get_port(cur_chip_id, peer_id, C2C_SEND);
    if (recv_port == send_port) {
      g_core_context.sccl_args.c2c_ports[idx++] = recv_port;
    } else {
      g_core_context.sccl_args.c2c_ports[idx++] = recv_port;
      g_core_context.sccl_args.c2c_ports[idx++] = send_port;
    }
  } else {
    int domain_size, port_num;
    int plane_id = 0;
    int peer_rank = 0;
    if (use_ring == 1) {
      domain_size = chip_num;
      port_num = 2;
    } else {
        domain_size = chip_num / 2;
        port_num = 3;
        peer_rank = (rank + domain_size) % chip_num;
        plane_id = rank / domain_size;
    }
    int local_rank = rank % domain_size;
    int left_rank = plane_id * domain_size + (local_rank + domain_size - 1) % domain_size;
    int right_rank = plane_id * domain_size + (local_rank + 1) % domain_size;
    int peer_chipid = chip_map[peer_rank];
    int left_chipid = chip_map[left_rank];
    int right_chipid = chip_map[right_rank];

    int chip_ids[3] = {right_chipid, left_chipid, peer_chipid};
    for (int i = 0; i < port_num; i++) {
      recv_port = tpu_cdma_get_port(cur_chip_id, chip_ids[i], C2C_RECV);
      send_port = tpu_cdma_get_port(cur_chip_id, chip_ids[i], C2C_SEND);
      if (recv_port == send_port) {
        g_core_context.sccl_args.c2c_ports[idx++] = recv_port;
      } else {
        g_core_context.sccl_args.c2c_ports[idx++] = recv_port;
        g_core_context.sccl_args.c2c_ports[idx++] = send_port;
      }
    }
  }
  g_core_context.sccl_args.used_c2c_port_num = idx;
  g_core_context.sccl_args.sccl_inited = 1;
}

int tpu_get_local_msg_id() {
  int range = LOCAL_MSG_ID_NUM;
  int msg_id = g_core_context.msg.local_id++;
  g_core_context.msg.local_id %= range;
  msg_id += tpu_core_index() * ENTRY_PER_MSG_CENTRAL;
  return msg_id;
}

int tpu_get_global_msg_id() {
  int range = tpu_core_num() * GLOBAL_MSG_ID_NUM;
  int msg_id = g_core_context.msg.global_id++;
  g_core_context.msg.global_id %= range;
  msg_id = (msg_id / GLOBAL_MSG_ID_NUM) * ENTRY_PER_MSG_CENTRAL +
           (msg_id % GLOBAL_MSG_ID_NUM) +
           (LOCAL_MSG_ID_NUM + CORE_SYNC_MSG_ID_NUM);
  return msg_id;
}
static int get_core_sync_msg_id() {
#ifdef SG_STAS_GEN
  int range = 2; // restrict 2 msgid for compiler
  int msg_id = g_core_context.msg.core_sync_id + LOCAL_MSG_ID_NUM;
#else
  // msg_id 0 and 1 are used by MLIR, and seperate the msg_id for des and pio mode
  int range = GLOBAL_MSG_ID_NUM - 2;
  int msg_id = g_core_context.msg.core_sync_id + LOCAL_MSG_ID_NUM + 2;
#endif
  g_core_context.msg.core_sync_id += 1;
  g_core_context.msg.core_sync_id %= range;
  return msg_id;
}

int tpu_get_ccl_msg_id() {
  int range = MSG_CENTRAL_NUM * C2C_MSG_ID_NUM;
  int msg_id = g_core_context.msg.c2c_id++;
  g_core_context.msg.c2c_id %= range;
  msg_id = (msg_id / C2C_MSG_ID_NUM) * ENTRY_PER_MSG_CENTRAL +
           (msg_id % C2C_MSG_ID_NUM) +
           (LOCAL_MSG_ID_NUM + GLOBAL_MSG_ID_NUM + CORE_SYNC_MSG_ID_NUM);
  return msg_id;
}

int tpu_get_schedule_msg_id()
{
  int range = MSG_CENTRAL_NUM * SCHEDULE_MSG_ID_NUM;
  int msg_id = g_core_context.msg.schedule_id++;
  g_core_context.msg.schedule_id %= range;
  msg_id = (msg_id / SCHEDULE_MSG_ID_NUM) * ENTRY_PER_MSG_CENTRAL +
           (msg_id % SCHEDULE_MSG_ID_NUM) +
           (LOCAL_MSG_ID_NUM + GLOBAL_MSG_ID_NUM + CORE_SYNC_MSG_ID_NUM + C2C_MSG_ID_NUM);
  return msg_id;
}

int tpu_next_msg_id() {
  return tpu_get_local_msg_id();
}

void tpu_bdc_send_msg(int msg_id, int wait_cnt) {
    atomic_send_msg_gen_cmd(ENGINE_BD, msg_id, wait_cnt, MASTER_THREAD, BDC_NODE);
}
void tpu_bdc_wait_msg(int msg_id, int send_cnt) {
    atomic_wait_msg_gen_cmd(ENGINE_BD, msg_id, send_cnt, MASTER_THREAD, BDC_NODE);
}
void tpu_hau_send_msg(int msg_id, int wait_cnt) {
    atomic_send_msg_gen_cmd(ENGINE_HAU, msg_id, wait_cnt, MASTER_THREAD, &id_node);
}
void tpu_hau_wait_msg(int msg_id, int send_cnt) {
    atomic_wait_msg_gen_cmd(ENGINE_HAU, msg_id, send_cnt, MASTER_THREAD, &id_node);
}
void tpu_gdma_send_msg(int msg_id, int wait_cnt) {
    atomic_send_msg_gen_cmd(ENGINE_GDMA, msg_id, wait_cnt, MASTER_THREAD, GDMA_NODE);
}
void tpu_gdma_wait_msg(int msg_id, int send_cnt) {
    atomic_wait_msg_gen_cmd(ENGINE_GDMA, msg_id, send_cnt, MASTER_THREAD, GDMA_NODE);
}

void tpu_sdma_send_msg(int msg_id, int wait_cnt) {
    atomic_send_msg_gen_cmd(ENGINE_SDMA, msg_id, wait_cnt, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_vsdma_send_msg(int msg_id, int wait_cnt, int port_id) {
#if defined(C2C_USE_DESCRIPTOR)
    port_id = DEFAULT_SDMA_PORT;
#endif
    atomic_send_msg_gen_cmd(ENGINE_VSDMA, msg_id, wait_cnt, port_id, &id_node);
}

void tpu_sdma_wait_msg(int msg_id, int send_cnt) {
    atomic_wait_msg_gen_cmd(ENGINE_SDMA, msg_id, send_cnt, DEFAULT_SDMA_PORT, &id_node);
}

void tpu_vsdma_wait_msg(int msg_id, int send_cnt, int port_id) {
#if defined(C2C_USE_DESCRIPTOR)
    port_id = DEFAULT_SDMA_PORT;
#endif
    atomic_wait_msg_gen_cmd(ENGINE_VSDMA, msg_id, send_cnt, port_id, &id_node);
}

void tpu_cdma_send_msg(int port, int msg_id, int wait_cnt) {
  atomic_cdma_nop_gen_cmd(port, &id_node);
  atomic_cdma_send_msg_gen_cmd(port, msg_id, wait_cnt, &id_node);
}

void tpu_cdma_wait_msg(int port, int msg_id, int send_cnt) {
  atomic_cdma_wait_msg_gen_cmd(port, msg_id, send_cnt, &id_node);
  atomic_cdma_nop_gen_cmd(port, &id_node);
}

void tpu_cdma_tx_send_msg(int port, int msg_id, int wait_cnt) {
  atomic_cdma_tx_send_msg_gen_cmd(port, msg_id, wait_cnt, &id_node);
}

void tpu_cdma_nop(int port) {
  atomic_cdma_nop_gen_cmd(port, &id_node);
}

void tpu_cdma_tx_wait_msg(int port, int msg_id, int send_cnt) {
  atomic_cdma_tx_wait_msg_gen_cmd(port, msg_id, send_cnt, &id_node);
}

void tpu_cdma_rx_send_msg(int port, int msg_id, int wait_cnt) {
  atomic_cdma_rx_send_msg_gen_cmd(port, msg_id, wait_cnt, &id_node);
}

void tpu_cdma_rx_wait_msg(int port, int msg_id, int send_cnt) {
  atomic_cdma_rx_wait_msg_gen_cmd(port, msg_id, send_cnt, &id_node);
}

void tpu_cdma_nop_sync(int port) {
  atomic_cdma_nop_gen_cmd(port, &id_node);
}

void tpu_sr_setup() {
  sr_setup();
}

void tpu_cdma_tx_rx_debug(int port) {
#if (defined(USING_CMODEL) && defined(DEBUG)) || (defined(USING_FW_DEBUG) && !defined(USING_CMODEL))
  u32 rd1, rd2;
  rd1 = READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 68);
  rd2 = READ_REG(CDMA_ENGINE_MAIN_CTRL(port) + 72);
  CORE_PRINT("[%d] cdma send cmd id: %d, recv cmd id: %d\n", port, rd1, rd2);
#endif
}

void tpu_sync_all_impl(int msg_id, int core_num)
{
  int msg_cnt = 3 * core_num;
  tpu_sync_start();
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_all() {
  int msg_id = get_core_sync_msg_id();
  tpu_sync_all_with_msg_id(msg_id);
}
RTM_EXPORT(tpu_sync_all);

void tpu_sync_all_with_msg_id(int msg_id)
{
  tpu_sync_all_impl(msg_id, tpu_core_num());
}

void tpu_sync_all_bdc() {
  int msg_id = get_core_sync_msg_id();
  int msg_cnt = tpu_core_num();
  tpu_sync_start();
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_all_gdma() {
  int msg_id = get_core_sync_msg_id();
  int msg_cnt = tpu_core_num();
  tpu_sync_start();
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_all_sdma() {
  int msg_id = get_core_sync_msg_id();
  int msg_cnt = tpu_core_num();
  tpu_sync_start();
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_all_hau() {
  int msg_id = get_core_sync_msg_id();
  int msg_cnt = tpu_core_num();
  tpu_sync_start();
  tpu_hau_send_msg(msg_id, msg_cnt);
  tpu_hau_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_core() {
  int msg_id = tpu_get_local_msg_id();
  int msg_cnt = 4;
  tpu_sync_start();
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_hau_send_msg(msg_id, msg_cnt);
  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_hau_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

void tpu_sync_core_with_external_engine() {
  int msg_id = tpu_get_local_msg_id();
  int msg_cnt = 5;
  tpu_sync_start();
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_vsdma_send_msg(msg_id, msg_cnt,tpu_core_index());
  tpu_hau_send_msg(msg_id, msg_cnt);
  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_hau_wait_msg(msg_id, msg_cnt);
  tpu_vsdma_wait_msg(msg_id, msg_cnt, tpu_core_index());
  tpu_sync_end();
}

void tpu_sync_core_innner() {
  int msg_id = tpu_get_local_msg_id() - tpu_core_index() * ENTRY_PER_MSG_CENTRAL;
  int msg_cnt = 4;
  tpu_sync_start();
  tpu_bdc_send_msg(msg_id, msg_cnt);
  tpu_gdma_send_msg(msg_id, msg_cnt);
  tpu_sdma_send_msg(msg_id, msg_cnt);
  tpu_hau_send_msg(msg_id, msg_cnt);

  tpu_bdc_wait_msg(msg_id, msg_cnt);
  tpu_gdma_wait_msg(msg_id, msg_cnt);
  tpu_sdma_wait_msg(msg_id, msg_cnt);
  tpu_hau_wait_msg(msg_id, msg_cnt);
  tpu_sync_end();
}

// the function which is called by runtime between two tasks
int tpu_core_barrier(int msg_id, int core_num) {
#ifdef REMOVE_POLLS_IN_LLM
  if (tpu_is_descriptor_mode())
  {
    return 0;
  }
#endif

#ifndef REMOVE_POLLS_IN_LLM
  tpu_initialize();
#endif

  tpu_sync_all_impl(msg_id, core_num);

#ifndef REMOVE_POLLS_IN_LLM
  tpu_poll();
#endif
  return 0;
}

void tpu_core_set_state(const unsigned char* state_data, int state_size){
  TPUKERNEL_ASSERT(state_size >= (int)sizeof(g_core_context));
  int old_workitem_id = g_core_context.workitem_id;
  int old_workitem_num = g_core_context.workitem_num;
  int old_msg_id = g_core_context.core_msg_id;
  int old_base_msg_id = g_core_context.base_msg_id;
  memcpy(&g_core_context, state_data, sizeof(g_core_context));
  g_core_context.core_msg_id = old_msg_id;
  g_core_context.workitem_id = old_workitem_id;
  g_core_context.workitem_num = old_workitem_num;
  g_core_context.base_msg_id = old_base_msg_id;
  //TODO set group info here
  g_core_context.group_id = 0;
  g_core_context.group_num = 1;
}

void tpu_core_get_state(unsigned char* state_data, int state_size){
  TPUKERNEL_ASSERT(state_size >= (int)sizeof(g_core_context));
  memcpy(state_data, &g_core_context, sizeof(g_core_context));
}
