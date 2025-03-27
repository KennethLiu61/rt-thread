#ifndef RANK_TABLE_H
#define RANK_TABLE_H
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "common.h"
#include <sstream>

#define RANK_TABLE_FILE "RANK_TABLE_FILE"
#define NODE_LIST "node_list"
#define DEVICE "device_list"
#define RANK_ID "rank"
#define DEVICE_ID "device_id"
#define DEVICE_IP "device_ip_emulator"

using json = nlohmann::json;

static int safe_get_int(json j, std::string key) {
    if(j[key].is_string()) {
        return std::stoi(j[key].get<std::string>());
    } else if (j[key].is_number()) {
        return j[key].get<int>();
    } else {
#ifdef DEBUG_RANK_TABLE
        std::cout << key << " is not a number\n" << std::endl;
#endif
        return -1;
    }
}

static int rank_table_valid() {
    char *path = getenv(RANK_TABLE_FILE);
    if (path == NULL) {
#ifdef DEBUG_RANK_TABLE
        std::cout << "RANK_TABLE_FILE not set\n" << std::endl;
#endif
        return 0;
    }
    if (access(path, F_OK) == -1) {
#ifdef DEBUG_RANK_TABLE
        std::cout << "file not exist\n" << std::endl;
#endif
        return 0;
    }
    return 1;
}

static int get_device_id(json j, int rank_id) {
    for(json::iterator s = j[NODE_LIST].begin(); s != j[NODE_LIST].end(); ++s) {
        json node = *s;
        for (json::iterator it = node[DEVICE].begin(); it != node[DEVICE].end(); ++it) {
            if(rank_id == safe_get_int(*it, RANK_ID)) return safe_get_int(*it, DEVICE_ID);
        }
    }
    return -1;
}

static std::string get_device_ip(json j, int dev_id) {
    for(json::iterator s = j[NODE_LIST].begin(); s != j[NODE_LIST].end(); ++s) {
        json node = *s;
        for (json::iterator it = node[DEVICE].begin(); it != node[DEVICE].end(); ++it) {
            if(dev_id == safe_get_int(*it, DEVICE_ID)) {
                if ((*it).find(DEVICE_IP) != (*it).end() && !(*it)[DEVICE_IP].get<std::string>().empty()){
                    return (*it)[DEVICE_IP].get<std::string>();
                }
                else{
#ifdef DEBUG_RANK_TABLE
                    std::cout << "WARNING: DEVICE_IP is empty!\n" << std::endl;
#endif
                    return std::string();
                }
            }
        }
    }
    return std::string();
}

static int get_device_id() {
    char *rank = getenv("OMPI_COMM_WORLD_RANK");
    if (rank == nullptr) {
        rank = getenv("LOCAL_RANK");
        if (rank == nullptr) {
            return 0;
        }
    }
    if(rank_table_valid() == 0) return atoi(rank);
    std::ifstream f(getenv(RANK_TABLE_FILE));
    json j = json::parse(f);
    return get_device_id(j, std::stoi(rank));
}

static int get_device_id(int rank_id) {
    if(rank_table_valid() == 0) return rank_id;
    std::ifstream f(getenv(RANK_TABLE_FILE));
    json j = json::parse(f);
    return get_device_id(j, rank_id);
}

static int get_rank_id(json j, int dev_id) {
    if(rank_table_valid() == 0) return dev_id;
    for(json::iterator s = j[NODE_LIST].begin(); s != j[NODE_LIST].end(); ++s) {
        json node = *s;
        for (json::iterator it = node[DEVICE].begin(); it != node[DEVICE].end(); ++it) {
            if(dev_id == safe_get_int(*it, DEVICE_ID)) return safe_get_int(*it, RANK_ID);
        }
    }
    return -1;
}

static int get_rank_id(int dev_id) {
    if(rank_table_valid() == 0) return dev_id;
    std::ifstream f(getenv(RANK_TABLE_FILE));
    json j = json::parse(f);
    return get_rank_id(j, dev_id);
}

static int get_device_num() {
    char *size = getenv("OMPI_COMM_WORLD_SIZE");
    if (size == nullptr) {
        size = getenv("LOCAL_WORLD_SIZE");
        if (size == nullptr) {
            return 1;
        }
    }
    return atoi(size);
}

static int get_group_id() {
  int group_idx = 0;
  const char* group_idx_env = getenv("TPU_GROUP_IDX");
  if ( group_idx_env ) group_idx = atoi(group_idx_env);
  return group_idx;
}
static int get_group_num() {
  int group_num = 1;
  const char* group_num_env = getenv("TPU_GROUP_NUM");
  if ( group_num_env ) group_num = atoi(group_num_env);
  return group_num;
}

static std::string get_device_ip(int dev_id) {
    std::ifstream f(getenv(RANK_TABLE_FILE));
    json j = json::parse(f);
    return get_device_ip(j, dev_id);
}

static std::vector<int> get_use_chip_list() {
    const char* use_chip_env = getenv("CHIP_MAP");
    std::string use_chip = use_chip_env ? use_chip_env : "";
    std::vector<int> use_chip_list;
    if(use_chip.empty()) {
        int device_num = get_device_num();
        for(int i = 0; i < device_num; i++) {
            use_chip_list.push_back(i);
        }
        return use_chip_list;
    }
    std::stringstream ss(use_chip);
    std::string item;
    while (std::getline(ss, item, ',')) {
        use_chip_list.push_back(std::stoi(item));
    }
    return use_chip_list;
}

static void make_chip_map(int* chip_map) {
    int device_num = get_device_num();
    std::vector<int> use_chip_list;

    if(rank_table_valid() == 0) {
        use_chip_list = get_use_chip_list();
        assert((int)use_chip_list.size() >= device_num);
    }

    for(int i = 0; i < device_num; i++) {
        chip_map[i] = rank_table_valid() == 0 ? use_chip_list[i] : get_device_id(i);
    }
}

static int get_use_ring() {
    int use_ring= atoi(getenv("USE_RING") ? getenv("USE_RING") : "1");
    return use_ring;
}
#endif