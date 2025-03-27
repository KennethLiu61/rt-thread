#pragma once
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <string>
#include <algorithm>

#define DIV_UP(a, b) (((a) - 1) / (b) + 1)

static int int4_to_value(unsigned char data, int is_upper, int is_signed){
  int int4_value = is_upper?(data>>4):(data&0xF);
  if(is_signed){
    int4_value = (int4_value<<28)>>28;
  }
  return int4_value;
}

static void using_seed(int seed){
  srand(seed);
  std::cout << "Using seed = " << seed << std::endl;
}

template<typename KeyType>
KeyType random_key(const std::map<KeyType, std::string>& m) {
  int index = rand()%m.size();
  auto iter = m.begin();
  std::advance(iter, index);
  return iter->first;
}

template<typename KeyType>
KeyType random_choose(const std::set<KeyType>& m) {
  int index = rand()%m.size();
  auto iter = m.begin();
  std::advance(iter, index);
  return *iter;
}


template<typename T>
void unique_push_back(std::vector<T>& vec, const T& v){
  if (std::find(vec.begin(), vec.end(), v) != vec.end()) {
    std::cerr << "[ERROR] " << v << " is duplicated!" << std::endl;
    throw;
  }
  vec.push_back(v);
}

static std::string mask_to_str(int mask, int dims) {
  std::string mask_str = "";
  for (int i = 0; i < dims; ++i) {
    mask_str += std::to_string(!!(mask & (1 << i)));
  }
  return mask_str;
}

template<typename T>
std::string list_to_string(const T* list, int len) {
  std::string list_str = "[";
  for(int i=0; i<len-1; i++){
    list_str += std::to_string(list[i])+",";
  }
  if(len>0){
    list_str += std::to_string(list[len - 1]);
  }
  list_str += "]";
  return list_str;
}

template<typename T>
int string_to_list(const std::string& str, T* list, int max_len) {
  std::string number = "";
  int start = 0;
  for(auto c: str){
    if(std::isdigit(c)){
      number += c;
    } else if(!number.empty()){
      if(start >= max_len) {
        std::cerr<<"List is to long: list="<<str<<", max_len"<<max_len<<std::endl;
        throw;
      }
      list[start++] = atoi(number.c_str());
      number = "";
    }
  }
  if (!number.empty()) {
    if (start >= max_len) {
      std::cerr << "List is to long: list=" << str << ", max_len" << max_len << std::endl;
      throw;
    }
    list[start++] = atoi(number.c_str());
  }
  return start;
}

static inline int update_indice(int* indice, const int* limit, int len){
    int carry_value = 1;
    for(int i=len-1; i>=0; i--){
        indice[i] += carry_value;
        if(indice[i]>=limit[i]){
            carry_value = 1;
            indice[i] = 0;
        } else {
            carry_value = 0;
            break;
        }
    }
    return carry_value == 0;
}
