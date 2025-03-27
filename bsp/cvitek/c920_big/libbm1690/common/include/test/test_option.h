#pragma once
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <getopt.h>
#include <stdio.h>
#include "common_def.h"
#include "common_utils.h"

class TestOption {
    typedef enum {
      OPTION_NONE,
      OPTION_INT,
      OPTION_DTYPE,
      OPTION_ENUM,
      OPTION_INT_LIST,
      OPTION_FLOAT,
      OPTION_INT64,
      OPTION_HEX64,
      OPTION_HEX32,
      OPTION_STRING,
      //step 0: define a new dtype
    } OPTION_DTYPE_T;

  public:
    TestOption() {
      add_none("help", 'h', "show this help");
      scale_value = 1;
      auto scale_str = getenv("OPTION_SCALE");
      if (scale_str) scale_value = atoi(scale_str);
    }

    // value float
    TestOption& vf(void* value_ptr, const std::string& name, char short_name = '\0'){
      return va(value_ptr, name, short_name, OPTION_FLOAT);
    }

    // value list
    TestOption& vl(void* value_ptr, int* size_ptr, int max_size, const std::string& name, char short_name = '\0'){
      list_sizes[name] = {max_size, size_ptr};
      return va(value_ptr, name, short_name, OPTION_INT_LIST);
    }

    // value bool
    TestOption& vb(void* value_ptr, const std::string& name, char short_name = '\0'){
      return va(value_ptr, name, short_name).c({0,1});
    }
    // value enum
    TestOption& ve(void* value_ptr, const std::string& name, char short_name = '\0'){
      return va(value_ptr, name, short_name, OPTION_ENUM);
    }
    // value int
    TestOption& vi(void* value_ptr, const std::string& name, char short_name = '\0'){
      return va(value_ptr, name, short_name, OPTION_INT);
    }

    // value all
    TestOption& va(void* value_ptr, const std::string& name, char short_name = '\0', OPTION_DTYPE_T ot = OPTION_INT){
      if(short_name == '\0') short_name = name.at(0);
      unique_push_back(names, name);
      unique_push_back(short_names, short_name);
      unique_push_back(pointers, value_ptr);
      helps.push_back(name);
      types.push_back(ot);
      can_scales.push_back(false);
      return *this;
    }

    // option type
    TestOption& t(OPTION_DTYPE_T ot = OPTION_DTYPE){
      types.back() = ot;
      return  *this;
    }
    // enum_list
    TestOption& el(const std::vector<std::string>& enum_vec){
      std::map<int, std::string> enum_map;
      for(size_t i = 0; i<enum_vec.size(); i++){
        enum_map[i] = enum_vec[i];
      }
      enum_maps[names.back()] = enum_map;
      return *this;
    }
    // enum
    TestOption& e(const std::map<int, std::string>& enum_map){
      enum_maps[names.back()] = enum_map;
      return *this;
    }
    // choices
    TestOption& c(const std::set<int>& choice){
      if(!choice.empty()) choices[names.back()] = choice;
      return *this;
    }

    // range
    TestOption& r(int min, int max){
      if(min<max) ranges[names.back()] = {min, max};
      return *this;
    }

    // range float
    TestOption& rf(float min, float max){
      if(min<max) franges[names.back()] = {min, max};
      return *this;
    }

    // help
    TestOption& h(const std::string& help){
      if(help!="") helps.back() = help;
      return *this;
    }

    // can_scale
    TestOption& s(){
      return can_scale();
    }

    TestOption& can_scale(){
      can_scales.back() = true;
      return *this;
    }

    TestOption& add_enum_by_list(int* value_ptr, const char* name, char short_name, const std::string& help="", const std::vector<std::string>& enum_vec={}) {
      std::map<int, std::string> enum_map;
      for(size_t i = 0; i<enum_vec.size(); i++){
        enum_map[i] = enum_vec[i];
      }
      return add_enum(value_ptr, name, short_name, help, enum_map);
    }

    TestOption& add_enum(int* value_ptr, const char* name, char short_name, const std::string& help="", const std::map<int, std::string>& enum_map={}) {
      enum_maps[name] = enum_map;
      return add_option(name, short_name, value_ptr, OPTION_ENUM, help);
    }

    TestOption& add_choice(int* value_ptr, const char* name, char short_name, const std::string& help="", const std::set<int>& choice={}) {
      return add_option(name, short_name, value_ptr, OPTION_INT, help, choice);
    }
    TestOption& add_fp_dtype(int* value_ptr, const char *name="dtype", char short_name='d', const std::string& help=""){
      return add_dtype(value_ptr, name, short_name, help, {SG_DTYPE_FP32, SG_DTYPE_FP16, SG_DTYPE_BFP16});
    }
    TestOption& add_bool(int* value_ptr, const char *name, char short_name, const std::string& help=""){
      return add_choice(value_ptr, name, short_name, help, {0,1});
    }

    TestOption& add_dtype(int* value_ptr, const char *name="dtype", char short_name='d', const std::string& help="", const std::set<int>& choices={}){
      return add_option(name, short_name, value_ptr, OPTION_DTYPE, help, choices);
    }
    TestOption& add_none(const char *name, char short_name, const std::string& help = "") {
      return add_option(name, short_name, nullptr, OPTION_NONE, help);
    }
    TestOption& add_seed(int* value_ptr, char flag = 'S') {
      *value_ptr = time(0);
      return add_int(value_ptr, "seed", flag, "random seed");
    }
    TestOption& add_loop(int* value_ptr, char flag = 'L') {
      return add_int(value_ptr, "loop", flag, "loop num for test");
    }

    TestOption& add_NCHW(int* N, int* C, int* H, int* W, const char* help_prefix="") {
      std::string help(help_prefix);
      return add_int(N, "batch", 'N', help+ " N")
          .add_int(C, "channel", 'C', help+ " C")
          .add_int(H, "height", 'H', help+ " H")
          .add_int(W, "width", 'W', help+ " W");
    }

    TestOption& add_list(int* list_ptr, int* len, int max_size, const char* name, char short_name, const std::string& help = ""){
      list_sizes[name] = {max_size, len};
      return add_option(name, short_name, list_ptr, OPTION_INT_LIST, help);
    }

    TestOption& add_string(std::string* value_ptr, const char *name, char short_name, const std::string& help="") {
      return add_option(name, short_name, (void*)value_ptr, OPTION_STRING);
    }

    std::string info_by_index(int i, bool is_kv = true){
        std::string info;
        if(is_kv) info += names[i]+"(-"+short_names[i]+")=";
        if(types[i] == OPTION_ENUM) {
          int value = *(int *)pointers[i];
          auto &enum_map = enum_maps[names[i]];
          info += value_to_string(value, types[i], enum_map[value]);
        } else if(types[i] == OPTION_INT_LIST) {
          auto& list_size = list_sizes[names[i]];
          auto len = list_size.second?(*list_size.second): list_size.first;
          auto list = (int*)pointers[i];
          info += list_to_string(list, len);
        } else if(types[i] == OPTION_FLOAT) {
          float value = *(float*)pointers[i];
          info += value_to_string(value, types[i]);
        } else if(types[i] == OPTION_HEX32) {
          auto value = *(uint32_t*)pointers[i];
          info += value_to_string(value, types[i]);
        } else if(types[i] == OPTION_HEX64) {
          auto value = *(uint64_t*)pointers[i];
          info += value_to_string(value, types[i]);
        } else if(types[i] == OPTION_INT64) {
          auto value = *(int64_t*)pointers[i];
          info += value_to_string(value, types[i]);
        } else if (types[i] == OPTION_STRING){
          auto value = *(std::string *)pointers[i];
          info += value;
        } else {
          int value = *(int *)pointers[i];
          info += value_to_string(value, types[i]);
        }
        // step 2: add info
        return info;
    }
    TestOption& add_int(int* value_ptr, const char *name, char short_name, const std::string& help = "", std::pair<int, int> range={0,0}) {
      return add_option(name, short_name, value_ptr, OPTION_INT, help, {}, range);
    }
    TestOption& add_float(float* value_ptr, const char *name, char short_name, const std::string& help = "", std::pair<int, int> range={0,0}) {
      if(range.first<range.second) franges[name] = range;
      return add_option(name, short_name, value_ptr, OPTION_FLOAT, help);
    }
    TestOption& add_int64(int64_t* value_ptr, const char *name, char short_name, const std::string& help = "") {
      return add_option(name, short_name, value_ptr, OPTION_INT64, help, {});
    }
    TestOption& add_hex64(uint64_t* value_ptr, const char *name, char short_name, const std::string& help = "") {
      return add_option(name, short_name, value_ptr, OPTION_HEX64, help, {});
    }
    TestOption& add_hex32(uint32_t* value_ptr, const char *name, char short_name, const std::string& help = "") {
      return add_option(name, short_name, value_ptr, OPTION_HEX32, help, {});
    }

    // step 0.5: add_xxx
    bool parse(int argc, char *argv[], const std::string& seed_name="seed") {
      bool changed = false;
      std::vector<struct option> option_list;

      std::string short_str;
      for (size_t i = 0; i < names.size(); i++) {
        struct option o;
        o.has_arg = types[i] == OPTION_NONE? no_argument : required_argument;
        o.name = names[i].c_str();
        o.val = short_names[i];
        o.flag = nullptr;
        option_list.push_back(o);
        short_str += short_names[i];
        if (types[i] != OPTION_NONE) {
          short_str += ':';
        }
      }
      int opt_idx = 0;
      int opt_val = 0;
      std::set<int> parsed_index;
      std::cout<<"===================================================="<<std::endl;
      while (-1 != (opt_val = getopt_long(argc, argv, short_str.c_str(), option_list.data(), &opt_idx))) {
        if (opt_val == 'h') {
          show_help();
          exit(0);
        }
        auto i =std::distance(short_names.begin(), std::find(short_names.begin(), short_names.end(), opt_val));
        parsed_index.insert(i);
        if (types[i] == OPTION_NONE) continue;;
        if (types[i] == OPTION_INT_LIST) {
          int *plist = static_cast<int *>(pointers[i]);
          auto size_info = list_sizes[names[i]];
          int max_len = size_info.first;
          int* plen = size_info.second;
          int real_len = string_to_list(optarg, plist, max_len);
          for(int j=0; j<real_len; j++){
            if(can_scales[i]) plist[j] *= scale_value;
            check_value(names[i], plist[i], argv[0]);
          }
          if(!plen) {
            if(max_len != real_len){
              std::cerr<<"parsed list len is invalid: real_len="<<real_len<<" vs "<<max_len<<std::endl;
              throw;
            }
          } else {
            *plen = real_len;
          }
          //step 3: add parse method
        } else if(types[i] == OPTION_FLOAT) {
          float *pvalue = static_cast<float*>(pointers[i]);
          *pvalue = atof(optarg);
          if(can_scales[i]) *pvalue *= scale_value;
          check_value(names[i], *pvalue, argv[0]);
        } else if(types[i] == OPTION_HEX64) {
          auto pvalue = static_cast<uint64_t*>(pointers[i]);
          *pvalue = std::strtoull(optarg, nullptr, 16);
          if(can_scales[i]) *pvalue *= scale_value;
          check_value(names[i], *pvalue, argv[0]);
        } else if(types[i] == OPTION_HEX32) {
          auto pvalue = static_cast<uint32_t*>(pointers[i]);
          *pvalue = std::strtoul(optarg, nullptr, 16);
          if(can_scales[i]) *pvalue *= scale_value;
          check_value(names[i], *pvalue, argv[0]);
        } else if(types[i] == OPTION_INT64) {
          auto pvalue = static_cast<int64_t*>(pointers[i]);
          *pvalue = std::strtoll(optarg, nullptr, 10);
          if(can_scales[i]) *pvalue *= scale_value;
          check_value(names[i], *pvalue, argv[0]);
        } else if (types[i] == OPTION_STRING) {
          auto *pvalue = static_cast<std::string *>(pointers[i]);
          *pvalue = optarg;
        } else {
          int *pvalue = static_cast<int *>(pointers[i]);
          *pvalue = atoi(optarg);
          if(can_scales[i]) *pvalue *= scale_value;
          check_value(names[i], *pvalue, argv[0]);
        }
        if(names[i] == seed_name) {
          int *pvalue = static_cast<int *>(pointers[i]);
          using_seed(*pvalue);
        }
        changed = true;
        std::cout << "option " << info_by_index(i)<<std::endl;
      }
      for(size_t i=0; i<names.size(); i++){
        if(parsed_index.count(i)) continue;
        if (types[i] == OPTION_NONE) continue;
        std::cout << "default " << info_by_index(i)<<std::endl;
        if(names[i] == seed_name) {
          int *pvalue = static_cast<int *>(pointers[i]);
          using_seed(*pvalue);
        }
      }
      std::cout<<"===================================================="<<std::endl;
      return changed;
    }
    void show_help(const char* prefix = "  ")
    {
      std::cout<<"Args:"<<std::endl;
      for (size_t i = 0; i < names.size(); i++) {
        if(types [i] == OPTION_NONE){
          std::cout << prefix << "--" << names[i] << "(-" << short_names[i] << "): " << helps[i];
        } else if(types [i] == OPTION_INT_LIST) {
          std::cout << prefix << "--" << names[i] << "(-" << short_names[i] << ") x,y,z..: " << helps[i];
        } else {
          std::cout << prefix << "--" << names[i] << "(-" << short_names[i] << ") xxx: " << helps[i];
        }
        if(types[i] == OPTION_NONE) {
          std::cout<<std::endl;
          continue;
        }
        if(types[i] == OPTION_DTYPE) {
          std::cout <<", ";
          if(choices.count(names[i])){
            auto choice = choices[names[i]];
            for(auto c: choice){
              std::cout << value_to_string(c, types[i]);
            }
          } else {
            for(int c=0; c<=SG_DTYPE_UINT4; c++){
              std::cout << value_to_string(c, types[i]) << " ";
            }
          }
        }

        if(types[i] == OPTION_INT){
          if(choices.count(names[i])){
            auto& choice = choices[names[i]];
            std::cout <<", choices={";
            for(auto c: choice){
              std::cout <<c<< ",";
            }
            std::cout<<"}";
          }
          if(ranges.count(names[i])){
            auto& range = ranges[names[i]];
            std::cout << ", range=["<<range.first<<", "<<range.second<<")";
          }
        }
        if(types[i] == OPTION_FLOAT){
          if(franges.count(names[i])){
            auto& range = franges[names[i]];
            std::cout << ", range=["<<range.first<<", "<<range.second<<")";
          }
        }
        if(types[i] == OPTION_STRING) {
          if (choices.count(names[i])) {
            auto& choice = choices[names[i]];
            for(auto c: choice){
              std::cout << c;
            }
            std::cout << ", ";
          } else {
            std::cout <<" ";
          }
        }
        // step 1: add show help info
        if(types[i] == OPTION_ENUM) {
          auto& enum_map = enum_maps[names[i]];
          std::cout<< ", ";
          for(auto item: enum_map){
            std::cout<< value_to_string(item.first, types[i], item.second)<<" ";
          }
        }

        std::cout << ", default " << info_by_index(i, false);

        if(types[i] == OPTION_INT_LIST) {
          auto size_info = list_sizes[names[i]];
          if(size_info.second){
            std::cout << ", max_len=" << size_info.first;
          } else {
            std::cout << ", fixed_len=" << size_info.first;
          }
        }
        std::cout<< std::endl;
      }
    }

    template<typename T>
    std::string value_to_string(T value, OPTION_DTYPE_T dtype, const std::string& name = "") {
      std::string value_str = "";
      if(dtype == OPTION_DTYPE) {
        if((int)value != -1) {
          value_str += std::to_string(value)+"("+sg_dtype_name((sg_data_type_t)value)+")";
        } else {
          value_str += std::to_string(value)+"(random)";
        }
      } else if(dtype == OPTION_ENUM) {
          value_str += std::to_string(value)+"("+name+")";
      } else if(dtype == OPTION_HEX64){
        char value_cstr[24];
        snprintf(value_cstr, sizeof(value_cstr), "0x%lx", (uint64_t)value);
        value_str += value_cstr;
      } else if(dtype == OPTION_HEX32){
        char value_cstr[24];
        snprintf(value_cstr, sizeof(value_cstr), "0x%x", (uint32_t)value);
        value_str += value_cstr;
      } else {
        value_str += std::to_string(value);
      }
      return value_str;
    }

    private:
    TestOption& add_option(const char *name, char short_name, void *value_ptr, OPTION_DTYPE_T dtype, const std::string& help = "", const std::set<int>& choice={}, std::pair<int, int> range = {0,0}) {
      return va(value_ptr, name, short_name, dtype).h(help).c(choice).r(range.first, range.second);
    }

    std::vector<OPTION_DTYPE_T> types;
    std::vector<std::string> names;
    std::vector<char> short_names;
    std::vector<std::string> helps;
    std::vector<void *> pointers;
    std::map<std::string, std::pair<int, int>> ranges;
    std::map<std::string, std::pair<float, float>> franges;
    std::map<std::string, std::set<int>> choices;
    std::map<std::string, std::map<int, std::string>> enum_maps;
    std::map<std::string, std::pair<int, int*>> list_sizes;
    std::vector<bool> can_scales;
    int scale_value = 1;

    void check_value(const std::string& name, float value){
      if(franges.count(name)){
        auto& range = franges[name];
        if(value < range.first || value>= range.second){
          std::cerr<<"[ERROR] Invalid value for "<<name<<" = "<<value<<std::endl;
          show_help();
          exit(-1);
        }
      }
    }
    void check_value(const std::string& name, int value, const std::string& program_name){
      if (choices.count(name)) {
        auto& choice = choices[name];
        if(!choice.count(value)) {
          std::cerr<<"[ERROR] Invalid value for "<<name<<" = "<<value<< " in " << program_name<< std::endl;
          show_help();
          exit(-1);
        }
      }
      if(ranges.count(name)){
        auto& range = ranges[name];
        if(value < range.first || value>= range.second){
          std::cerr<<"[ERROR] Invalid value for "<<name<<" = "<<value<< " in " << program_name<< std::endl;
          show_help();
          exit(-1);
        }
      }
      if(enum_maps.count(name)){
        auto& enum_map = enum_maps[name];
        if(!enum_map.count(value)){
          std::cerr<<"[ERROR] Invalid value for "<<name<<" = "<<value<< " in " << program_name<< std::endl;
          show_help();
          exit(-1);
        }
      }
    }
};

