#pragma once
#include <stdlib.h>
#include "bmlib_runtime.h"
class TestDevice {
public:
  TestDevice() {
    auto device_str = getenv("USING_DEVICE");
    if (device_str) _device_id = atoi(device_str);
    auto ret = bm_dev_request(&_handle, _device_id);
    if(BM_SUCCESS != ret) {
      _handle = nullptr;
    }
  }
  bm_handle_t handle() const {
    if(!_handle){
      throw "device open failed!";
    }
    return _handle;
  }
  int device_id() const {
    return _device_id;
  }
  int core_num() const {
    return _core_num;
  }
  ~TestDevice(){
    bm_dev_free(_handle);
  }
private:
  bm_handle_t _handle = nullptr;
  int _device_id = 0;
  int _core_num = 1;
};

