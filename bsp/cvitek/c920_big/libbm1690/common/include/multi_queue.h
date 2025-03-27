#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class MultiQueue
{
private:
    size_t limit_;
    size_t que_num_;
    size_t valid_que_idx_;
    std::vector<std::queue<T>> qs_;

    bool stop_ = false;
    std::mutex mut_;
    std::condition_variable push_cond_, pop_cond_;

public:
    MultiQueue(size_t limit, size_t que_num) : limit_(limit), que_num_(que_num)
    {
        for (size_t que_idx = 0; que_idx < que_num_; ++que_idx) {
            std::queue<T> q_;
            qs_.emplace_back(q_);
        }
        valid_que_idx_ = 0;
    }

    bool push(T v,uint32_t idx)
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        push_cond_.wait(
            lock,
            [this, idx]() {
                return this->qs_[idx].size() < limit_ || stop_;
            });
        if (stop_) return false;
        qs_[idx].push(std::move(v));
        pop_cond_.notify_one();
        return true;
    }

    bool has_non_empty() {
      uint32_t start_que_idx = this->valid_que_idx_ + 1;
      for (uint32_t i = start_que_idx; i < this->qs_.size() + start_que_idx; ++i) {
        if (this->qs_[i % this->qs_.size()].size() > 0) {
          this->valid_que_idx_ = i % this->qs_.size();
          return true;
        }
      }
      return false;
    }

    std::shared_ptr<T> pop()
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        pop_cond_.wait(
            lock,
            [this]() {
                return has_non_empty() || stop_;
            });
        if (this->qs_[valid_que_idx_].empty()) return {};
        auto ret = std::make_shared<T>(std::move(this->qs_[valid_que_idx_].front()));
        this->qs_[valid_que_idx_].pop();
        push_cond_.notify_one();
        return ret;
    }
    
    std::shared_ptr<T> pop(uint32_t idx){
        std::unique_lock<decltype(mut_)> lock(mut_);
        pop_cond_.wait(
            lock,
            [this, idx]() {
                return this->qs_[idx].size() || stop_;
            });
        if (this->qs_[idx].empty()) return {};
        auto ret = std::make_shared<T>(std::move(this->qs_[idx].front()));
        this->qs_[idx].pop();
        push_cond_.notify_one();
        return ret;
    }
    void join()
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        stop_ = true;
        push_cond_.notify_all();
        pop_cond_.notify_all();
    }

    void reset()
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        stop_ = false;
    }
};
