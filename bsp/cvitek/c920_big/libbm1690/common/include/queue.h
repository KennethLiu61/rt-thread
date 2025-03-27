#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class Queue
{
private:
    size_t limit_;
    std::queue<T> q_;

    bool stop_ = false;
    std::mutex mut_;
    std::condition_variable push_cond_, pop_cond_;

public:
    Queue(size_t limit) : limit_(limit)
    {
    }

    bool push(T v)
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        push_cond_.wait(
            lock,
            [this]() {
                return this->q_.size() < limit_ || stop_;
            });
        if (stop_) return false;
        q_.push(std::move(v));
        pop_cond_.notify_one();
        return true;
    }

    std::shared_ptr<T> pop()
    {
        std::unique_lock<decltype(mut_)> lock(mut_);
        pop_cond_.wait(
            lock,
            [this]() {
                return this->q_.size() || stop_;
            });
        if (this->q_.empty()) return {};
        auto ret = std::make_shared<T>(std::move(this->q_.front()));
        this->q_.pop();
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
