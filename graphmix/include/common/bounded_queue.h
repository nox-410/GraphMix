#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <utility>
#include "ps/base.h"

template<typename T> class ThreadsafeBoundedQueue {
 public:
  ThreadsafeBoundedQueue(size_t limit) : limit_(limit) { }
  ~ThreadsafeBoundedQueue() { }

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void Push(T new_value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond2_.wait(lk, [this] {return queue_.size() < limit_; });
    queue_.push(std::move(new_value));
    cond_.notify_one();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void WaitAndPop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this]{return !queue_.empty();});
    *value = std::move(queue_.front());
    queue_.pop();
    cond2_.notify_one();
  }

  bool TryPop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      return false;
    } else {
      *value = std::move(queue_.front());
      queue_.pop();
      cond2_.notify_one();
      return true;
    }
  }
  const size_t limit_;
 private:
  mutable std::mutex mu_, mu2_;
  std::queue<T> queue_;
  std::condition_variable cond_, cond2_;
};

