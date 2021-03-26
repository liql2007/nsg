//
// Created by liql2007 on 2021/3/23.
//

#ifndef EFANNA2E_TASKSYNCQUEUE_H
#define EFANNA2E_TASKSYNCQUEUE_H

#include <mutex>
#include <queue>
#include <condition_variable>

namespace compress {

template<typename T>
class TaskSyncQueue {
public:
  TaskSyncQueue(size_t queueSize) : _queueSize(queueSize) {}

  TaskSyncQueue(const TaskSyncQueue &) = delete;

  TaskSyncQueue &operator=(const TaskSyncQueue &) = delete;

  bool push(T task) {
    bool succ = false;
    {
      std::unique_lock <std::mutex> lock(_mutex);
      if (_queueSize > 0) {
        _pushCond.wait(lock,
                       [this]() {
                         return _queue.size() < _queueSize || _frozen;
                       });
      }
      if (!_frozen) {
        _queue.push(std::move(task));
        succ = true;
      }
    }
    if (succ) {
      _popCond.notify_one();
    }
    return succ;
  }

  bool push(const T &task, uint32_t timeout) {
    bool succ = false;
    std::chrono::duration<uint64_t, std::milli> expire(timeout);
    {
      std::unique_lock <std::mutex> lock(_mutex);
      if (_queueSize == 0
          || _pushCond.wait_for(lock, expire,
                                [this]() {
                                  return _queue.size() < _queueSize || _frozen;
                                })) {
        if (!_frozen) {
          _queue.push(std::move(task));
          succ = true;
        }
      }
    }
    if (succ) {
      _popCond.notify_one();
    }
    return succ;
  }

  T pop() {
    T v{};
    bool succ = false;
    {
      std::unique_lock <std::mutex> lock(_mutex);
      _popCond.wait(lock,
                    [this]() {
                      return !_queue.empty() || _frozen;
                    });
      if (!_frozen) {
        v = std::move(_queue.front());
        _queue.pop();
        succ = true;
      }
    }
    if (succ) {
      _pushCond.notify_one();
    }
    return v;
  }

  bool pop(uint32_t timeout, T *v) {
    bool succ = false;
    std::chrono::duration<uint64_t, std::milli> expire(timeout);
    {
      std::unique_lock <std::mutex> lock(_mutex);
      if (_popCond.wait_for(lock, expire,
                            [this]() {
                              return !_queue.empty() || _frozen;
                            })) {
        if (!_frozen) {
          *v = std::move(_queue.front());
          _queue.pop();
          succ = true;
        }
      }
    }
    if (succ) {
      _pushCond.notify_one();
    }
    return succ;
  }

  size_t size() const {
    std::lock_guard <std::mutex> lock(_mutex);
    return _queue.size();
  }

  void freeze() {
    _mutex.lock();
    _frozen = true;
    _mutex.unlock();
    wakeupAll();
  }

  void unfreeze() {
    _mutex.lock();
    _frozen = false;
    _mutex.unlock();
    wakeupAll();
  }

  bool isFrozen() const {
    std::lock_guard <std::mutex> lock(_mutex);
    return _frozen;
  }

  void wakeupAll() const {
    _pushCond.notify_all();
    _popCond.notify_all();
  }

  template<typename Container>
  void stealData(Container *container) {
    std::lock_guard <std::mutex> lock(_mutex);
    while (!_queue.empty()) {
      container->push_back(std::move(_queue.front()));
      _queue.pop();
    }
  }

  void swapData(std::queue <T> *output) {
    std::lock_guard <std::mutex> lock(_mutex);
    output->swap(_queue);
  }

private:
  mutable std::mutex _mutex;
  mutable std::condition_variable _pushCond;
  mutable std::condition_variable _popCond;
  size_t _queueSize{0};
  volatile bool _frozen{false};
  std::queue <T> _queue;
};

}

#endif //EFANNA2E_TASKSYNCQUEUE_H
