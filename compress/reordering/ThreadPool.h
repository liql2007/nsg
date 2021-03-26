#ifndef EFANNA2E_THREADPOOL_H
#define EFANNA2E_THREADPOOL_H

#include <compress/reordering/TaskSyncQueue.h>
#include <thread>

namespace compress {

class ThreadPool {
public:
  using Task = std::function<void()>;

  ThreadPool() = default;

  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;

  ThreadPool &operator=(const ThreadPool &) = delete;

  bool start(uint32_t nthread, uint32_t queueSize);

  void stop() const;

  void waitStopped();

  void waitAllFinished() const;

  uint32_t pendingTaskCount() const;

  bool push(Task task, bool unlimit = false);

private:
  void doTaskCycle() const;

  void pushTaskCycle() const;

private:
  using TaskQueue = TaskSyncQueue<Task>;

  std::unique_ptr<TaskQueue> _taskQueue;
  std::unique_ptr<TaskQueue> _unlimitQueue;
  std::vector <std::unique_ptr<std::thread>> _threads;
  volatile mutable bool _stop { false };

  mutable uint32_t _unfinishedTaskCount { 0 };
  mutable std::condition_variable _unfinishedTaskCondVar;
  mutable std::mutex _unfinishedTaskLock;
};

}


#endif //EFANNA2E_THREADPOOL_H
