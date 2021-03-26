#include <compress/reordering/ThreadPool.h>
#include <cassert>

namespace compress {

ThreadPool::~ThreadPool() {
  stop();
  waitStopped();
}

bool ThreadPool::start(uint32_t nthread, uint32_t queueSize) {
  assert(queueSize != 0 && _taskQueue == nullptr);
  if (nthread == 0) {
    nthread = std::thread::hardware_concurrency();
    if (nthread == 0) {
      return false;
    }
  }

  _unfinishedTaskCount = 0;
  _taskQueue = std::make_unique<TaskQueue>(queueSize);
  _unlimitQueue = std::make_unique<TaskQueue>(UINT32_MAX);
  _threads.resize(nthread);

  bool launchFailed = false;
  for (uint32_t i = 0; i < nthread; i++) {
    try {
      auto thread = std::make_unique<std::thread>([this]() { doTaskCycle(); });
      _threads.emplace_back(std::move(thread));
    } catch (const std::system_error&) {
      launchFailed = true;
      break;
    }
  }
  try {
    auto thread = std::make_unique<std::thread>([this]() { pushTaskCycle(); });
    _threads.emplace_back(std::move(thread));
  } catch (const std::system_error&) {
    launchFailed = true;
  }
  if (launchFailed) {
    stop();
    waitStopped();
  }
  return !launchFailed;
}

void ThreadPool::stop() const {
  _stop = true;
  _taskQueue->wakeupAll();
  _unlimitQueue->wakeupAll();
}

void ThreadPool::waitStopped() {
  for (const auto& t : _threads) {
    if (t && t->joinable()) {
      t->join();
    }
  }
  _threads.clear();
}

void ThreadPool::waitAllFinished() const {
  std::unique_lock<std::mutex> lockGuard(_unfinishedTaskLock);
  _unfinishedTaskCondVar.wait(lockGuard, [this](){
    return _unfinishedTaskCount == 0;});
}

uint32_t ThreadPool::pendingTaskCount() const {
  return  (_taskQueue ? _taskQueue->size() : 0) +
          (_unlimitQueue ? _unlimitQueue->size() : 0);
}

bool ThreadPool::push(Task task, bool unlimit) {
  bool ret;
  if (unlimit) {
    ret = _unlimitQueue->push(std::move(task));
  } else {
    ret = _taskQueue->push(std::move(task));
  }
  if (ret) {
    std::unique_lock<std::mutex> lockGuard(_unfinishedTaskLock);
    _unfinishedTaskCount += 1;
  }
  return ret;
}

void ThreadPool::doTaskCycle() const {
  constexpr const uint32_t timeout = 500;
  while (!_stop) {
    Task task;
    if (!_taskQueue->pop(timeout, &task)) {
      continue;
    }
    task();
    std::unique_lock<std::mutex> lockGuard(_unfinishedTaskLock);
    --_unfinishedTaskCount;
    _unfinishedTaskCondVar.notify_all();
  }
}

void ThreadPool::pushTaskCycle() const {
  constexpr const uint32_t timeout = 500;
  while (!_stop) {
    Task task;
    if (!_unlimitQueue->pop(timeout, &task)) {
      continue;
    }
    _taskQueue->push(std::move(task));
  }
}


}
