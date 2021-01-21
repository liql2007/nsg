//
// Created by liql2007 on 2021/1/21.
//

#ifndef EFANNA2E_PRIORITYQUEUE_H
#define EFANNA2E_PRIORITYQUEUE_H

#include <algorithm>

namespace efanna2e {

template<typename T,
  typename Comparator = std::less <T>,
  typename Container = std::vector <T>>
class PriorityQueue {
public:
  using DataComparator = Comparator;
  using DataContainer = Container;

  PriorityQueue(Container *seq)
    : _seq(seq) {
    _seq->clear();
  }

  ~PriorityQueue() = default;

  void push(T v) {
    _seq->push_back(std::move(v));
    Comparator cmp;
    std::push_heap(_seq->begin(), _seq->end(), cmp);
  }

  template<typename... Args>
  void emplace(Args &&... args) {
    _seq->emplace_back(std::forward<Args>(args)...);
    Comparator cmp;
    std::push_heap(_seq->begin(), _seq->end(), cmp);
  }

  void pop() {
    Comparator cmp;
    std::pop_heap(_seq->begin(), _seq->end(), cmp);
    _seq->pop_back();
  }

  const T &top() const {
    assert(!_seq->empty());
    return _seq->front();
  }

  T &top() {
    assert(!_seq->empty());
    return _seq->front();
  }

  unsigned size() const {
    return _seq->size();
  }

  bool empty() const {
    return _seq->empty();
  }

  void clear() {
    _seq->clear();
  }

  void sort() {
    Comparator cmp;
    std::sort_heap(_seq->begin(), _seq->end(), cmp);
  }

private:
  Container *_seq{nullptr};
};

}

#endif //EFANNA2E_PRIORITYQUEUE_H
