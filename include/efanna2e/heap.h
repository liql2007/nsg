//
// Created by liql2007 on 2021/1/21.
//

#ifndef EFANNA2E_HEAP_H
#define EFANNA2E_HEAP_H

#include <algorithm>

namespace efanna2e {

template<typename T, typename Comparator = std::less <T>>
class Heap {
public:
  Heap(unsigned topk, T *storage)
    : _storage(storage), _topk(topk) {
  }

  ~Heap() = default;

  Heap(const Heap &) = delete;

  Heap &operator=(const Heap &) = delete;

  Heap(Heap &&) = default;

  Heap &operator=(Heap &&rhs) = default;

  bool push(T v) {
    // keep the topk high priority items
    Comparator cmp;
    if (_len < _topk) {
      _storage[_len++] = std::move(v);
      std::push_heap(_storage, _storage + _len, cmp);
      return true;
    }
    if (!cmp(v, _storage[0])) {
      return false;
    }
//    std::pop_heap(_storage, _storage + _topk, cmp);
//    _storage[_topk - 1] = std::move(v);
//    std::push_heap(_storage, _storage + _topk, cmp);
    _storage[0] = std::move(v);
    siftDown();
    return true;
  }

  template<typename... Args>
  bool emplace(Args &&... args) {
    // keep the topk high priority items
    Comparator cmp;
    if (_len < _topk) {
      construct(&_storage[_len++], std::forward<Args>(args)...);
      std::push_heap(_storage, _storage + _len, cmp);
      return true;
    }

    T v{};
    construct(&v, std::forward<Args>(args)...);

    if (!cmp(v, _storage[0])) {
      return false;
    }

//    std::pop_heap(_storage, _storage + _topk, cmp);
//    _storage[_topk - 1] = std::move(v);
//    std::push_heap(_storage, _storage + _topk, cmp);
    _storage[0] = std::move(v);
    siftDown();
    return true;
  }

  void siftDown() {
    if (_len < 2) {
      return;
    }
    Comparator cmp;
    unsigned child = 0;
    unsigned start = 0;
    child = 2 * start + 1;
    if ((child + 1) < _len && cmp(_storage[child], _storage[child + 1])) {
      ++child;
    }
    if (cmp(_storage[child], _storage[start])) {
      return;
    }
    T top = std::move(_storage[0]);
    do {
      _storage[start] = std::move(_storage[child]);
      start = child;
      if ((_len - 2) / 2 < child) {
        break;
      }
      child = 2 * child + 1;
      if ((child + 1) < _len && cmp(_storage[child], _storage[child + 1])) {
        ++child;
      }
    } while(!cmp(_storage[child], top));
    _storage[start] = std::move(top);
  }

  void siftDown2() {
    if (_len < 2) {
      return;
    }
    Comparator cmp;
    unsigned pos = 0;
    unsigned child = 0;
    T val = _storage[0];
    while (pos * 2 + 1 < _len) {
      child = 2 * pos + 1;
      if ((child + 1) < _len && cmp(_storage[child], _storage[child + 1])) {
        ++child;
      }
      if (cmp(_storage[child], val)) {
        break;
      }
      _storage[pos] = std::move(_storage[child]);
      pos = child;
    }
    if (pos > 0) {
      _storage[pos] = val;
    }
  }

  bool hint(const T &v) const {
    Comparator cmp;
    return _len < _topk || cmp(v, _storage[0]);
  }

  void pop() {
    assert(!empty());
    Comparator cmp;
    std::pop_heap(_storage, _storage + _len--, cmp);
  }

  // sort items by priority
  // if size of elements is less than topk
  // all elements of [_len.._topk-1] are undefined
  void sort() {
    assert(_len <= _topk);
    Comparator cmp;
    std::sort_heap(_storage, _storage + _len, cmp);
  }

  const T &root() const {
    assert(!empty());
    return _storage[0];
  }

  unsigned size() const { return _len; }

  unsigned capacity() const { return _topk; }

  bool empty() const { return _len == 0; }

  void clear() { _len = 0; }

  void shift(unsigned len) {
    _len = len;
  }

  const T *data() const { return _storage; }

  T *data() { return _storage; }

private:
  template<typename... Args>
  void construct(void *p, Args &&... args) const {
    ::new(p) T(std::forward<Args>(args)...);
  }

private:
  T *_storage{nullptr};
  unsigned _len{0};
  unsigned _topk{0};
};

}

#endif //EFANNA2E_HEAP_H
