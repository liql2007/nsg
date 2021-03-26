#include <compress/reordering/RecursiveGraphBisection.h>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <parallel/algorithm>
#include <omp.h>
#ifdef __SSE__
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#endif

namespace compress {

namespace {

template <std::size_t N>
class Log2 {
  static_assert(N >= 0, "precompute count must >= 0");
public:
  Log2() {
    for (std::size_t n = 0; n < N; ++n) {
      _values[n] = std::log2f(n);
    }
  }

  float operator()(std::uint32_t n) const {
    if (n >= N) {
      return std::log2f(n);
//      int r;
//      __asm__("bsrl %1,%0" : "=&r" (r) : "rm" (n));
//      return r;
    }
    return _values[n];
  }

private:
  float _values[N]{};
};

static const Log2<4096> log2;

inline float termCost(float logn1, float logn2, uint32_t deg1, uint32_t deg2) {
#if 0
  return (logn1 - log2(deg1 + 1)) * deg1 + (logn2 - log2(deg2 + 1)) * deg2;
#else
  __m128 _deg = _mm_cvtepi32_ps(_mm_set_epi32(deg1, deg1, deg2, deg2));
  __m128 _log = _mm_set_ps(logn1, log2(deg1 + 1), logn2, log2(deg2 + 1));
  _log = _mm_mul_ps(_deg, _log);
  float vals[4];
  _mm_store_ps(vals, _log);
  return vals[3] - vals[2] + vals[1] - vals[0];
#endif
}

}

struct RecursiveGraphBisection::Context {
  DegreeVector leftDegrees;
  DegreeVector rightDegrees;
  GainsVector preNodeGains;
  std::mutex _locks[256];
};

bool RecursiveGraphBisection::genNewOrderMapping(
  const ReversGraph& rGraph, std::vector<label_t>* mapping) {
  auto nodeCount = rGraph.size();

  _ids.resize(nodeCount);
  for (std::size_t i = 0; i < _ids.size(); ++i) {
    _ids[i] = i;
  }
  if (nodeCount <= 2) {
    mapping->swap(_ids);
    return true;
  }

  _gains.resize(nodeCount);
  _rGraph = &rGraph;
  if (!_pool.start(0, 512)) {
    return false;
  }

  _maxDepth = static_cast<uint8_t>(log2(nodeCount));
  _maxDepth = _maxDepth > 10 ? _maxDepth - 5 : _maxDepth;
  _threadFullDepth = log2(std::thread::hardware_concurrency());
  printf("_threadFullDepth: %d\n", _threadFullDepth);

  Range range(0, nodeCount);
  _pool.push([&] { graphBisection(range, 1); });
  _pool.waitAllFinished();
  _pool.stop();
  _pool.waitStopped();

  mapping->resize(nodeCount);
  for (std::size_t i = 0; i < _ids.size(); ++i) {
    (*mapping)[_ids[i]] = i;
  }
  return true;
}

void RecursiveGraphBisection::graphBisection(Range range, uint8_t depth) {
  thread_local Context context;
  if (UNLIKELY(range.begin == 0)) {
    printf("layer [%u]\n", depth);
  }
  auto partition = range.split();
  auto& leftRange = partition.first;
  auto& rightRange = partition.second;
  const constexpr auto iterCount = 10;
  if (depth <= _threadFullDepth) {
    uint32_t threadNum = (1 << (1 + _threadFullDepth - depth)) + 1;
    __gnu_parallel::sort(_ids.data() + range.begin, _ids.data() + range.end,
                         __gnu_parallel::default_parallel_tag(threadNum));
    reorderPart2(leftRange, rightRange, threadNum, iterCount, &context);
  } else {
    std::sort(_ids.begin() + range.begin, _ids.begin() + range.end);
    reorderPart(leftRange, rightRange, iterCount, &context);
  }

  if (depth < _maxDepth && range.size() > 2) {
    auto task1 = [=] { graphBisection(leftRange, depth + 1); };
    _pool.push(task1, true);
    auto task2 = [=] { graphBisection(rightRange, depth + 1); };
    _pool.push(task2, true);
  } else {
    std::sort(_ids.begin() + leftRange.begin,
              _ids.begin() + leftRange.end);
    std::sort(_ids.begin() + rightRange.begin,
              _ids.begin() + rightRange.end);
  }
}

void RecursiveGraphBisection::reorderPart(const Range &leftRange,
                                          const Range &rightRange,
                                          uint32_t iterCount, Context *ctx) {
  ctx->leftDegrees.clearOrInit(_rGraph->size());
  ctx->rightDegrees.clearOrInit(_rGraph->size());

  computeDegrees(leftRange, &ctx->leftDegrees);
  computeDegrees(rightRange, &ctx->rightDegrees);

  static auto gainsCmp = [this](label_t lhs, label_t rhs) {
    return _gains[lhs] > _gains[rhs];
  };

  for (decltype(iterCount) iteration = 0; iteration < iterCount; ++iteration) {
    computeMoveGains(leftRange, rightRange, ctx->leftDegrees,
                     ctx->rightDegrees, &ctx->preNodeGains);
    computeMoveGains(rightRange, leftRange, ctx->rightDegrees,
                     ctx->leftDegrees, &ctx->preNodeGains);
    std::sort(_ids.data() + leftRange.begin, _ids.data() + leftRange.end,
              gainsCmp);
    std::sort(_ids.data() + rightRange.begin, _ids.data() + rightRange.end,
              gainsCmp);
    auto swapped = swapLocation(leftRange, rightRange,
                                &ctx->leftDegrees, &ctx->rightDegrees);
    if (UNLIKELY(!swapped)) {
      break;
    }
  }
}

void RecursiveGraphBisection::reorderPart2(
  const Range& leftRange, const Range& rightRange,
  uint32_t threadNum, uint32_t iterCount, Context* ctx) {
  ctx->leftDegrees.clearOrInit(_rGraph->size());
  ctx->rightDegrees.clearOrInit(_rGraph->size());

  computeDegrees(leftRange, &ctx->leftDegrees);
  computeDegrees(rightRange, &ctx->rightDegrees);

  static auto gainsCmp = [this](label_t lhs, label_t rhs) {
    return _gains[lhs] > _gains[rhs];
  };

  for (decltype(iterCount) iteration = 0; iteration < iterCount; ++iteration) {
    computeMoveGains2(leftRange, rightRange, ctx->leftDegrees,
                     ctx->rightDegrees, &ctx->preNodeGains, ctx->_locks, threadNum);
    computeMoveGains2(rightRange, leftRange, ctx->rightDegrees,
                     ctx->leftDegrees, &ctx->preNodeGains, ctx->_locks, threadNum);
    __gnu_parallel::sort(_ids.data() + leftRange.begin, _ids.data() + leftRange.end,
                         gainsCmp, __gnu_parallel::default_parallel_tag(threadNum));
    __gnu_parallel::sort(_ids.data() + rightRange.begin, _ids.data() + rightRange.end,
                         gainsCmp, __gnu_parallel::default_parallel_tag(threadNum));
//    auto swapped = swapLocation2(leftRange, rightRange, &ctx->leftDegrees,
//                                 &ctx->rightDegrees, ctx->_locks, threadNum);
    auto swapped = swapLocation(leftRange, rightRange,
                                &ctx->leftDegrees, &ctx->rightDegrees);
    if (UNLIKELY(!swapped)) {
      break;
    }
  }
}

void RecursiveGraphBisection::computeDegrees(const Range& range,
                                             DegreeVector* degrees) const {
  for (auto index = range.begin; index != range.end; ++index) {
    auto id = _ids[index];
    const label_t* iter, *end;
    std::tie(iter, end) = _rGraph->getPreNodes(id);
    while (iter != end) {
      auto fromId = *iter++;
      degrees->setValue(fromId, (*degrees)[fromId] + 1);
    }
  }
}

void RecursiveGraphBisection::computeMoveGains(
  const Range& fromRange, const Range& toRange, const DegreeVector& fromDegree,
  const DegreeVector& toDegree, GainsVector* preNodeGains) {
  preNodeGains->clearOrInit(fromDegree.size());

  const auto logn1 = log2(fromRange.size());
  const auto logn2 = log2(toRange.size());

  for (auto index = fromRange.begin; index != fromRange.end; ++index) {
    auto id = _ids[index];
    float gain = 0.0;
    const label_t* iter, *end;
    std::tie(iter, end) = _rGraph->getPreNodes(id);
    while (iter != end) {
      auto preNodeId = *iter++;
      float termGain = 0;
      if (LIKELY(!preNodeGains->getValue(preNodeId, &termGain))) {
        auto fromDeg = fromDegree[preNodeId];
        auto toDeg = toDegree[preNodeId];
        if (fromDeg != toDeg + 1) {
          termGain = termCost(logn1, logn2, fromDeg, toDeg) -
                     termCost(logn1, logn2, fromDeg - 1, toDeg + 1);
        }
        preNodeGains->setValue(preNodeId, termGain);
      }
      gain += termGain;
    }
    _gains[id] = gain;
  }
}

void RecursiveGraphBisection::computeMoveGains2(
  const Range& fromRange, const Range& toRange, const DegreeVector& fromDegree,
  const DegreeVector& toDegree, GainsVector* preNodeGains,
  std::mutex* locks, uint32_t threadNum) {
  preNodeGains->clearOrInit(fromDegree.size());

  const auto logn1 = log2(fromRange.size());
  const auto logn2 = log2(toRange.size());

#pragma omp parallel for num_threads(threadNum)
  for (auto index = fromRange.begin; index < fromRange.end; ++index) {
    auto id = _ids[index];
    float gain = 0.0;
    const label_t* iter, *end;
    std::tie(iter, end) = _rGraph->getPreNodes(id);
    while (iter != end) {
      auto preNodeId = *iter++;
      float termGain = 0;
      if (!preNodeGains->getValue(preNodeId, &termGain)) {
        auto fromDeg = fromDegree[preNodeId];
        auto toDeg = toDegree[preNodeId];
        if (fromDeg != toDeg + 1) {
          termGain = termCost(logn1, logn2, fromDeg, toDeg) -
                     termCost(logn1, logn2, fromDeg - 1, toDeg + 1);
        }
//        auto& lock = locks[preNodeId & 0xFF];
//        lock.lock();
        preNodeGains->setValue(preNodeId, termGain);
//        lock.unlock();
      }
      gain += termGain;
    }
    _gains[id] = gain;
  }
}

bool RecursiveGraphBisection::swapLocation(
  const Range& lefRange, const Range& rightRange,
  DegreeVector* leftDegree, DegreeVector* rightDegree) {
  auto end = std::min(lefRange.size(), rightRange.size());
  decltype(end) i = 0;
  for (; i < end; ++i) {
    auto& lId = _ids[lefRange.begin + i];
    auto& rId = _ids[rightRange.begin + i];
    if (UNLIKELY(_gains[lId] + _gains[rId] <= 0)) {
      break;
    }
    const label_t* iter, *end;
    std::tie(iter, end) = _rGraph->getPreNodes(lId);
    while (iter != end) {
      auto preNodeId = *iter++;
      leftDegree->setValue(preNodeId, (*leftDegree)[preNodeId] - 1);
      rightDegree->setValue(preNodeId, (*rightDegree)[preNodeId] + 1);
    }

    std::tie(iter, end) = _rGraph->getPreNodes(rId);
    while (iter != end) {
      auto preNodeId = *iter++;
      leftDegree->setValue(preNodeId, (*leftDegree)[preNodeId] + 1);
      rightDegree->setValue(preNodeId, (*rightDegree)[preNodeId] - 1);
    }
    std::swap(lId, rId);
  }
  return i != 0;
}

bool RecursiveGraphBisection::swapLocation2(
  const Range& lefRange, const Range& rightRange,
  DegreeVector* leftDegree, DegreeVector* rightDegree,
  std::mutex* locks, uint32_t threadNum) {
  auto end = std::min(lefRange.size(), rightRange.size());
  volatile bool changed = false;
  volatile auto thresholdId = end;
#pragma omp parallel for num_threads(threadNum)
  for (decltype(end) i = 0; i < end; ++i) {
    if (i > thresholdId) {
      continue;
    }
    auto& lId = _ids[lefRange.begin + i];
    auto& rId = _ids[rightRange.begin + i];
    if (UNLIKELY(_gains[lId] + _gains[rId] <= 0)) {
      thresholdId = i;
      continue;
    }
    changed = true;
    const label_t* iter, *end;
    std::tie(iter, end) = _rGraph->getPreNodes(lId);
    while (iter != end) {
      auto preNodeId = *iter++;
      auto& lock = locks[preNodeId & 0xFF];
      lock.lock();
      leftDegree->setValue(preNodeId, (*leftDegree)[preNodeId] - 1);
      rightDegree->setValue(preNodeId, (*rightDegree)[preNodeId] + 1);
      lock.unlock();
    }

    std::tie(iter, end) = _rGraph->getPreNodes(rId);
    while (iter != end) {
      auto preNodeId = *iter++;
      auto& lock = locks[preNodeId & 0xFF];
      lock.lock();
      leftDegree->setValue(preNodeId, (*leftDegree)[preNodeId] + 1);
      rightDegree->setValue(preNodeId, (*rightDegree)[preNodeId] - 1);
      lock.unlock();
    }
    std::swap(lId, rId);
  }
  return changed;
}

}