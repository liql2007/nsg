#ifndef EFANNA2E_RECURSIVEGRAPHBISECTION_H
#define EFANNA2E_RECURSIVEGRAPHBISECTION_H

#include <vector>
#include <cassert>
#include <compress/reordering/ThreadPool.h>
#include <compress/reordering/ReversGraph.h>
#include <compress/reordering/GenerationVector.h>

namespace compress {

class RecursiveGraphBisection {
public:
  bool genNewOrderMapping(const ReversGraph& rGraph,
                          std::vector<label_t>* mapping);

private:
  struct Range {
    label_t begin{0};
    label_t end{0};

    Range(label_t start, label_t end) : begin(start), end(end) {}

    label_t size() const { return end - begin; }

    std::pair<Range, Range> split() const {
      auto mid = begin + size() / 2;
      return {{begin, mid}, {mid,   end}};
    }
  };

  struct Context;

  using DegreeVector = GenerationVector<uint32_t, std::size_t>;
  using GainsVector = GenerationVector<float, std::size_t>;

  void graphBisection(Range range, uint8_t depth);

  void reorderPart(const Range &leftRange,
                   const Range &rightRange,
                   uint32_t iterCount, Context *ctx);
  void reorderPart2(const Range& leftRange, const Range& rightRange,
                    uint32_t threadNum, uint32_t iterCount, Context* ctx);

  void computeDegrees(const Range& range, DegreeVector* degrees) const;

  void computeMoveGains(const Range& fromRange, const Range& toRange,
                        const DegreeVector& fromDegree,
                        const DegreeVector& toDegree,
                        GainsVector* preNodeGains);
  void computeMoveGains2(const Range& fromRange, const Range& toRange,
                        const DegreeVector& fromDegree,
                        const DegreeVector& toDegree,
                        GainsVector* preNodeGains,
                         std::mutex* locks, uint32_t threadNum);

  bool swapLocation(const Range& lefRange, const Range& rightRange,
                    DegreeVector* leftDegree, DegreeVector* rightDegree);

  bool swapLocation2(const Range& lefRange, const Range& rightRange,
                     DegreeVector* leftDegree, DegreeVector* rightDegree,
                     std::mutex* locks, uint32_t threadNum);

private:
  const ReversGraph* _rGraph { nullptr };
  std::vector<label_t> _ids;
  std::vector<float> _gains;
  std::uint8_t _maxDepth { 0 };
  int32_t _threadFullDepth {0 };
  ThreadPool _pool;
};

}

#endif //EFANNA2E_RECURSIVEGRAPHBISECTION_H
