//
// Created by liql2007 on 2021/3/24.
//

#ifndef EFANNA2E_REVERSGRAPH_H
#define EFANNA2E_REVERSGRAPH_H

namespace compress {

using label_t = uint32_t;

class ReversGraph {
public:
  explicit ReversGraph(const std::vector<label_t> &inDegrees) {
    assert(!inDegrees.empty());
    _offset.resize(inDegrees.size());
    std::size_t nextOffset = 0;
    for (std::size_t id = 0; id < inDegrees.size(); ++id) {
      _offset[id] = nextOffset;
      nextOffset = nextOffset + 1 + inDegrees[id];
    }
    _data.resize(nextOffset);
  }

  void addEdge(label_t fromId, label_t toId) {
    auto index = _offset[toId];
    auto &num = _data[index];
    _data[index + 1 + num] = fromId;
    ++num;
  }

  std::tuple<const label_t *, const label_t *> getPreNodes(label_t id) const {
    auto index = _offset[id];
    auto num = _data[index];
    return std::make_tuple(_data.data() + index + 1,
                           _data.data() + index + 1 + num);
  }

  label_t size() const { return _offset.size(); }

private:
  std::vector <std::size_t> _offset;
  std::vector <label_t> _data;
};

}

#endif //EFANNA2E_REVERSGRAPH_H
