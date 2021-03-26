#include <compress/reordering/Reordering.h>
#include <cstring>

namespace compress {

bool Reordering::genReorderMapping(
  const std::vector<std::vector<label_t>>& graph,
  std::vector<label_t>* mapping) {
  if (graph.empty()) {
    return true;
  }
  auto& inDegrees = *mapping;
  inDegrees.resize(graph.size(), 0);
  for (label_t fromId = 0; fromId < graph.size(); ++fromId) {
    for (auto toId : graph[fromId]) {
      ++inDegrees[toId];
    }
  }
  ReversGraph rGraph(inDegrees);
  for (label_t fromId = 0; fromId < graph.size(); ++fromId) {
    for (auto toId : graph[fromId]) {
      rGraph.addEdge(fromId, toId);
    }
  }

  mapping->clear();
  RecursiveGraphBisection bp;
  if (!bp.genNewOrderMapping(rGraph, mapping)) {
    printf("genNewOrderMapping failed");
    return false;
  }
  return true;
}

void Reordering::reorder(std::size_t vecSize,
                         const std::vector<label_t>& mapping,
                         std::vector<std::vector<label_t>>* graph,
                         std::vector<label_t>* eps,
                         uint8_t* vecs) {
  for (label_t fromId = 0; fromId < graph->size(); ++fromId) {
    for (auto& toId : graph->at(fromId)) {
      toId = mapping[toId];
    }
  }
  for (auto& id : *eps) {
    id = mapping[id];
  }

  std::vector<label_t> cyclePath;
  std::vector<uint8_t> vecBuff(vecSize);
  std::vector<bool> handled(graph->size(), false);
  for (label_t id = 0; id < graph->size(); ++id) {
    if (handled[id]) {
      continue;
    }
    handled[id] = true;
    auto newId = mapping[id];
    if (newId == id) {
      continue;
    }
    cyclePath.clear();
    cyclePath.push_back(id);
    while(newId != id) {
      cyclePath.push_back(newId);
      handled[newId] = true;
      newId = mapping[newId];
    }

    auto lastId = cyclePath.back();
    auto lastNeighbors = std::move(graph->at(lastId));
    std::memcpy(vecBuff.data(), vecs + lastId * vecSize, vecSize);
    for (std::size_t i = cyclePath.size() - 1; i > 0; --i) {
      auto dstId = cyclePath[i];
      auto srcId = cyclePath[i - 1];
      auto& dstNN = graph->at(dstId);
      auto& srcNN = graph->at(srcId);
      dstNN = std::move(srcNN);
      std::memcpy(vecs + dstId * vecSize, vecs + srcId * vecSize, vecSize);
    }
    graph->at(id) = std::move(lastNeighbors);
    std::memcpy(vecs + id * vecSize,  vecBuff.data(), vecSize);
  }
}

}
