//
// Created by liql2007 on 2020/12/23.
//

#include <cassert>
#include <memory>
#include <efanna2e/index_nsg.h>
#include <faiss/Clustering.h>
#include <efanna2e/test_helper.h>

namespace {

void loadDocIds(const PartInfo& part, std::vector<unsigned>& docIds) {
  std::ifstream in(part.idPath.c_str(), std::ios::binary);
  docIds.resize(part.vecNum);
  in.read((char*)docIds.data(), docIds.size() * sizeof(unsigned));
  if (in.bad()) {
    std::cerr << "read doc id failed" << std::endl;
    in.close();
    exit(-1);
  }
  in.close();
}

void mergeNsgEdge(const PartInfo& part, const efanna2e::IndexNSG& partIndex,
                  efanna2e::IndexNSG& index) {
  std::vector<unsigned> docIds;
  loadDocIds(part, docIds);
  auto& graph = index.graph();
  const auto& partGraph = partIndex.graph();
  auto dim = index.GetDimension();
  auto isExist = [](const std::vector<unsigned> vec, unsigned v) {
    for (auto vi : vec) {
      if (vi == v) {
        return true;
      }
    }
    return false;
  };
#pragma omp parallel for
  for (size_t i = 0; i < partGraph.size(); ++i) {
    auto gid = docIds[i];
    auto& neighbors = graph[gid];
    if (neighbors.empty()) { // first add
      auto partVecPtr = partIndex.getData() + i * dim;
      auto vecPtr = const_cast<float*>(index.getData() + gid * dim);
      std::memcpy(vecPtr, partVecPtr, dim * sizeof(float));
    }
    neighbors.reserve(neighbors.size() + partGraph[i].size());
    for (auto partVecId : partGraph[i]) {
      auto partGid = docIds[partVecId];
      if (!isExist(neighbors, partGid)) {
        neighbors.push_back(partGid);
      }
    }
  }
  auto& eps = index.getEps();
  auto& partEps = partIndex.getEps();
  for (auto partDocId : partEps) {
    auto partGid = docIds[partDocId];
    if (!isExist(eps, partGid)) {
      eps.push_back(partGid);
    }
  }
}

void pruneEdge(unsigned R, efanna2e::IndexNSG& index) {
  std::cout << "prune edge" << std::endl;
  efanna2e::DistanceL2 distance;
  auto& graph = index.graph();
  auto data = index.getData();
  auto dim = index.GetDimension();
#pragma omp parallel for
  for (size_t id = 0; id < graph.size(); ++id) {
    auto& neighbors = graph[id];
    if (neighbors.size() < R) {
      continue;
    }
    auto vec = data + id * dim;
    std::vector<efanna2e::Neighbor> pool(neighbors.size());
    for (unsigned i = 0; i < neighbors.size(); ++i) {
      auto nid = neighbors[i];
      assert(nid != id);
      auto nVec = data + nid * dim;
      pool[i].id = nid;
      pool[i].distance = distance.compare(vec, nVec, dim);
    }
    std::sort(pool.begin(), pool.end());
    unsigned resultSize = 1;
    unsigned checkIndex = 1;
    while(++checkIndex < pool.size() && resultSize < R) {
      auto &p = pool[checkIndex];
      bool occlude = false;
      for (unsigned t = 0; t < resultSize; t++) {
        if (p.id == pool[t].id) {
          occlude = true;
          break;
        }
        float djk = distance.compare(data + dim * (size_t)pool[t].id,
                                      data + dim * (size_t)p.id,
                                       (unsigned)dim);
#ifdef SSG
        float cos_ij = (p.distance + pool[t].distance - djk) / 2 /
                     sqrt(p.distance * pool[t].distance);
        if (cos_ij > efanna2e::cosinThreshold) {
          occlude = true;
          break;
        }
#else
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
#endif
      }
      if (!occlude) pool[resultSize++] = p;
    }
    neighbors.resize(resultSize);
    for (unsigned i = 0; i < resultSize; ++i) {
      neighbors[i] = pool[i].id;
    }
  }
}

void statisticAndSet(efanna2e::IndexNSG& index) {
  unsigned max = 0, min = 1e6;
  double avg = 0;
  auto& graph = index.graph();
  for (size_t i = 0; i < graph.size(); i++) {
    auto size = graph[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  index.setWidth(max);
  avg /= index.getVecNum();
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %.3lf\n", max, min, avg);
}

}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " multi_index_dir R" << std::endl;
    exit(-1);
  }
  auto multi_index_path = argv[1];
  unsigned R = (unsigned)atoi(argv[2]);

  Partitions parts;
  parts.deserialize(multi_index_path);

  efanna2e::IndexNSG index(parts.dim, parts.totalVecNum, efanna2e::L2, nullptr);
  float* vecData = new float[parts.totalVecNum * parts.dim];
  index.graph().resize(parts.totalVecNum);
  index.setData(vecData);

  auto bb = std::chrono::high_resolution_clock::now();
  auto partCount = parts.partInfos.size();
  double avgDegree = 0;
  for (unsigned i = 0; i < partCount; ++i) {
    std::cout << "** Merge NSG: " << i + 1 << std::endl;
    const auto& part = parts.partInfos[i];

    float* partVecData = NULL;
    unsigned pointNum, dim;
    load_data(part.docPath.c_str(), partVecData, pointNum, dim);
    assert(pointNum == part.vecNum);
    assert(dim == parts.dim);
    std::unique_ptr<float[]>holder(partVecData);
    efanna2e::IndexNSG partIndex(dim, pointNum, efanna2e::L2, nullptr);
    partIndex.Load(part.nsgPath.c_str());
    partIndex.setData(partVecData);
    avgDegree += partIndex.getAvgDegree();

    mergeNsgEdge(part, partIndex, index);
  }
  std::cout << "part index avg degree: " << avgDegree / partCount << "\n";
  statisticAndSet(index);
  pruneEdge(R, index);
//  efanna2e::Parameters paras;
//  paras.Set<unsigned>("L", R);
//  index.tree_grow(paras);
  statisticAndSet(index);

  auto ee = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = ee - bb;
  std::cout << "merge time: " << diff.count() << "\n";

  index.Save(parts.getMergedNsgPath().c_str());
  // save_data(parts.getMergedVecPath().c_str(), vecData, parts.totalVecNum, parts.dim);

  return 0;
}
