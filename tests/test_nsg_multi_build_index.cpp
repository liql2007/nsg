//
// Created by liql2007 on 2020/12/23.
//

#include <cassert>
#include <memory>
#include <efanna2e/util.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/index_nsg.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/test_helper.h>

namespace {

Partitions partitionData(const char* dataPath, const char* kMeansCentroidsPath,
                         const char* multiIndexPath) {
  std::cout << "** Partition Vectors" << std::endl;
  float* vecData = nullptr;
  unsigned pointNum, dim;
  load_data(dataPath, vecData, pointNum, dim);
  std::unique_ptr<float[]> vecHolder(vecData);

  float* centroidsData = nullptr;
  unsigned partNum, centroidDim;
  load_data(kMeansCentroidsPath, centroidsData, partNum, centroidDim);
  std::unique_ptr<float[]> centroidHolder(centroidsData);
  assert(partNum > 1);
  assert(centroidDim == dim);

  Partitions parts = Partitions::create(multiIndexPath, partNum);
  parts.dim = dim;
  parts.totalVecNum = pointNum;

  std::vector<std::unique_ptr<std::ofstream>> docStreams;
  std::vector<std::unique_ptr<std::ofstream>> idStreams;
  docStreams.reserve(partNum);
  idStreams.reserve(partNum);
  for (const auto& part : parts.partInfos) {
    auto docStream = std::make_unique<std::ofstream>(
      part.docPath.c_str(), std::ios::binary | std::ios::out);
    docStreams.push_back(std::move(docStream));
    auto idStream = std::make_unique<std::ofstream>(
      part.idPath.c_str(), std::ios::binary | std::ios::out);
    idStreams.push_back(std::move(idStream));
  }

  auto addDocFun = [&](unsigned id, const float* vec, unsigned globalId) {
    docStreams[id]->write((const char*)&dim, sizeof(unsigned));
    docStreams[id]->write((const char*)vec, sizeof(float) * dim);
    idStreams[id]->write((const char*)&globalId, sizeof(unsigned));
    ++parts.partInfos[id].vecNum;
  };

  faiss::IndexFlatL2 classifyIndex(dim);
  classifyIndex.add(partNum, centroidsData);
  const constexpr unsigned ProjectClusterNum = 2;
  const constexpr unsigned BatchNum = 4096;
  faiss::Index::idx_t label[ProjectClusterNum * BatchNum];
  float dis[ProjectClusterNum * BatchNum];
  for (unsigned gid = 0; gid < pointNum; gid += BatchNum) {
    auto vecs = vecData + gid * dim;
    auto realBatchNum = std::min(BatchNum, pointNum - gid);
    classifyIndex.search(realBatchNum, vecs, ProjectClusterNum, &dis[0], &label[0]);
    for (unsigned vi = 0; vi < realBatchNum; ++vi) {
      auto vec = vecs + vi * dim;
      for (unsigned k = 0; k < ProjectClusterNum; ++k) {
        addDocFun(label[vi * ProjectClusterNum + k], vec, gid + vi);
      }
    }
  }
  for (unsigned i = 0; i < parts.partInfos.size(); ++i) {
    docStreams[i]->close();
    idStreams[i]->close();
  }
  return parts;
}

void buildKNN(const PartInfo& part, const char* knnBuildCmd) {
  char buildCmd[1024];
  sprintf(buildCmd, knnBuildCmd, part.docPath.c_str(), part.knnPath.c_str());
  std::cout << buildCmd << std::endl;
  if (system(buildCmd) != 0) {
    std::cerr << "build knn failed" << std::endl;
    exit(-1);
  }
}

void createPartGroundTruth(const PartInfo& part, const float* vecData,
                           unsigned pointNum, unsigned dim) {
  auto s = std::chrono::high_resolution_clock::now();
  efanna2e::DistanceL2 distance;
  const constexpr unsigned QueryNum = 200;
  const constexpr unsigned TopK = 100;
  std::mt19937 rng(time(nullptr));
  std::vector<unsigned> queryIds(QueryNum);
  efanna2e::GenRandom(rng, queryIds.data(), QueryNum, pointNum);
  std::vector<std::vector<unsigned>> topNeighbors(QueryNum);
  std::vector<float> qVecs(QueryNum * dim);
#pragma omp parallel for
  for (unsigned i = 0; i < QueryNum; ++i) {
    auto qId = queryIds[i];
    efanna2e::Neighbor nn(qId, 0, true);
    std::vector<efanna2e::Neighbor> neighborPool;
    neighborPool.reserve(TopK + 1);
    neighborPool.resize(TopK);
    neighborPool[0] = std::move(nn);
    unsigned poolSize = 1;
    auto q = vecData + qId * dim;
    std::memcpy(qVecs.data() + i * dim, q, dim * sizeof(float));
    for (unsigned vId = 0; vId < pointNum; ++vId) {
      if (vId == qId) {
        continue;
      }
      auto v = vecData + vId * dim;
      float dist = distance.compare(v, q, dim);
      efanna2e::Neighbor nn(vId, dist, true);
      efanna2e::InsertIntoPool(neighborPool.data(), poolSize, nn);
      if (poolSize < TopK) {
        ++poolSize;
      }
    }
    assert(poolSize == TopK);
    std::sort(neighborPool.begin(), neighborPool.end(),
              [](const efanna2e::Neighbor& l, const efanna2e::Neighbor& r) {
                return l.distance < r.distance; });
    auto& queryTopNeighbor = topNeighbors[i];
    queryTopNeighbor.reserve(TopK);
    for (const auto& nn : neighborPool) {
      queryTopNeighbor.push_back(nn.id);
    }
  }
  save_data(part.groundTruthPath.c_str(), topNeighbors);
  save_data(part.queryPath.c_str(), qVecs.data(), QueryNum, dim);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "ground truth time: " << diff.count() << "\n";

//  std::cout << "query:" << std::endl;
//  print_vector(qVecs.data(), dim);
//  for (unsigned i = 0; i < 3; ++i) {
//    auto id = topNeighbors[0][i];
//    std::cout << "neighbor " << i + 1 << ":" << id << std::endl;
//    print_vector(vecData + id * dim, dim);
//  }
}

}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " data_path kMeans_centroids_path multi_index_dir knn_build_cmd L R C"
              << std::endl;
    exit(-1);
  }
  auto data_path = argv[1];
  auto kMeans_centroids_path = argv[2];
  auto multi_index_dir = argv[3];
  auto knn_build_cmd = argv[4];
  unsigned L = (unsigned)atoi(argv[5]);
  unsigned R = (unsigned)atoi(argv[6]);
  unsigned C = (unsigned)atoi(argv[7]);
  std::cout << "L: " << L << std::endl;
  std::cout << "R: " << R << std::endl;
  std::cout << "C: " << C << std::endl;
  std::cout << "KNN build cmd: " << knn_build_cmd << std::endl;

#define OnlyBuildNSG 0

#if OnlyBuildNSG > 0
  Partitions parts;
  parts.deserialize(multi_index_dir);
#else
  auto parts = partitionData(data_path, kMeans_centroids_path, multi_index_dir);
  parts.serialize();
#endif

  double totalNsgBuildTime = 0;
  for (unsigned i = 0; i < parts.partInfos.size(); ++i) {
    std::cout << "** Partition: " << i + 1 << std::endl;
    const auto& part = parts.partInfos[i];

#if OnlyBuildNSG == 0
    std::cout << "Build KNN" << std::endl;
    buildKNN(part, knn_build_cmd);
#endif

    float* vecData = NULL;
    unsigned pointNum, dimI;
    load_data(part.docPath.c_str(), vecData, pointNum, dimI);
    assert(dimI == parts.dim);
    std::unique_ptr<float[]>holder(vecData);
    std::cout << "vector num: " << pointNum / 10000.0 << "W" << std::endl;

#if OnlyBuildNSG == 0
    std::cout << "Create Ground Truth" << std::endl;
    createPartGroundTruth(part, vecData, pointNum, dimI);
#endif

    std::cout << "Build NSG" << std::endl;
    efanna2e::IndexNSG index(parts.dim, pointNum, efanna2e::L2, nullptr);
    auto s = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<std::string>("nn_graph_path", part.knnPath);

    index.Build(pointNum, vecData, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "build time: " << diff.count() << "\n";
    totalNsgBuildTime += diff.count();
    index.Save(part.nsgPath.c_str());
  }
  std::cout << "Part(" << parts.partInfos.size() << ") NSG avg build time: "
            << totalNsgBuildTime / parts.partInfos.size() << std::endl;
  std::cout << "NSG total build time: " << totalNsgBuildTime << std::endl;
  return 0;
}