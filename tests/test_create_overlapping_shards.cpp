//
// Created by liql2007 on 2020/12/23.
//

#include <cassert>
#include <memory>
#include <efanna2e/index_nsg.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/test_helper.h>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0]
              << " data_path kMeans_centroids_path multi_index_dir"
              << std::endl;
    exit(-1);
  }
  auto dataPath = argv[1];
  auto kMeansCentroidsPath = argv[2];
  auto multiIndexPath = argv[3];

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
  const constexpr size_t ProjectClusterNum = 2;
  const constexpr size_t BatchNum = 4096;
  faiss::Index::idx_t label[ProjectClusterNum * BatchNum];
  float dis[ProjectClusterNum * BatchNum];
  for (size_t gid = 0; gid < pointNum; gid += BatchNum) {
    auto vecs = vecData + gid * dim;
    auto realBatchNum = std::min(BatchNum, pointNum - gid);
    classifyIndex.search(realBatchNum, vecs, ProjectClusterNum, &dis[0], &label[0]);
    for (size_t vi = 0; vi < realBatchNum; ++vi) {
      auto vec = vecs + vi * dim;
      for (unsigned k = 0; k < ProjectClusterNum; ++k) {
        addDocFun(label[vi * ProjectClusterNum + k], vec, gid + vi);
      }
    }
  }
  parts.serialize();

  for (unsigned i = 0; i < parts.partInfos.size(); ++i) {
    docStreams[i]->close();
    idStreams[i]->close();
    std::cout << "** Partition: " << i + 1 << std::endl;
    const auto &part = parts.partInfos[i];

    float* vecData = NULL;
    unsigned pointNum, dimI;
    load_data(part.docPath.c_str(), vecData, pointNum, dimI);
    assert(dimI == parts.dim);
    std::unique_ptr<float[]>holder(vecData);
    std::cout << "vector num: " << pointNum / 10000.0 << "W" << std::endl;
    std::cout << "Create Ground Truth" << std::endl;
    GroundTruth::createPartGroundTruth(
      part.queryPath.c_str(), part.groundTruthPath.c_str(),
      vecData, pointNum, dimI, 100, 100);
  }

  return 0;
}