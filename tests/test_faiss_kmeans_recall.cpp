//
// Created by liql2007 on 2020/12/23.
//

#include <cassert>
#include <cstring>
#include <faiss/IndexFlat.h>
#include <efanna2e/test_helper.h>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << argv[0]
              << " data_path query_path ground_truth_path kmeans_centroids_path"
              << std::endl;
    exit(-1);
  }

  auto data_path = argv[1];
  auto query_path = argv[2];
  auto ground_truth_path = argv[3];
  auto kmeans_centroids_path = argv[4];

  float* vec_data = nullptr;
  unsigned points_num, dim;
  load_data(data_path, vec_data, points_num, dim);

  float* query_data = nullptr;
  unsigned query_num, query_dim;
  load_data(query_path, query_data, query_num, query_dim);
  assert(dim == query_dim);

  GroundTruth truth(10);
  truth.load(ground_truth_path);
  assert(truth.queryNum == query_num);

  float* centroids_data = nullptr;
  unsigned centroid_num, centroid_dim;
  load_data(kmeans_centroids_path, centroids_data, centroid_num, centroid_dim);
  assert(dim == centroid_dim);
  assert(centroid_num > 1);

  faiss::IndexFlatL2 index (dim);
  index.add(centroid_num, centroids_data);
  float sumRecallRate = 0;
  for (unsigned i = 0; i < query_num; ++i) {
    float* q = query_data + i * dim;
    faiss::Index::idx_t label[2];
    float dis[2];
    index.search(1, q, 2, &dis[0], &label[0]);

    std::vector<float> vecVec(truth.truthItemNum * dim);
    unsigned* ts = truth.data + i * truth.truthItemNum;
    for (unsigned ti = 0; ti < truth.truthItemNum; ++ti) {
      auto vec = vec_data + ts[ti] * dim;
      std::memcpy(vecVec.data() + ti * dim, vec, dim * sizeof(float));
    }
    std::vector<faiss::Index::idx_t> labelVec(truth.truthItemNum);
    std::vector<float> disVec(truth.truthItemNum);
    index.search(truth.truthItemNum, vecVec.data(), 1, disVec.data(), labelVec.data());
    unsigned matchNum = 0;
    for (unsigned ti = 0; ti < truth.truthItemNum; ++ti) {
      // if (labelVec[ti] == label[0] || labelVec[ti] == label[1]) {
      if (labelVec[ti] == label[0]) {
        ++matchNum;
      }
    }
    auto recallRate = (float)matchNum / truth.truthItemNum;
    sumRecallRate += recallRate;
  }
  std::cout << "AVG Recall Rate: " << sumRecallRate / query_num << std::endl;
  return 0;
}