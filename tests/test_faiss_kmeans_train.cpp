//
// Created by liql2007 on 2020/12/23.
//

#include <limits>
#include <time.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/test_helper.h>

void save_kmeans_centroids(const char* filename, unsigned dim,
                           const std::vector<float>& centroids) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  if (centroids.empty() || centroids.size() % dim != 0) {
    std::cerr << "centroids error" << std::endl;
    exit(-1);
  }
  for (unsigned i = 0; i < centroids.size(); i += dim) {
    out.write((char*)&dim, sizeof(unsigned));
    out.write((char*)(&centroids[i]), dim * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << argv[0]
              << " learn_data_file centroids_result_file K iterNum"
              << std::endl;
    exit(-1);
  }

  auto learn_data_path = argv[1];
  auto centroids_result_path = argv[2];
  unsigned k = (unsigned)atoi(argv[3]);
  unsigned iterNum = (unsigned)atoi(argv[4]);
  std::cout << "train data path: " << learn_data_path << std::endl;
  std::cout << "centroids result path: " << centroids_result_path << std::endl;
  std::cout << "K: " << k << ",  IterNum: " << iterNum << std::endl;

  float* learn_data = NULL;
  unsigned points_num, dim;
  load_data(argv[1], learn_data, points_num, dim);
  std::cout << "train vector num: " << points_num << std::endl;

  faiss::ClusteringParameters cp;
  cp.max_points_per_centroid = std::numeric_limits<int>::max();
  cp.niter = iterNum;
  cp.verbose = true;
  cp.seed = 123456;
  cp.nredo = 1;

  faiss::Clustering clus (dim, k, cp);
  faiss::IndexFlatL2 index (dim);
  clus.train(points_num, learn_data, index);
  save_kmeans_centroids(centroids_result_path, dim, clus.centroids);
  return 0;
}