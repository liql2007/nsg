//
// Created by liql2007 on 2020/12/23.
//

#include <cassert>
#include <memory>
#include <efanna2e/index_nsg.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/test_helper.h>

namespace {

void buildKNN(const PartInfo& part, const char* knnBuildCmd) {
  char buildCmd[1024];
  sprintf(buildCmd, knnBuildCmd, part.docPath.c_str(), part.knnPath.c_str());
  std::cout << buildCmd << std::endl;
  if (system(buildCmd) != 0) {
    std::cerr << "build knn failed" << std::endl;
    exit(-1);
  }
}

}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0]
              << " multi_index_dir knn_build_cmd L R C"
              << std::endl;
    exit(-1);
  }
  auto multi_index_dir = argv[1];
  auto knn_build_cmd = argv[2];
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);
  std::cout << "L: " << L << std::endl;
  std::cout << "R: " << R << std::endl;
  std::cout << "C: " << C << std::endl;
  std::cout << "KNN build cmd: " << knn_build_cmd << std::endl;

#define OnlyBuildNSG 0

  Partitions parts;
  parts.deserialize(multi_index_dir);

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