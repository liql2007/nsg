//
// Created by liql2007 on 2021/1/4.
//

#include <efanna2e/test_helper.h>

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0]
              << " data_path query_path ground_truth_path queryNum topK"
              << std::endl;
    exit(-1);
  }
  auto dataPath = argv[1];
  auto queryPath = argv[2];
  auto groundTruthPath = argv[3];
  auto queryNum = (unsigned)atoi(argv[4]);
  auto topK = (unsigned)atoi(argv[5]);

  float* vecData = nullptr;
  unsigned pointNum, dim;
  load_data(dataPath, vecData, pointNum, dim);

  GroundTruth::createPartGroundTruth(
    queryPath, groundTruthPath, vecData, pointNum, dim, queryNum, topK);
}