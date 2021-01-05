//
// Created by liql2007 on 2021/1/4.
//

#include <efanna2e/test_helper.h>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0]
              << " data_path sample_data_path sample_num"
              << std::endl;
    exit(-1);
  }
  auto dataPath = argv[1];
  auto sampleDataPath = argv[2];
  auto sampleNum = (size_t)atoi(argv[3]);

  float* vecData = nullptr;
  unsigned pointNum, dim;
  load_data(dataPath, vecData, pointNum, dim);

  std::mt19937 rng(time(nullptr));
  std::vector<unsigned> vecIds(sampleNum);
  efanna2e::GenRandom(rng, vecIds.data(), sampleNum, pointNum);
  std::vector<float> sampleData(sampleNum * dim);
  for (size_t i = 0; i < sampleNum; ++i) {
    auto vecId = vecIds[i];
    auto vec = vecData + vecId * dim;
    memcpy(&sampleData[i * dim], vec, dim * sizeof(float));
  }
  save_data(sampleDataPath, sampleData.data(), sampleNum, dim);
}