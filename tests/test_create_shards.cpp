//
// Created by liql2007 on 2021/1/4.
//

#include <efanna2e/test_helper.h>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0]
              << " data_path shard_dir shard_num"
              << std::endl;
    exit(-1);
  }
  auto dataPath = argv[1];
  auto shardDirPath = argv[2];
  auto shardNum = (unsigned)atoi(argv[3]);

  float* vecData = nullptr;
  unsigned pointNum, dim;
  load_data(dataPath, vecData, pointNum, dim);

  auto minSize = pointNum / shardNum;
  auto leftNum = pointNum % shardNum;
  for (unsigned i = 0; i < shardNum; ++i) {
    auto path = std::string(shardDirPath) + "/shard" + std::to_string(i + 1) + ".fvecs";
    auto size = minSize + (i < leftNum ? 1 : 0);
    save_data(path.c_str(), vecData, size, dim);
    vecData += size * dim;
    std::cout << "shard size: " << size << std::endl;
  }
}