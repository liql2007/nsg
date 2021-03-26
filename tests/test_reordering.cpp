//
// Created by liql2007 on 2021/3/24.
//

#include <compress/reordering/Reordering.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <efanna2e/test_helper.h>

using namespace compress;

void utTest() {
  uint32_t nodeCount = 4;
  std::vector<std::vector<label_t>> graph(nodeCount);
  graph[0] = {1, 3};

  std::vector<label_t> mapping;
  Reordering::genReorderMapping(graph, &mapping);

  std::vector<label_t> expectMapping {2, 3, 0, 1};
  if (mapping != expectMapping) {
    printf("unexpect mapping:\n");
    for (label_t id = 0; id < mapping.size(); ++id) {
      printf("%u --> %u\n", id, mapping[id]);
    }
    exit(-1);
  }
  printf("reordering mapping is ok!\n");

  uint32_t dim = 32;
  std::vector<float> vecs(nodeCount * dim);
  for (uint32_t i = 0; i < nodeCount; ++i) {
    float* vec = vecs.data() + i * dim;
    for (uint32_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
  }
  std::vector<label_t> eps {0};
  Reordering::reorder(dim * sizeof(float), mapping, &graph, &eps,
                      reinterpret_cast<uint8_t*>(vecs.data()));
  if (graph[0].size() != 2 || graph[0] != std::vector<label_t>{3, 1}) {
    printf("graph[0] %zu\n", graph[0].size());
    exit(-1);
  }
  if (!graph[1].empty()) {
    printf("graph[0] %zu", graph[1].size());
    exit(-1);
  }
  if (!graph[2].empty()) {
    printf("graph[0] %zu", graph[2].size());
    exit(-1);
  }
  if (graph[3].size() != 2 || graph[3] != std::vector<label_t>{2, 1}) {
    printf("graph[0] %zu", graph[3].size());
    exit(-1);
  }
  if (vecs[0] != 2 || vecs[dim] != 3 || vecs[2 * dim] != 0 ||
      vecs[3 * dim] != 1) {
    printf(" vecs: %.1f, %.1f, %.1f, %.1f", vecs[0], vecs[dim],
           vecs[2 * dim], vecs[3 * dim]);
    exit(-1);
  }
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0]
              << " data_file nsg_path ground_truth "
                 "new_data_file new_nsg_path new_ground_truth"
              << std::endl;
    exit(-1);
  }
  std::string dataPath = argv[1];
  std::string nsgPath = argv[2];
  std::string groundTruthPath = argv[3];

  float* data_load;
  unsigned points_num, dim;
  std::cout << dataPath << std::endl;
  load_data(dataPath.c_str(), data_load, points_num, dim);

  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  std::cout << nsgPath << std::endl;
  index.Load(nsgPath.c_str());

  unsigned truthItemNum;
  unsigned queryNum;
  unsigned* gdata;
  std::cout << groundTruthPath << std::endl;
  load_data(groundTruthPath.c_str(), gdata, queryNum, truthItemNum);

  std::vector<uint32_t> mapping;
  if (!index.ReorderGraph((uint8_t*)data_load, &mapping)) {
    printf("reorder error");
    exit(-1);
  }
  for (unsigned i = 0; i < queryNum; ++i) {
    auto arr = gdata + i * truthItemNum;
    for (unsigned j = 0; j < truthItemNum; ++j) {
      arr[j] = mapping[arr[j]];
    }
  }


  std::string cvDataPath = argv[4];
  std::string cvNsgPath = argv[5];
  std::string cvGroundTruthPath = argv[6];
  std::cout << cvDataPath << std::endl;
  save_data(cvDataPath.c_str(), data_load, points_num, dim);

  std::cout << cvNsgPath << std::endl;
  index.Save(cvNsgPath.c_str());

  std::cout << cvGroundTruthPath << std::endl;
  save_data(cvGroundTruthPath.c_str(), gdata, queryNum, truthItemNum);

  return 0;
}