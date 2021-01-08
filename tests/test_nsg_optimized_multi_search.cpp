#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <efanna2e/test_helper.h>
#include <sys/mman.h>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
#ifdef __AVX__
    std::cout << "__AVX__" << std::endl;
#endif

  if (argc != 8) {
    std::cout << argv[0]
              << " data_file_pattern query_file nsg_path_pattern nsg_num search_L search_K ground_truth"
              << std::endl;
    exit(-1);
  }
  auto dataPathPattern = argv[1];
  auto queryFilePath = argv[2];
  auto nsgPathPattern = argv[3];
  auto nsgNum = (unsigned)atoi(argv[4]);
  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);
  auto groundTruthPath = argv[7];
  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;
  std::cout << nsgPathPattern << ", " << nsgNum << std::endl;

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  float* query_load = NULL;
  unsigned query_num, dim;
  std::cout << "load query: " << queryFilePath << std::endl;
  load_data(queryFilePath, query_load, query_num, dim);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  std::vector<efanna2e::IndexNSG*> indexVec(nsgNum);
  std::vector<unsigned> offsetVec(nsgNum, 0);
  unsigned max_points_num = 0;
  for (unsigned i = 0; i < nsgNum; ++i) {
    char path[1024];

    sprintf(path, dataPathPattern, i+1);
    float* data_load = NULL;
    unsigned points_num, dimI;
    std::cout << "load vec: " << path << std::endl;
    load_data(path, data_load, points_num, dimI);
    assert(dimI == dim);
    max_points_num = std::max(max_points_num, points_num);

    indexVec[i] = new efanna2e::IndexNSG(dim, points_num, efanna2e::FAST_L2, nullptr);
    sprintf(path, nsgPathPattern, i+1);
    std::cout << "load nsg: " << path << std::endl;
    indexVec[i]->Load(path);
    indexVec[i]->OptimizeGraph(data_load);
    delete[] data_load;

    if (i > 0) {
      offsetVec[i] = offsetVec[i-1] + indexVec[i-1]->getVecNum();
    }
  }

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  unsigned totalHops = 0;
  unsigned totalVisit = 0;
  std::cout << "Begin search" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  std::vector<unsigned> flags(max_points_num + 1, 0);
  mlock(flags.data(), flags.size() * sizeof(unsigned));
  std::vector<efanna2e::Neighbor> globalRetSet, partRetSet;
  for (size_t i = 0; i < query_num; i++) {
    auto query = query_load + i * dim;
    globalRetSet.clear();
    partRetSet.clear();
    for (unsigned partId = 0; partId < nsgNum; ++partId) {
      flags.back() = i * nsgNum + partId;
      auto& index = *indexVec[partId];
      index.SearchWithOptGraph(query, paras, flags, partRetSet);
      totalHops += index.getHops();
      totalVisit += index.getVisitNum();
      if (partId == 0) {
        globalRetSet = partRetSet;
      } else {
        for (unsigned j = 0; j < K; ++j) {
          auto& partRes = partRetSet[j];
          partRes.id += offsetVec[partId];
          InsertIntoPool(globalRetSet.data(), K, partRes);
        }
      }
    }
    res[i].resize(K);
    for (size_t j = 0; j < K; j++) {
      res[i][j] = globalRetSet[j].id;
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";
  std::cerr << "QPS: " << query_num / diff.count() << std::endl;
  std::cerr << "AVG latency(ms): " << 1000 * diff.count() / query_num
            << std::endl;
  std::cerr << "AVG visit num: " << (float) totalVisit / query_num
            << std::endl;
  std::cerr << "AVG hop num: " << (float) totalHops / query_num
            << std::endl;

  GroundTruth truth(K);
  truth.load(groundTruthPath);
  truth.recallRate(res);

  return 0;
}
