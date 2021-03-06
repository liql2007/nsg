//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <efanna2e/test_helper.h>
#include <compress/stream_vbyte/StreamVByte.h>
#include <sys/mman.h>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
#ifdef __AVX__
    std::cout << "__AVX__" << std::endl;
#endif

  if (argc != 7) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K ground_truth"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);
  auto nsgPath = argv[3];
  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);
  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;
  std::cout << nsgPath << std::endl;
  auto groundTruthPath = argv[6];

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(nsgPath);

#define PT 1
#if PT == 1
  printf("[StreamVByteSIMD] compress\n");
  using Packer = efanna2e::NeighborsPacker<compress::StreamVByteSIMD>;
#elif PT == 2
  printf("[StreamVByte] compress\n");
  using Packer = efanna2e::NeighborsPacker<compress::StreamVByte>;
#else
  printf("[None] compress\n");
  using Packer = efanna2e::NeighborsPacker<void>;
#endif
  Packer packer;
  index.OptimizeGraph(data_load, packer);
  delete[] data_load;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

//  while(true) {
  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);
  unsigned totalHops = 0;
  unsigned totalVisit = 0;
  std::cout << "Begin search" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  std::vector<efanna2e::Neighbor> retset;
  std::vector<efanna2e::Neighbor> queueData;
  queueData.reserve(L * 2);
  std::vector<unsigned> flags(index.getVecNum() + 1, 0);
  mlock(flags.data(), flags.size() * sizeof(unsigned));
  for (size_t i = 0; i < query_num; i++) {
    flags.back() = i + 1;
    // index.SearchWithOptGraph(query_load + i * dim, paras, flags, retset);
    index.SearchWithOptGraph(query_load + i * dim, paras, flags,
                             retset, packer);
    // index.SearchWithOptGraph2(query_load + i * dim, paras, flags, queueData, retset);
    totalHops += index.getHops();
    totalVisit += index.getVisitNum();
    for (size_t j = 0; j < K; j++) {
      res[i][j] = retset[j].id;
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "eps num: " << index.getEps().size() << "\n";
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
  // }

  return 0;
}
