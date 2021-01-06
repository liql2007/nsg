#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

namespace efanna2e {

const double kPi = 3.14159265358979323846264;
const float cosinThreshold = std::cos(60.0 / 180 * kPi);

#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  unsigned epNum = eps_.size();
  out.write((char *)&epNum, sizeof(unsigned));
  out.write((char *)eps_.data(), epNum * sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  unsigned epNum = 0;
  in.read((char *)&epNum, sizeof(unsigned));
  assert(epNum > 0);
  eps_.resize(epNum);
  in.read((char *)eps_.data(), epNum * sizeof(unsigned));
//  if (eps_.size() < 20 && nd_ > 1000) {
//    std::mt19937 rng(rand());
//    while (eps_.size() < 20) {
//      auto id = rng() % nd_;
//      if (std::find(eps_.begin(), eps_.end(), id) == eps_.end()) {
//        eps_.push_back(id);
//      }
//    }
//  }

  // eps_.resize(1);

  // width=100;
  double cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  assert(final_graph_.size() == nd_);
  avgDegree_ = cc / nd_;
  std::cerr << "Average Degree = " << avgDegree_
            << ",  Width:" << width << std::endl;
}
void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  boost::dynamic_bitset<> flags{nd_, 0};
  return get_neighbors(query, parameter, flags, retset, fullset);
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.reserve(L + 1);
  std::vector<unsigned> init_ids(eps_.size());
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  for (unsigned i = 0; i < init_ids.size() ; i++) {
    init_ids[i] = eps_[i];
    flags[init_ids[i]] = true;
  }
  while (init_ids.size() < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids.push_back(id);
    flags[id] = true;
  }

  unsigned poolSize = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset.emplace_back(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    poolSize++;
  }

  std::sort(retset.begin(), retset.begin() + poolSize);
  if (poolSize > L) {
    poolSize = L;
  }
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (poolSize == L && dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), poolSize, nn);

        if (poolSize < L) ++poolSize;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const unsigned q, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &pool) {
  unsigned maxC = parameter.Get<unsigned>("C");
  flags[q] = true;
  for (unsigned i = 0; i < final_graph_[q].size() && pool.size() < maxC; i++) {
    unsigned nid = final_graph_[q][i];
    for (unsigned nn = 0; nn < final_graph_[nid].size() && pool.size() < maxC; nn++) {
      unsigned nnid = final_graph_[nid][nn];
      if (flags[nnid]) continue;
      flags[nnid] = true;
      float dist = distance_->compare(data_ + dimension_ * q,
                                      data_ + dimension_ * nnid, dimension_);
      pool.push_back(Neighbor(nnid, dist, true));
    }
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  eps_.resize(1);
  eps_[0] = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  eps_[0] = tmp[0].id;
}

void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  width = range;
  unsigned start = 0;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
#ifndef SSG
  unsigned maxc = parameter.Get<unsigned>("C");
  if (pool.size() > maxc) {
    pool.resize(maxc);
  }
#endif
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size()) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
#ifdef SSG
      float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                     sqrt(p.distance * result[t].distance);
      if (cos_ij > cosinThreshold) {
        occlude = true;
        break;
      }
#else
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
#endif
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    LockGuard guard(locks[des]);
    for (size_t j = 0; j < range; j++) {
      if (des_pool[j].distance == -1) break;
      if (n == des_pool[j].id) {
        dup = 1;
        break;
      }
      temp_pool.push_back(des_pool[j]);
    }

    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
#ifdef SSG
          float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                     sqrt(p.distance * result[t].distance);
          if (cos_ij > cosinThreshold) {
            occlude = true;
            break;
          }
#else
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
#endif
        }
        if (!occlude) result.push_back(p);
      }
      {
        // LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
        if (result.size() < range) {
          des_pool[result.size()].distance = -1;
        }
      }
    } else {
      // LockGuard guard(locks[des]);
      des_pool[temp_pool.size()-1] = sn;
      if (temp_pool.size() < range) des_pool[temp_pool.size()].distance = -1;
//      for (unsigned t = 0; t < range; t++) {
//        if (des_pool[t].distance == -1) {
//          des_pool[t] = sn;
//          if (t + 1 < range) des_pool[t + 1].distance = -1;
//          break;
//        }
//      }
    }
  }
}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);

  auto s1 = std::chrono::high_resolution_clock::now();
  double totalPoolSize = 0;
#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
    double poolSize = 0;
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
#ifdef SSG
      get_neighbors(n, parameters, flags, pool);
#else
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
#endif
      poolSize += pool.size();
      sync_prune(n, pool, parameters, flags, cut_graph_);
      /*
    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
      }
      */
    }
#pragma omp critical
    totalPoolSize += poolSize;
  }
  auto s2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = s2 - s1;
  std::cout << "avg candidate num: " << totalPoolSize / nd_ << std::endl;
  std::cout << "first link time: " << diff.count() << "\n";

#pragma omp parallel for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_);
  }
  auto s3 = std::chrono::high_resolution_clock::now();
  diff = s3 - s2;
  std::cout << "inter link time: " << diff.count() << "\n";
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  auto s1 = std::chrono::high_resolution_clock::now();

  tree_grow(parameters);

  auto s2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = s2 - s1;
  std::cout << "dfs time: " << diff.count() << "\n";

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  width = max;
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset;
  retset.reserve(L + 1);
  std::vector<unsigned> init_ids(eps_.size());
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  for (unsigned i = 0; i < init_ids.size() ; i++) {
    init_ids[i] = eps_[i];
    flags[init_ids[i]] = true;
  }

//  while (init_ids.size() < L) {
//    unsigned id = rand() % nd_;
//    if (flags[id]) continue;
//    init_ids.push_back(id);
//    flags[id] = true;
//  }

  unsigned poolSize = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset.emplace_back(id, dist, true);
    // flags[id] = true;
    poolSize++;
  }

  std::sort(retset.begin(), retset.begin() + poolSize);
  if (poolSize > L) {
    poolSize = L;
  }
  int k = 0;
  while (k < (int)poolSize) {
    int nk = poolSize;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (poolSize == L && dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), poolSize, nn);

        if (poolSize < L) ++poolSize;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
  std::vector<Neighbor> retset;
  SearchWithOptGraph(query, parameters, retset);
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query,
                                  const Parameters &parameters,
                                  std::vector<Neighbor>& retset) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  hops = 0;
  visitNum = 0;

  retset.clear();
  retset.reserve(L + 1);
  std::vector<unsigned> init_ids(eps_.size());
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
//  unsigned tmp_l = 0;
//  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
//  unsigned MaxM_ep = *neighbors;
//  neighbors++;

  for (unsigned i = 0; i < init_ids.size() ; i++) {
    init_ids[i] = eps_[i];
    flags[init_ids[i]] = true;
  }

//  while (init_ids.size() < 10) {
//    unsigned id = rand() % nd_;
//    if (flags[id]) continue;
//    init_ids.push_back(id);
//    flags[id] = true;
//  }


  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  unsigned poolSize = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset.emplace_back(id, dist, true);
    flags[id] = true;
    ++visitNum;
    poolSize++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + poolSize);
  retset.resize(L + 1);
  if (poolSize > L) {
    poolSize = L;
  }
  int k = 0;
  while (k < (int)poolSize) {
    int nk = poolSize;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      ++hops;
      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        ++visitNum;
        if (poolSize == L && dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), poolSize, nn);

        if (poolSize < L) ++poolSize;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
  opt_graph_ = (char *)malloc(node_size * nd_);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}
void IndexNSG::tree_grow(const Parameters &parameter) {
  unsigned root = eps_[0];
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned pre_linked_cnt = 0;
  unsigned treeNum = 0;
  double avgTreeSize = 0;
  unsigned linked_cnt = 0;
  while (linked_cnt < nd_) {
    DFS(flags, root, linked_cnt);
    // std::cout << unlinked_cnt << '\n';
    ++treeNum;
    avgTreeSize += (linked_cnt - pre_linked_cnt);
    pre_linked_cnt = linked_cnt;
    if (linked_cnt >= nd_) break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  std::cout << "tree num: " << treeNum
            << ", avg tree size: " << avgTreeSize / treeNum << std::endl;
//  for (size_t i = 0; i < nd_; ++i) {
//    if (final_graph_[i].size() > width) {
//      width = final_graph_[i].size();
//    }
//  }
}
}
