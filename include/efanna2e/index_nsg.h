#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e {

extern const double kPi;
extern const float cosinThreshold;

template <class Compressor>
class NeighborsPacker {
public:
  void init(uint32_t maxNeighborCount, uint32_t nodeCount) {
    _maxNeighborCount = maxNeighborCount;
    _buffer.resize(maxNeighborCount);
    _data.resize((std::size_t)nodeCount *
                 Compressor::maxCompressedBytes(maxNeighborCount));
  }

  std::size_t metaBytes() { return sizeof(std::size_t); }

  std::size_t pack(std::vector<uint32_t>& neighbors, uint8_t* meta) {
    assert(neighbors.size() <= _maxNeighborCount);
    std::sort(neighbors.begin(), neighbors.end());
    *reinterpret_cast<std::size_t*>(meta) = _index;
    auto es = Compressor::encode(neighbors.data(), neighbors.size(),
                                 _data.data() + _index);
    _index += es;
    return es;
  }

  uint32_t* unpack(const uint8_t* meta, uint32_t count) {
    assert(count <= _maxNeighborCount);
    auto index = *reinterpret_cast<const std::size_t*>(meta);
    Compressor::decode(_data.data() + index, count, _buffer.data());
    return _buffer.data();
  }

private:
  uint32_t _maxNeighborCount = 0;
  std::vector<uint32_t> _buffer;
  std::vector<uint8_t> _data;
  std::size_t _index = 0;
};

template <>
class NeighborsPacker<void> {
public:
  void init(uint32_t maxNeighborCount, uint32_t nodeCount) {
    _maxNeighborCount = maxNeighborCount;
    _data.resize(sizeof(uint32_t) * nodeCount * maxNeighborCount);
  }

  std::size_t metaBytes() { return sizeof(std::size_t); }

  std::size_t pack(std::vector<uint32_t>& neighbors, uint8_t* meta) {
    assert(neighbors.size() <= _maxNeighborCount);
    *reinterpret_cast<std::size_t*>(meta) = _index;
    auto size = neighbors.size() * sizeof(uint32_t);
    std::memcpy(_data.data() + _index, neighbors.data(), size);
    _index += size;
    return size;
  }

  uint32_t* unpack(const uint8_t* meta, uint32_t count) {
    (void) count;
    assert(count <= _maxNeighborCount);
    auto index = *reinterpret_cast<const std::size_t*>(meta);
    return reinterpret_cast<uint32_t*>(_data.data() + index);
  }

private:
  uint32_t _maxNeighborCount = 0;
  std::vector<uint8_t> _data;
  std::size_t _index = 0;
};

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;

  template <class Packer=NeighborsPacker<void>>
  void SearchWithOptGraph(const float *query,
                          const Parameters &parameters,
                          std::vector<unsigned>& flags,
                          std::vector<Neighbor>& retset,
                          Packer& packer);

  void SearchWithOptGraph(const float *query,
                          const Parameters &parameters,
                          std::vector<unsigned>& flags,
                          std::vector<Neighbor>& retset) {
    NeighborsPacker<void> packer;
    SearchWithOptGraph(query, parameters, flags, retset, packer);
  }

  void SearchWithOptGraph2(const float *query,
                           const Parameters &parameters,
                           std::vector<unsigned>& flags,
                           std::vector<Neighbor>& queueData,
                           std::vector<Neighbor>& retset);

  bool ReorderGraph(uint8_t* vecs, std::vector<uint32_t>* mapping);

  template <class Packer=NeighborsPacker<void>>
  void OptimizeGraph(float* data, Packer& packer);

  void OptimizeGraph(float* data) {
    NeighborsPacker<void> packer;
    OptimizeGraph(data, packer);
  }

  unsigned getVisitNum() const { return visitNum; }

  unsigned getHops() const { return hops; }

  typedef std::vector<std::vector<unsigned > > CompactGraph;

  CompactGraph& graph() { return final_graph_; }

  const CompactGraph& graph() const { return final_graph_; }

  unsigned getWidth() const { return width; }

  void setWidth(unsigned w) { width = w; }

  const std::vector<unsigned>& getEps() const { return eps_; }

  std::vector<unsigned>& getEps() { return eps_; }

  void tree_grow(const Parameters &parameter);

  protected:
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);

  void get_neighbors(const unsigned q,
                     const Parameters &parameter,
                     boost::dynamic_bitset<> &flags,
                     std::vector<Neighbor> &pool);

    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_, SimpleNeighbor *backupGraph);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    // unsigned ep_;
    std::vector<unsigned> eps_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
    unsigned visitNum;
    unsigned hops;
};

template <class Packer>
void IndexNSG::OptimizeGraph(float* data, Packer& packer) {
  packer.init(width, nd_);
  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = sizeof(unsigned) + packer.metaBytes();
  node_size = data_len + neighbor_len;
  opt_graph_ = (char *)malloc(node_size * nd_);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  std::size_t oriSize = 0;
  std::size_t compressedSize = 0;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    assert(k <= width);
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    auto eSize = packer.pack(final_graph_[i], reinterpret_cast<uint8_t*>(
      cur_node_offset + sizeof(unsigned)));
    compressedSize += eSize + sizeof(unsigned);
    oriSize += (k + 1) * sizeof(unsigned);
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  printf("origin size: %.2lfGB, new size: %.2lfGB, rate: %.1lf%%\n",
         oriSize / 1024.0 / 1024.0 / 1024.0,
         compressedSize / 1024.0 / 1024.0 / 1024.0,
         100.0 * compressedSize / oriSize);
  CompactGraph().swap(final_graph_);
}

template <class Packer>
void IndexNSG::SearchWithOptGraph(const float *query,
                                  const Parameters &parameters,
                                  std::vector<unsigned>& flags,
                                  std::vector<Neighbor>& retset,
                                  Packer& packer) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  hops = 0;
  visitNum = 0;

  retset.clear();
  retset.reserve(L + 1);
  std::vector<unsigned> init_ids(eps_.size());
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);


  // flags.reset();
  auto visited = flags.back();
//  unsigned tmp_l = 0;
//  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
//  unsigned MaxM_ep = *neighbors;
//  neighbors++;

  for (unsigned i = 0; i < init_ids.size() ; i++) {
    init_ids[i] = eps_[i];
    flags[init_ids[i]] = visited;
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
    flags[id] = visited;
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
      neighbors = packer.unpack(reinterpret_cast<uint8_t*>(neighbors+1), MaxM);
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id] == visited) continue;
        flags[id] = visited;
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

}

#endif //EFANNA2E_INDEX_NSG_H
