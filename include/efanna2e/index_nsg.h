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
  void SearchWithOptGraph(const float *query,
                          const Parameters &parameters,
                          std::vector<unsigned>& flags,
                          std::vector<Neighbor>& retset);
  void SearchWithOptGraph2(const float *query,
                           const Parameters &parameters,
                           std::vector<unsigned>& flags,
                           std::vector<Neighbor>& queueData,
                           std::vector<Neighbor>& retset);
  void OptimizeGraph(float* data);

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
}

#endif //EFANNA2E_INDEX_NSG_H
