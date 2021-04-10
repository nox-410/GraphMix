#pragma once

#include <metis.h>

#include "common/binding.h"
#include "common/sarray.h"
#include "graph/graph_type.h"
#include "graph/sampler.h"

class PyGraph {
private:
  size_t nnodes_;
  SArray<node_id> edge_index_u_, edge_index_v_;
  SArray<graph_float> f_feat_;
  SArray<graph_int> i_feat_;
  std::string format_="coo";
  ps::SamplerType tag_ = ps::SamplerType::kNumSamplerType;
  SArray<graph_int> extra_;
public:
  PyGraph(SArray<node_id> edge_index_u, SArray<node_id> edge_index_v, size_t num_nodes, std::string format="coo");
  ~PyGraph() {}

  // Getter & Setter
  auto getFormat() { return format_; }
  auto nNodes() { return nnodes_; }
  auto nEdges() { return edge_index_v_.size(); }
  auto getEdgeIndex() { return py::make_tuple(binding::svec_nocp(edge_index_u_), binding::svec_nocp(edge_index_v_)); }
  py::array_t<graph_float> getFloatFeat();
  py::array_t<graph_int> getIntFeat();
  void setFeature(SArray<graph_float>, SArray<graph_int>);
  void setTag(int tag) { tag_ = static_cast<ps::SamplerType>(tag); }
  auto getTag() { return tag_; }
  void setExtra(SArray<graph_int> extra);
  py::array_t<graph_int> getExtra();

  //Graph common API
  void addSelfLoop();
  void removeSelfLoop();
  void convert2coo();
  void convert2csr();
  double denseEfficiency();
  std::vector<long> degree();
  py::array_t<float> gcnNorm(bool use_original_gcn_norm);

  //Graph Partition API
  py::list part_graph(int nparts, bool balance_edge);
  std::vector<idx_t> partition(idx_t nparts, bool balance_edge);
  py::array_t<idx_t> PyPartition(idx_t nparts);

  static void initBinding(py::module &m);
};

std::shared_ptr<PyGraph> makeGraph(py::array_t<node_id> edge_index, size_t num_nodes);
