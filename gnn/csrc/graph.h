#ifndef GNN_GRAPH_H
#define GNN_GRAPH_H

#include <metis.h>

#include "common/binding.h"
#include "common/sarray.h"
#include "common/graph_type.h"

class PyGraph {
private:
  size_t nnodes_;
  SArray<long> edge_index_u_, edge_index_v_;
public:
  PyGraph(SArray<long> edge_index_u, SArray<long> edge_index_v, size_t num_nodes);
  ~PyGraph() {}

  //Getter
  auto nNodes() { return nnodes_; }
  auto nEdges() { return edge_index_u_.size(); }
  auto getEdgeIndex() { return py::make_tuple(bind::svec_nocp(edge_index_u_), bind::svec_nocp(edge_index_v_)); }

  //Graph common API
  void addSelfLoop();
  void removeSelfLoop();
  double denseEfficiency();
  std::vector<long> degree();
  py::array_t<float> gcnNorm(bool use_original_gcn_norm);

  //Graph Partition API
  py::list part_graph(int nparts, bool balance_edge);
  std::vector<idx_t> partition(idx_t nparts, bool balance_edge);
  py::array_t<idx_t> PyPartition(idx_t nparts);
};

PyGraph makeGraph(py::array_t<long> edge_index, size_t num_nodes);

#endif /* GNN_GRAPH_H */
