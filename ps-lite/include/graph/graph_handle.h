#pragma once

#include "ps/psf/PSFunc.h"
#include "graph/graph_type.h"
#include "common/binding.h"
#include "common/MPMCQueue.h"

#include <thread>

namespace ps {

class GraphHandle {
public:
  GraphHandle() : graph_queue_(10) {}

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response);
  void serve(const PSFData<GraphPull>::Request &request, PSFData<GraphPull>::Response &response);
  static void initBinding(py::module &m);
  void initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset);
  void initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges);
  void push(const GraphMiniBatch &graph) { graph_queue_.push(graph); }
private:
  std::vector<NodeData> nodes_;
  GraphMetaData meta_;
  node_id num_local_nodes_;
  node_id local_offset_;
  rigtorp::mpmc::Queue<GraphMiniBatch> graph_queue_;
};

} // namespace ps
