#pragma once

#include "ps/psf/PSFunc.h"
#include "graph/graph_type.h"

#include "common/binding.h"

namespace ps {

class GraphHandle {
public:
  GraphHandle() {}
  GraphHandle(const GraphHandle& handle) {}

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response);
  static void initBinding(py::module &m);
  void initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset);
  void initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges);
private:
  std::vector<NodeData> nodes_;
  GraphMetaData meta_;
  size_t num_local_nodes_;
  node_id local_offset_;
};

} // namespace ps
