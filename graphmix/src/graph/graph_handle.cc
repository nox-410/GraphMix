#include "graph/graph_handle.h"

#include "ps/internal/postoffice.h"

namespace ps {

void GraphHandle::serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response) {
  auto keys = get<0>(request);
  if (keys.empty()) return;
  size_t n = keys.size();
  SArray<size_t> offset(n + 1);
  SArray<graph_float> f_feat(n * meta_.f_len);
  SArray<graph_int> i_feat(n * meta_.i_len);
  offset[0] = 0;
  for (size_t i = 0; i < n; i++) {
    CHECK(keys[i] >= local_offset_ && keys[i] < local_offset_ + num_local_nodes_);
    auto node = nodes_[keys[i] - local_offset_];
    offset[i + 1] = offset[i] + node.edge.size();
  }
  SArray<node_id> edge(offset[n]);
  for (size_t i = 0; i < n; i++) {
    auto node = nodes_[keys[i] - local_offset_];
    std::copy(node.f_feat.begin(), node.f_feat.end(), &f_feat[i * meta_.f_len]);
    std::copy(node.i_feat.begin(), node.i_feat.end(), &i_feat[i * meta_.i_len]);
    std::copy(node.edge.begin(), node.edge.end(), &edge[offset[i]]);
  }
  get<0>(response) = f_feat;
  get<1>(response) = i_feat;
  get<2>(response) = edge;
  get<3>(response) = offset;
}

void GraphHandle::serve(const PSFData<GraphPull>::Request &request, PSFData<GraphPull>::Response &response) {
  GraphMiniBatch result;
  graph_queue_.pop(result);
}

void GraphHandle::initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset) {
  PYTHON_CHECK_ARRAY(offset);
  meta_.f_len = f_len;
  meta_.i_len = i_len;
  meta_.rank = Postoffice::Get()->my_rank();
  meta_.nrank = Postoffice::Get()->num_servers();
  meta_.offset = std::vector<node_id>(meta_.nrank + 1);
  assert(offset.size() == meta_.nrank + 1);
  for (int i = 0; i < meta_.nrank + 1; i++)
    meta_.offset[i] = offset.at(i);
  num_local_nodes_ = meta_.offset[meta_.rank + 1] - meta_.offset[meta_.rank];
  local_offset_ = meta_.offset[meta_.rank];
  nodes_.reserve(num_local_nodes_);
}

void GraphHandle::initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges) {
  PYTHON_CHECK_ARRAY(f_feat);
  PYTHON_CHECK_ARRAY(i_feat);
  PYTHON_CHECK_ARRAY(edges);
  CHECK(f_feat.ndim() == 2 && f_feat.shape(0) == num_local_nodes_ && (size_t)f_feat.shape(1) == meta_.f_len);
  CHECK(i_feat.ndim() == 2 && i_feat.shape(0) == num_local_nodes_ && (size_t)i_feat.shape(1) == meta_.i_len);
  CHECK(edges.ndim() == 2 && edges.shape(0) == 2);
  size_t nedges = edges.shape(1);
  for (node_id i = 0 ; i < num_local_nodes_; i++) {
    NodeData node;
    node.i_feat.CopyFrom(i_feat.data(i, 0), meta_.i_len);
    node.f_feat.CopyFrom(f_feat.data(i, 0), meta_.f_len);
    nodes_.emplace_back(node);
  }
  for (size_t i = 0; i < nedges; i++) {
    node_id u = edges.at(0, i), v = edges.at(1, i);
    CHECK(u >= local_offset_ && u < local_offset_ + num_local_nodes_);
    nodes_[u - local_offset_].edge.push_back(v);
  }
}

void GraphHandle::initBinding(py::module &m) {
  py::class_<GraphHandle, std::shared_ptr<GraphHandle>>(m, "Graph handle")
    .def("init_meta", &GraphHandle::initMeta)
    .def("init_data", &GraphHandle::initData);
}

} // namespace ps
