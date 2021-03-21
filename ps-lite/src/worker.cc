#include "ps/worker/worker.h"

static Key mapWkeyToSkey(node_id idx) {
  const static std::vector<Range>& server_range = Postoffice::Get()->GetServerKeyRanges();
  int server = idx % server_range.size();
  Key k = server_range[server].begin() + idx;
  return k;
}

Worker::Worker() : _kvworker(0, 0) {}

Worker::query_t
Worker::pushData(py::array_t<node_id> indices, py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges) {
  PYTHON_CHECK_ARRAY(f_feat);
  PYTHON_CHECK_ARRAY(i_feat);
  PYTHON_CHECK_ARRAY(edges);
  size_t nnodes = indices.size();
  assert(f_feat.ndim() == 2 && f_feat.shape(0) == nnodes);
  assert(i_feat.ndim() == 2 && i_feat.shape(0) == nnodes);
  assert(edges.ndim() == 2 && edges.shape(0) == 2);
  size_t f_len = f_feat.shape(1), i_len = i_feat.shape(1);
  size_t nedges = edges.shape(1);

  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();

  auto nodes = NodePack();
  for (size_t i = 0 ; i < nnodes; i++) {
    NodeData node;
    node.i_feat = SArray<graph_int>(i_feat.mutable_at(i), i_len);
    node.f_feat = SArray<graph_float>(f_feat.mutable_at(i), f_len);
    nodes.emplace(indices.at(i), node);
  }
  std::unordered_map<node_id, size_t> deg;
  for (size_t i = 0 ; i < nedges; i++) deg[edges.at(0, i)]++;
  for (auto &node : nodes) node.second.edge.reserve(deg[node.first]);
  for (size_t i = 0; i < nedges; i++) {
    node_id u = edges.at(0, i), v = edges.at(1, i);
    nodes[u].edge.push_back(v);
  }
  auto cb = getCallBack<NodePush>();
  for (auto &node : nodes) {
    PSFData<NodePush>::Request request(mapWkeyToSkey(node.first), node.second.f_feat, node.second.i_feat, node.second.edge);
    int ts = _kvworker.Request<NodePush>(request, cb);
    timestamps.push_back(ts);
  }
  return cur_query;
}

Worker::query_t
Worker::pullData(py::array_t<node_id> indices, NodePack &nodes) {
  PYTHON_CHECK_ARRAY(indices);
  return pullData_impl(indices.data(), indices.size(), nodes);
}

Worker::query_t
Worker::pullData_impl(const node_id* indices, size_t n, NodePack &nodes) {
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();
  for (size_t i = 0; i < n; i++) {
    node_id idx = indices[i];
    auto cb = getCallBack<NodePull>(nodes[idx]);
    PSFData<NodePull>::Request request(mapWkeyToSkey(idx));
    int ts = _kvworker.Request<NodePull>(request, cb);
    timestamps.push_back(ts);
  }
  return cur_query;
}

/*
    wait_data waits until a query success
*/
void Worker::waitData(query_t query) {
  data_mu.lock();
  auto iter = query2timestamp.find(query);
  if (iter == query2timestamp.end()) {
    data_mu.unlock();
    LG << "Wait on empty query " << query;
    return;
  } else {
    auto timestamps = std::move(iter->second);
    query2timestamp.erase(iter);
    data_mu.unlock();
    for (int t : timestamps) {
      waitTimestamp(t);
    }
  }
}

void Worker::initBinding(py::module& m) {
  py::class_<Worker>(m, "graph worker")
    .def("push", &Worker::pushData)
    .def("pull", &Worker::pullData)
    .def("wait", &Worker::waitData);
  m.def("get_handle", Worker::Get, py::return_value_policy::reference);
}

Worker& Worker::Get() {
  static Worker w;
  return w;
}
