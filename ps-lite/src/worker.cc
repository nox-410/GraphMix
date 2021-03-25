#include "ps/worker/worker.h"

int Worker::getserver(node_id idx) {
  int server = 0;
  while (idx >= meta_.offset[server + 1]) server++;
  return server;
}

Worker::Worker() : _kvworker(0, 0) {}

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
  int nserver = Postoffice::Get()->num_servers();
  std::vector<SArray<node_id>> keys(nserver);
  for (size_t i = 0; i < n; i++) {
    keys[getserver(indices[i])].push_back(indices[i]);
    nodes[indices[i]] = NodeData(); // avoid race condition in callback
  }
  for (int server = 0; server < nserver; server++) {
    if (keys[server].size() == 0) continue;
    auto pull_keys = keys[server];
    PSFData<NodePull>::Request request(pull_keys);
    auto callback = [pull_keys] (const PSFData<NodePull>::Response &response, NodePack &nodes){
      auto f_feat = std::get<0>(response);
      auto i_feat= std::get<1>(response);
      auto edge = std::get<2>(response);
      auto offset = std::get<3>(response);
      auto f_len = f_feat.size() / (offset.size() - 1), i_len = i_feat.size() / (offset.size() - 1);
      CHECK_EQ(offset[offset.size() - 1], edge.size()) << std::endl;
      for (size_t i = 0; i < pull_keys.size(); i++) {
        auto &node = nodes[pull_keys[i]];
        node.f_feat = f_feat.segment(i * f_len, (i + 1) * f_len);
        node.i_feat = i_feat.segment(i * i_len, (i + 1) * i_len);
        node.edge = edge.segment(offset[i], offset[i + 1]);
      }
    };
    auto cb = std::bind(callback, std::placeholders::_1, std::ref(nodes));
    int ts = _kvworker.Request<NodePull>(request, cb, server);
    timestamps.push_back(ts);
  }
  return cur_query;
}

/*
    wait_data waits until a query success
*/
void Worker::waitData(query_t query) {
  py::gil_scoped_release release;
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

void Worker::initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset, int target_server) {
  PYTHON_CHECK_ARRAY(offset);
  meta_.f_len = f_len;
  meta_.i_len = i_len;
  meta_.rank = target_server;
  meta_.nrank = Postoffice::Get()->num_servers();
  meta_.offset = std::vector<node_id>(meta_.nrank + 1);
  assert(offset.size() == meta_.nrank + 1);
  for (int i = 0; i < meta_.nrank + 1; i++)
    meta_.offset[i] = offset.at(i);
}

void Worker::initBinding(py::module& m) {
  py::class_<Worker>(m, "graph worker")
    .def("pull", &Worker::pullData)
    .def("wait", &Worker::waitData)
    .def("init_meta", &Worker::initMeta);
  m.def("get_handle", Worker::Get, py::return_value_policy::reference);
}

Worker& Worker::Get() {
  static Worker w;
  return w;
}
