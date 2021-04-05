#include "graph/graph_client.h"

int GraphClient::getserver(node_id idx) {
  int server = 0;
  while (idx >= meta_.offset[server + 1]) server++;
  return server;
}

GraphClient::GraphClient() : _kvworker() {}

GraphClient::query_t
GraphClient::pullData(py::array_t<node_id> indices, NodePack &nodes) {
  PYTHON_CHECK_ARRAY(indices);
  return pullData_impl(indices.data(), indices.size(), nodes);
}

GraphClient::query_t
GraphClient::pullData_impl(const node_id* indices, size_t n, NodePack &nodes) {
  py::gil_scoped_release release;
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();
  int nserver = Postoffice::Get()->num_servers();
  std::vector<SArray<node_id>> keys(nserver);
  nodes.reserve(n);
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

GraphClient::query_t
GraphClient::pullGraph() {
  py::gil_scoped_release release;
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();
  PSFData<GraphPull>::Request request;
  auto cb = [cur_query, this] (const PSFData<GraphPull>::Response &response) {
    auto &csr_i = std::get<2>(response);
    auto &csr_j = std::get<3>(response);
    auto graph = std::make_shared<PyGraph>(csr_i, csr_j, csr_i.size() - 1);
    graph->setFeature(std::get<0>(response), std::get<1>(response));
    graph_map_[cur_query] = graph;
  };
  auto ts = _kvworker.Request<GraphPull>(request, cb, meta_.rank);
  timestamps.push_back(ts);
  return cur_query;
}

/*
    wait_data waits until a query success
*/
void GraphClient::waitData(query_t query) {
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

std::shared_ptr<PyGraph> GraphClient::resolveGraph(query_t query) {
  waitData(query);
  if (graph_map_.count(query) == 0) {
    throw std::runtime_error("Graph for the query is not found.");
  }
  auto result = graph_map_[query];
  graph_map_.erase(query);
  return result;
}

void GraphClient::initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset, int target_server) {
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

void GraphClient::initBinding(py::module& m) {
  py::class_<GraphClient, std::shared_ptr<GraphClient>>(m, "graph client")
    .def("pull", &GraphClient::pullData)
    .def("pull_graph", &GraphClient::pullGraph)
    .def("wait", &GraphClient::waitData)
    .def("resolve", &GraphClient::resolveGraph)
    .def("init_meta", &GraphClient::initMeta);
  m.def("get_client", GraphClient::Get);
}

std::shared_ptr<GraphClient> GraphClient::Get() {
  static std::shared_ptr<GraphClient> ptr;
  static std::once_flag oc;
  std::call_once(oc, []() { ptr = std::make_shared<GraphClient>(); });
  return ptr;
}
