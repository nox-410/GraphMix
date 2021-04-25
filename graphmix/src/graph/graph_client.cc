#include "graph/graph_client.h"

#include <pybind11/eval.h>

int GraphClient::getserver(node_id idx) {
  int server = 0;
  while (idx >= meta_.offset[server + 1]) server++;
  return server;
}

GraphClient::GraphClient(int port) {
  py::gil_scoped_release release;
  stand_alone_ = port > 0;
  kvapp_ = std::make_unique<KVApp<EmptyHandler>>(0, 0, nullptr, port);
  PSFData<MetaPull>::Request request;
  auto cb = [this](const PSFData<MetaPull>::Response &response) {
    auto char_buf = std::get<0>(response);
    std::string st(char_buf.begin(), char_buf.end());
    {
      py::gil_scoped_acquire acquire;
      py::dict meta = py::eval(py::str(st)).cast<py::dict>();
      initMeta(meta);
    }
  };
  int ts = kvapp_->Request<MetaPull>(request, cb , 0);
  kvapp_->Wait(ts);
}

GraphClient::query_t
GraphClient::pullData(py::array_t<node_id> indices, NodePack &nodes) {
  CHECK(!stand_alone_) << "PullNode under standalone mode is not implemented.";
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
  int nserver = meta_.nrank;
  std::vector<SArray<node_id>> keys(nserver);
  nodes.reserve(n);
  for (size_t i = 0; i < n; i++) {
    keys[getserver(indices[i])].push_back(indices[i]);
    nodes[indices[i]] = makeNodeData(); // avoid race condition in callback
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
        node->f_feat.resize(f_len);
        node->i_feat.resize(i_len);
        node->edge.resize(offset[i+1]-offset[i]);
        std::copy(&f_feat[i*f_len], &f_feat[(i + 1) * f_len], node->f_feat.data());
        std::copy(&i_feat[i*i_len], &i_feat[(i + 1) * i_len], node->i_feat.data());
        std::copy(&edge[offset[i]], &edge[offset[i+1]], node->edge.data());
      }
    };
    auto cb = std::bind(callback, std::placeholders::_1, std::ref(nodes));
    int ts = kvapp_->Request<NodePull>(request, cb, server);
    timestamps.push_back(ts);
  }
  return cur_query;
}

GraphClient::query_t
GraphClient::pullGraph(py::args args) {
  SArray<int> priority;
  for (auto item : args) {
    SamplerType sampler = item.cast<SamplerType>();
    priority.push_back(static_cast<int>(sampler));
  }
  py::gil_scoped_release release;
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();
  PSFData<GraphPull>::Request request(std::move(priority));
  auto cb = [cur_query, this] (const PSFData<GraphPull>::Response &response) {
    auto &f_feat = std::get<0>(response);
    auto &i_feat = std::get<1>(response);
    auto &csr_i = std::get<2>(response);
    auto &csr_j = std::get<3>(response);
    int tag = std::get<4>(response);
    size_t num_nodes = f_feat.size() / meta_.f_len;
    // only for graphsage minibatch, coo-format are used
    std::string format = tag == static_cast<int>(SamplerType::kGraphSage) ? "coo" : "csr";
    auto graph = std::make_shared<PyGraph>(csr_i, csr_j, num_nodes, format);
    graph->setFeature(f_feat, i_feat);
    graph->setTag(tag);
    graph->setExtra(std::get<5>(response));
    graph_map_[cur_query] = graph;
  };
  auto ts = kvapp_->Request<GraphPull>(request, cb, meta_.rank);
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

void GraphClient::initMeta(py::dict meta) {
  dict_meta_ = meta;
  if (stand_alone_)
    meta_.rank = 0;
  else
    // decide which server to use
    meta_.rank = Postoffice::Get()->my_rank() * Postoffice::Get()->num_servers() / Postoffice::Get()->num_workers();
  meta_.f_len = meta["float_feature"].cast<size_t>();
  meta_.i_len = meta["int_feature"].cast<size_t>();
  meta_.num_nodes = meta["node"].cast<size_t>();
  meta_.nrank = meta["num_part"].cast<size_t>();
    py::list offset = meta["partition"]["offset"];
  CHECK(int(offset.size()) == meta_.nrank);
  meta_.offset = std::vector<node_id>(meta_.nrank + 1);
  for (int i = 0; i < meta_.nrank; i++)
    meta_.offset[i] = offset[i].cast<node_id>();
  meta_.offset.back() = meta_.num_nodes;
}

std::shared_ptr<GraphClient> createClient(int port) {
  return std::make_shared<GraphClient>(port);
}

void GraphClient::initBinding(py::module& m) {
  py::class_<GraphClient, std::shared_ptr<GraphClient>>(m, "graph client")
    .def_property_readonly("meta", &GraphClient::getMeta)
    .def("pull_node", &GraphClient::pullData)
    .def("pull_graph", &GraphClient::pullGraph)
    .def("wait", &GraphClient::waitData)
    .def("resolve", &GraphClient::resolveGraph);
  m.def("get_client", GraphClient::Get);
  m.def("creat_client", createClient);
}

std::shared_ptr<GraphClient> GraphClient::Get() {
  static std::shared_ptr<GraphClient> ptr;
  static std::once_flag oc;
  std::call_once(oc, []() { ptr = std::make_shared<GraphClient>(); });
  return ptr;
}
