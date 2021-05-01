#pragma once

#include "ps/kvapp.h"
#include "graph/graph.h"
#include "common/binding.h"

using namespace ps;

class GraphClient {
public:
  GraphClient(int port = -1);
  static std::shared_ptr<GraphClient> Get();
  // for data push&pull
  typedef uint64_t query_t;
  query_t pullData(py::array_t<node_id> indices, NodePack &nodes);
  query_t pullGraph(py::args args);
  /*
    wait_data waits until a query success
  */
  void waitData(query_t query);
  std::shared_ptr<PyGraph> resolveGraph(query_t query);
  static void initBinding(py::module &m);
  auto& getKVApp() { return kvapp_; }
  py::dict getMeta() { return dict_meta_; }
  void initMeta(py::dict meta);
private:
  py::dict dict_meta_;
  query_t pullData_impl(const node_id* indices, size_t n, NodePack &nodes);
  void waitTimestamp(int timestamp) { kvapp_->Wait(timestamp); }
  // used this hold to thread_pool return object
  std::unordered_map<query_t, std::vector<int>> query2timestamp;
  // data_pull & data_push query, increase 1 each call
  query_t next_query = 0;
  // protect query2timestamp, graph_map_ and next_query
  std::mutex data_mu;
  std::unique_ptr<KVApp<EmptyHandler>> kvapp_;
  GraphMetaData meta_;
  std::unordered_map<query_t, std::shared_ptr<PyGraph>> graph_map_;
  bool stand_alone_;
  int getserver(node_id idx);
};
