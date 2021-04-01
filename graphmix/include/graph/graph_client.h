#pragma once

#include "ps/kvapp.h"
#include "common/binding.h"

using namespace ps;

class GraphClient {
public:
  GraphClient();
  static std::shared_ptr<GraphClient> Get();
  // for data push&pull
  typedef uint64_t query_t;
  query_t pullData(py::array_t<node_id> indices, NodePack &nodes);
  /*
    wait_data waits until a query success
  */
  void waitData(query_t query);
  static void initBinding(py::module &m);
  void initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset, int target_server);

private:
  query_t pullData_impl(const node_id* indices, size_t n, NodePack &nodes);
  void waitTimestamp(int timestamp) { _kvworker.Wait(timestamp); }
  // used this hold to thread_pool return object
  std::unordered_map<query_t, std::vector<int>> query2timestamp;
  // data_pull & data_push query, increase 1 each call
  query_t next_query = 0;
  // protect query2timestamp and next_query
  std::mutex data_mu;
  KVApp<EmptyHandler> _kvworker;
  GraphMetaData meta_;
  int getserver(node_id idx);
};
