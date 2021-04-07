#pragma once

#include "graph/graph_type.h"
#include "ps/kvapp.h"
#include "common/cache.h"
#include "common/MPMCQueue.h"

namespace ps {

struct sampleState {
  std::vector<node_id> query_nodes;
  PSFData<NodePull>::Response response;
};

class GraphHandle;

class RemoteHandle {
public:
  RemoteHandle(std::shared_ptr<KVApp<GraphHandle>>);
  void initCache(size_t, cache::policy);
  //void getNode(sampleState &&state);
private:
  std::unique_ptr<cache::Cache<node_id, NodeData>> cache_;
  std::mutex cache_mtx_;
  rigtorp::mpmc::Queue<sampleState> recvqueue_;
  std::shared_ptr<KVApp<GraphHandle>> kvapp_;
  // size_t onFlight_ = 0;
};

} // namespace ps
