#pragma once

#include "graph/graph_type.h"
#include "common/cache.h"
#include "common/MPMCQueue.h"

namespace ps {

struct sampleState {};

class RemoteHandle {
public:
  RemoteHandle(size_t cache_size, cache::policy cache_policy);
private:
  std::shared_ptr<cache::Cache<node_id, NodeData>> cache_;
  std::mutex cache_mtx_;
  rigtorp::mpmc::Queue<sampleState> recvqueue_;
};

} // namespace ps
