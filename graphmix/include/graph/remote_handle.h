#pragma once

#include "graph/sampler.h"
#include "ps/kvapp.h"
#include "common/cache.h"
#include "common/MPMCQueue.h"

#include <set>

namespace ps {

class GraphHandle;

class _sampleState {
public:
  std::mutex mtx;
  int wait_num;
  std::unordered_set<node_id> query_nodes;
  NodePack recvNodes;
  SamplerType type;
};

typedef std::shared_ptr<_sampleState> sampleState;

sampleState makeSampleState();

class RemoteHandle {
public:
  RemoteHandle(std::shared_ptr<KVApp<GraphHandle>>, GraphHandle*);
  void initCache(size_t, cache::policy);
  void queryRemote(sampleState state);
  sampleState getSampleState(SamplerType type);
private:
  // Insert cache, push to recv_queue_ in defaultCallback
  void defaultCallback(const sampleState &state);
  void partialCallback(sampleState state, SArray<node_id> pull_keys, const PSFData<NodePull>::Response &response);
  void filterNode(sampleState &state);
  std::unique_ptr<cache::Cache<node_id, NodeData>> cache_;
  std::mutex cache_mtx_;
  std::unordered_map<SamplerType, std::unique_ptr<rigtorp::MPMCQueue<sampleState>>> recv_queue_;
  std::shared_ptr<KVApp<GraphHandle>> kvapp_;
  std::shared_ptr<GraphHandle> handle_;
  // size_t onFlight_ = 0;
};

} // namespace ps
