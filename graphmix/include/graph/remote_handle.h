#pragma once

#include "graph/sampler.h"
#include "ps/kvapp.h"
#include "common/cache.h"
#include "ps/internal/threadsafe_queue.h"

#include <set>

namespace ps {

class GraphHandle;

class RemoteHandle {
public:
  RemoteHandle(std::shared_ptr<KVApp<GraphHandle>>, GraphHandle*);
  void initCache(size_t, cache::policy);
  void queryRemote(sampleState state);
  void pushStopCommand(SamplerType);
  sampleState getSampleState(SamplerType type);

  // Profile data
  size_t total_cnt_ = 0, cache_miss_cnt_ = 0, nonlocal_cnt_ = 0;
private:
  // Insert cache, push to recv_queue_ in defaultCallback
  void defaultCallback(const sampleState &state);
  void partialCallback(sampleState state, SArray<node_id> pull_keys, const PSFData<NodePull>::Response &response);
  void filterNode(sampleState &state);
  std::unique_ptr<cache::Cache<node_id, NodeData>> cache_;
  std::mutex cache_mtx_;
  std::unordered_map<SamplerType, std::unique_ptr<ThreadsafeQueue<sampleState>>> recv_queue_;
  std::shared_ptr<KVApp<GraphHandle>> kvapp_;
  std::shared_ptr<GraphHandle> handle_;
  std::atomic<int> on_flight_[static_cast<int>(SamplerType::kNumSamplerType)];
};

} // namespace ps
