#include "graph/remote_handle.h"

namespace ps {

RemoteHandle::RemoteHandle(std::shared_ptr<KVApp<GraphHandle>> app) {
  CHECK(app);
  kvapp_ = app;
  for (int i = 0; i< static_cast<int>(SamplerType::kNumSamplerType); i++) {
    SamplerType tp = static_cast<SamplerType>(i);
    auto ptr = std::make_unique<rigtorp::MPMCQueue<sampleState>>(256);
    recv_queue_.emplace(tp, std::move(ptr));
  }
}

void RemoteHandle::initCache(size_t cache_size, cache::policy cache_policy) {
  CHECK(!cache_) << "Already have a cache";
  switch (cache_policy)
  {
  case cache::policy::LRU:
    cache_ = std::make_unique<cache::LRUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFU:
    cache_ = std::make_unique<cache::LFUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFUOpt:
    cache_ = std::make_unique<cache::LFUOptCache<node_id, NodeData>>(cache_size);
    break;
  default:
    std::runtime_error("Cache Policy Error");
  }
}

void RemoteHandle::defaultCallback(const sampleState &state) {
  CHECK(state);
  // cache insert
  if (cache_) {
    std::lock_guard lock(cache_mtx_);
    for (node_id node : state->query_nodes) {
      CHECK(state->recvNodes.count(node));
      cache_->insert(node, state->recvNodes[node]);
    }
  }
  // If we are receiving too much, we will discard them.
  // This is not likely to happen
  recv_queue_[state->type]->try_push(state);
}

sampleState makeSampleState() {
  return std::make_shared<_sampleState>();
}

sampleState RemoteHandle::getSampleState(SamplerType type) {
  sampleState state;
  bool success = false;
  success = recv_queue_[type]->try_pop(state);
  if (success) {
    return state;
  } else {
    state = makeSampleState();
    state->type = type;
    return state;
  }
}

void RemoteHandle::queryRemote(sampleState state) {
  CHECK(state != nullptr);
  if (cache_) {
    std::lock_guard lock(cache_mtx_);
  }
  PSFData<NodePull>::Request request;
  std::get<0>(request) = state->query_nodes;
  //auto cb = std::bind(&RemoteHandle::defaultCallback, this, std::placeholders::_1, state);
  //kvapp_->Request<NodePull>(request, cb, 0);
}

} // namespace ps
