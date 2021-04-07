#include "graph/remote_handle.h"

namespace ps {

RemoteHandle::RemoteHandle(std::shared_ptr<KVApp<GraphHandle>> app)
  : recvqueue_(10) {
  kvapp_ = app;
}

void RemoteHandle::initCache(size_t cache_size, cache::policy cache_policy) {
  CHECK(!cache_);
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

} // namespace ps
