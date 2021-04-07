#include "graph/remote_handle.h"

namespace ps {

RemoteHandle::RemoteHandle(size_t cache_size, cache::policy cache_policy) : recvqueue_(10) {
  switch (cache_policy)
  {
  case cache::policy::LRU:
    cache_ = std::make_shared<cache::LRUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFU:
    cache_ = std::make_shared<cache::LFUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFUOpt:
    cache_ = std::make_shared<cache::LFUOptCache<node_id, NodeData>>(cache_size);
    break;
  default:
    std::runtime_error("Cache Policy Error");
  }
}

} // namespace ps
