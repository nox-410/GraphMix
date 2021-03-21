#pragma once

#include "ps/psf/PSFunc.h"
#include "common/graph_type.h"

#include "common/thread_safe_hash_map.h"

namespace ps {

class KVServerMatrixHandle {
public:
  KVServerMatrixHandle() {}
  KVServerMatrixHandle(const KVServerMatrixHandle& handle) {}

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response) {
    Key k = get<0>(request);
    auto iter = const_store.find(k);
    if (iter != const_store.end()) {
      get<0>(response) = iter->second.f_feat;
      get<1>(response) = iter->second.i_feat;
      get<2>(response) = iter->second.edge;
    } else {
      LF << "Key does not exist on PS in DensePull:" << k;
    }
  }

  void serve(const PSFData<NodePush>::Request &request, PSFData<NodePush>::Response &response) {
    Key k = get<0>(request);
    if (const_store.find(k) == const_store.end()) {
      auto node = NodeData();
      node.f_feat = get<1>(request);
      node.i_feat = get<2>(request);
      node.edge = get<3>(request);
      store[k] = node;
    } else {
      LF << "Key already exist on PS in DensePush:" << k;
    }
  }

private:
  typedef threadsafe_unordered_map<Key, NodeData> tmap;
  tmap store;
  const tmap& const_store = store; // const reference to force compiler to use read lock
};

} // namespace ps
