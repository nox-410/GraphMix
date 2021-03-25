#pragma once

#include "ps/psf/PSFunc.h"
#include "graph/graph_type.h"

#include "common/thread_safe_hash_map.h"

namespace ps {

class GraphHandle {
public:
  GraphHandle() {}
  GraphHandle(const GraphHandle& handle) {}

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response);
  void serve(const PSFData<NodePush>::Request &request, PSFData<NodePush>::Response &response);

private:
  std::vector<NodeData> nodes;
  GraphMetaData data;

  typedef threadsafe_unordered_map<node_id, NodeData> tmap;
  tmap store;
  size_t i_len, f_len;
  const tmap& const_store = store; // const reference to force compiler to use read lock
};

} // namespace ps
