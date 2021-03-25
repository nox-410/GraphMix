#include "graph/graph_handle.h"

namespace ps {
void GraphHandle::serve(const PSFData<NodePush>::Request& request, PSFData<NodePush>::Response& response) {
  node_id k = get<0>(request);
  if (const_store.find(k) == const_store.end()) {
    auto node = NodeData();
    node.f_feat = get<1>(request);
    node.i_feat = get<2>(request);
    node.edge = get<3>(request);
    store[k] = node;
    i_len = node.i_feat.size();
    f_len = node.f_feat.size();
  } else {
    LF << "Key already exist on PS in NodePush:" << k;
  }
}

void GraphHandle::serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response) {
  auto keys = get<0>(request);
  if (keys.empty()) return;
  size_t n = keys.size();
  SArray<size_t> offset(n + 1);
  SArray<graph_float> f_feat(n * f_len);
  SArray<graph_int> i_feat(n * i_len);
  offset[0] = 0;
  for (size_t i = 0; i < n; i++) {
    auto iter = const_store.find(keys[i]);
    if (iter != const_store.end()) {
      offset[i + 1] = offset[i] + iter->second.edge.size();
    } else {
      LF << "Key does not exist on PS in NodePull:" << keys[i];
    }
  }
  SArray<node_id> edge(offset[n]);
  for (size_t i = 0; i < n; i++) {
    auto node = const_store.find(keys[i])->second;
    std::copy(node.f_feat.begin(), node.f_feat.end(), &f_feat[i * f_len]);
    std::copy(node.i_feat.begin(), node.i_feat.end(), &i_feat[i * i_len]);
    std::copy(node.edge.begin(), node.edge.end(), &edge[offset[i]]);
  }
  get<0>(response) = f_feat;
  get<1>(response) = i_feat;
  get<2>(response) = edge;
  get<3>(response) = offset;
}

} // namespace ps
