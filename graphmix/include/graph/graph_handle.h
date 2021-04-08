#pragma once

#include "ps/kvapp.h"
#include "graph/remote_handle.h"
#include "graph/sampler.h"
#include "common/binding.h"
#include "common/MPMCQueue.h"

#include <map>

namespace ps {

class GraphHandle : public std::enable_shared_from_this<GraphHandle> {
public:
  GraphHandle() {}
  ~GraphHandle();

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response);
  void serve(const PSFData<GraphPull>::Request &request, PSFData<GraphPull>::Response &response);
  static void initBinding(py::module &m);
  void initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset);
  void initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges);
  void push(const GraphMiniBatch &graph, SamplerType type);

  node_id nNodes() { return num_local_nodes_; }
  node_id offset() { return local_offset_; }
  node_id numGraphNodes() { return meta_.num_nodes; }
  size_t iLen() { return meta_.i_len; }
  size_t fLen() { return meta_.f_len; }
  NodeData getNode(node_id idx) { return nodes_[idx - local_offset_]; }
  bool isLocalNode(node_id idx) { return idx >= local_offset_ && idx < local_offset_ + num_local_nodes_; }
  int getServer(node_id idx);
  void createRemoteHandle(std::shared_ptr<KVApp<GraphHandle>> app);
  void initCache(double ratio, cache::policy policy);
  auto& getRemote() { return remote_; }

  void addLocalNodeSampler(size_t batch_size);
  void addGlobalNodeSampler(size_t batch_size);
  py::tuple getProfileData() {
    return py::make_tuple(remote_->cache_miss_cnt_, remote_->nonlocal_cnt_, remote_->total_cnt_);
  }
private:
// ---------------------- static node data -------------------------------------
  std::vector<NodeData> nodes_;
  GraphMetaData meta_;
  node_id num_local_nodes_;
  node_id local_offset_;
// ---------------------- sampler management -----------------------------------
  std::map<SamplerType, std::unique_ptr<rigtorp::MPMCQueue<GraphMiniBatch>>> graph_queue_;
  std::vector<SamplerPTR> samplers_;
// ---------------------- Remote data handle -----------------------------------
  std::unique_ptr<RemoteHandle> remote_;
};

std::shared_ptr<GraphHandle> StartServer();

} // namespace ps
