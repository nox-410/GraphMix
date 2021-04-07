#pragma once

#include "ps/kvapp.h"
#include "graph/remote_handle.h"
#include "graph/sampler.h"
#include "common/binding.h"
#include "common/MPMCQueue.h"

namespace ps {

class GraphHandle : public std::enable_shared_from_this<GraphHandle> {
public:
  GraphHandle() : graph_queue_(10) {}
  ~GraphHandle();

  void serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response);
  void serve(const PSFData<GraphPull>::Request &request, PSFData<GraphPull>::Response &response);
  static void initBinding(py::module &m);
  void initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset);
  void initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges);
  void push(const GraphMiniBatch &graph) { graph_queue_.push(graph); }

  void addLocalNodeSampler(size_t batch_size);
  node_id nNodes() { return num_local_nodes_; }
  node_id offset() { return local_offset_; }
  size_t iLen() { return meta_.i_len; }
  size_t fLen() { return meta_.f_len; }
  NodeData getNode(node_id idx) { return nodes_[idx]; }
  void createRemoteHandle(std::shared_ptr<KVApp<GraphHandle>> app);
  void initCache(double ratio, cache::policy policy);
private:
// ---------------------- static node data -------------------------------------
  std::vector<NodeData> nodes_;
  GraphMetaData meta_;
  node_id num_local_nodes_;
  node_id local_offset_;
// ---------------------- sampler management -----------------------------------
  rigtorp::mpmc::Queue<GraphMiniBatch> graph_queue_;
  std::vector<SamplerPTR> samplers_;
// ---------------------- Remote data handle -----------------------------------
  std::unique_ptr<RemoteHandle> remote_;
};

std::shared_ptr<GraphHandle> StartServer();

} // namespace ps
