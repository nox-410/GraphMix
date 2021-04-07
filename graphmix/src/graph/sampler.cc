#include "graph/sampler.h"
#include "graph/graph_handle.h"

namespace ps {

BaseSampler::BaseSampler(GraphHandle *handle) : handle_(handle->shared_from_this()) {}

void BaseSampler::sample_loop() {
  while (!killed_) {
    GraphMiniBatch graph = sample_once();
    handle_->push(graph);
  }
  handle_.reset();
}

void BaseSampler::sample_start() {
  thread_ = std::thread(&BaseSampler::sample_loop, this);
}

void BaseSampler::kill() {
  killed_ = true;
  thread_.join();
}

void BaseSampler::join() {
  thread_.join();
}

// construct a set of node into a graph
// unused edges are removed, used edges are reindexed
GraphMiniBatch BaseSampler::construct(const NodePack &node_pack) {
  GraphMiniBatch graph;
  size_t n = node_pack.size();
  graph.f_feat.resize(n * handle_->fLen());
  graph.i_feat.resize(n * handle_->iLen());
  graph.csr_i.resize(n + 1);
  std::unordered_map<node_id, node_id> idx_map;
  for (auto &node : node_pack)
    idx_map[node.first] = idx_map.size();
  for (auto &node : node_pack) {
    node_id idx = idx_map[node.first];
    for (node_id neighbor : node.second->edge) {
      if (idx_map.count(neighbor)) {
        graph.csr_j.push_back(idx_map[neighbor]);
      }
    }
    graph.csr_i[idx + 1] = graph.csr_j.size();
    std::copy(node.second->f_feat.begin(), node.second->f_feat.end(), &graph.f_feat[idx * handle_->fLen()]);
    std::copy(node.second->i_feat.begin(), node.second->i_feat.end(), &graph.i_feat[idx * handle_->iLen()]);
  }
  return graph;
}

LocalNodeSampler::LocalNodeSampler(GraphHandle *handle, size_t batch_size) : BaseSampler(handle) {
  batch_size_ = batch_size;
}

GraphMiniBatch LocalNodeSampler::sample_once() {
  NodePack node_pack;
  auto nodes = rd_.unique(batch_size_, handle_->nNodes());
  for (node_id node: nodes) {
    node_pack.emplace(node + handle_->offset(), handle_->getNode(node));
  }
  return construct(node_pack);
}

}
