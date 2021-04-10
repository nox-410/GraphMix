#include "graph/sampler.h"
#include "graph/graph_handle.h"

namespace ps {

sampleState makeSampleState(SamplerType type) {
  sampleState state;
  if (type==SamplerType::kRandomWalk) {
    state = std::make_shared<_randomWalkState>();
  } else {
    state = std::make_shared<_sampleState>();
  }
  state->type = type;
  return state;
}

BaseSampler::BaseSampler(GraphHandle *handle) : handle_(handle->shared_from_this()) {}

void BaseSampler::sample_start() {
  auto func = [this] () {
    while (true) {
      sampleState state = handle_->getRemote()->getSampleState(type());
      CHECK(state->type == type());
      if (state->stopSampling)
        return;
      sample_once(std::move(state));
    }
    handle_.reset();
  };
  thread_ = std::thread(func);
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

void LocalNodeSampler::sample_once(sampleState state) {
  NodePack node_pack;
  auto nodes = rd_.unique(batch_size_, handle_->nNodes());
  for (node_id node: nodes) {
    node_pack.emplace(node + handle_->offset(), handle_->getNode(node + handle_->offset()));
  }
  handle_->push(construct(node_pack), type());
}

GlobalNodeSampler::GlobalNodeSampler(GraphHandle *handle, size_t batch_size) : BaseSampler(handle) {
  batch_size_ = batch_size;
}

void GlobalNodeSampler::sample_once(sampleState state) {
  if (state->recvNodes.empty()) {
    // new samples
    auto nodes = rd_.unique(batch_size_, handle_->numGraphNodes());
    for (auto node : nodes) state->query_nodes.emplace(node);
    handle_->getRemote()->queryRemote(std::move(state));
  } else {
    handle_->push(construct(state->recvNodes), type());
  }
}

void RandomWalkSampler::sample_once(sampleState state_base) {
  auto state = std::static_pointer_cast<_randomWalkState>(state_base);
  if (state->rw_round == rw_length_) {
    // if ready
    handle_->push(construct(state->recvNodes), type());
    return;
  }
  if (state->frontier.empty()) {
    // Start a new sample
    auto nodes = rd_.unique(rw_head_, handle_->nNodes());
    for (node_id node: nodes) {
      state->frontier.emplace(node + handle_->offset());
      state->recvNodes.emplace(node + handle_->offset(), handle_->getNode(node + handle_->offset()));
    }
  }
  // select neighbor for frontier
  auto new_frontier = decltype(state->frontier)();
  state->query_nodes.clear();
  for (node_id node : state->frontier) {
    auto rd = rd_.randInt(state->recvNodes[node]->edge.size());
    node_id nxt_node = state->recvNodes[node]->edge[rd];
    if (!state->recvNodes.count(nxt_node))
      state->query_nodes.emplace(nxt_node);
    new_frontier.emplace(nxt_node);
  }
  state->frontier = std::move(new_frontier);
  state->rw_round++;
  handle_->getRemote()->queryRemote(std::move(state));
}

}
