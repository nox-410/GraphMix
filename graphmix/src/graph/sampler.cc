#include "graph/sampler.h"
#include "graph/graph_handle.h"

namespace ps {

sampleState makeSampleState(SamplerType type) {
  sampleState state;
  if (type == SamplerType::kRandomWalk) {
    state = std::make_shared<_randomWalkState>();
  } else if (type == SamplerType::kGraphSage) {
    state = std::make_shared<_graphSageState>();
  } else {
    state = std::make_shared<_sampleState>();
  }
  state->type = type;
  return state;
}

BaseSampler::BaseSampler(GraphHandle *handle) : handle_(handle->shared_from_this()) {}

void BaseSampler::sample_start() {
  auto func = [this] () {
    while (!killed_) {
      sampleState state = handle_->getRemote()->getSampleState(type());
      CHECK(state->type == type());
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

void LocalNodeSampler::sample_once(sampleState state) {
  NodePack node_pack;
  auto nodes = rd_.unique(batch_size_, handle_->nNodes());
  for (node_id node: nodes) {
    node_pack.emplace(node + handle_->offset(), handle_->getNode(node + handle_->offset()));
  }
  handle_->push(construct(node_pack), type());
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
  if (state->rw_round == 0) {
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
    size_t num_neighbor = state->recvNodes[node]->edge.size();
    if (num_neighbor == 0) continue;
    auto rd = rd_.randInt(num_neighbor);
    node_id nxt_node = state->recvNodes[node]->edge[rd];
    if (!state->recvNodes.count(nxt_node))
      state->query_nodes.emplace(nxt_node);
    new_frontier.emplace(nxt_node);
  }
  state->frontier = std::move(new_frontier);
  state->rw_round++;
  handle_->getRemote()->queryRemote(std::move(state));
}

void GraphSageSampler::sample_once(sampleState state_base) {
  auto state = std::static_pointer_cast<_graphSageState>(state_base);
  if (state->expand_round == depth_) {
    // if ready
    auto graph = construct(state->recvNodes);
    graph.extra.reserve(state->recvNodes.size());
    for (auto &node : state->recvNodes) {
      if (state->core_node.count(node.first)) {
        graph.extra.push_back(1);
      } else {
        graph.extra.push_back(0);
      }
    }
    handle_->push(graph, type());
    return;
  }
  if (state->expand_round == 0) {
    // Start a new sample
    auto nodes = rd_.unique(batch_size_, train_index.size());
    for (node_id node: nodes) {
      state->frontier.emplace(train_index[node]);
      state->recvNodes.emplace(train_index[node], handle_->getNode(train_index[node]));
    }
  }

  // select neighbor for frontier
  auto new_frontier = decltype(state->frontier)();
  state->query_nodes.clear();
  for (node_id node : state->frontier) {
    for (size_t i = 0; i < width_; i++) {
      size_t num_neighbor = state->recvNodes[node]->edge.size();
      if (num_neighbor == 0) continue;
      auto rd = rd_.randInt(num_neighbor);
      node_id nxt_node = state->recvNodes[node]->edge[rd];
      if (!state->recvNodes.count(nxt_node))
        state->query_nodes.emplace(nxt_node);
      new_frontier.emplace(nxt_node);
    }
  }

  if (state->expand_round == 0) state->core_node = std::move(state->frontier);
  state->frontier = std::move(new_frontier);
  state->expand_round++;
  handle_->getRemote()->queryRemote(std::move(state));
}

void GraphSageSampler::try_build_index(size_t index) {
  if (!train_index.empty()) return;
  CHECK(index >= 0 && index < handle_->iLen());
  for (node_id i = handle_->offset(); i < handle_->offset() + handle_->nNodes(); i++) {
    if (handle_->getNode(i)->i_feat[index] == 1) {
      train_index.push_back(i);
    }
  }
  CHECK(train_index.size() >= batch_size_)
    << "GraphSage index build fails train_index < batch_size " << train_index.size() << "<" << batch_size_;
}

std::vector<node_id> GraphSageSampler::train_index;

} // namespace ps
