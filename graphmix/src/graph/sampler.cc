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

BaseSampler::BaseSampler(GraphHandle *handle, SamplerTag tag)
  : handle_(handle->shared_from_this()), tag_(tag) {}

void BaseSampler::sample_start() {
  auto func = [this] () {
    while (!killed_) {
      sampleState state = handle_->getRemote()->getSampleState(type(), tag());
      CHECK(state->type == type());
      CHECK(state->tag == tag());
      sample_once(std::move(state));
    }
  };
  thread_ = std::thread(func);
}

// construct a set of node into a graph
// unused edges are removed, used edges are reindexed
GraphMiniBatch BaseSampler::construct(const NodePack &node_pack) {
  GraphMiniBatch graph;
  graph.tag = tag();
  graph.type = static_cast<int>(type());
  size_t n = node_pack.size();
  graph.f_feat.resize(n * handle_->fLen());
  graph.i_feat.resize(n * handle_->iLen());
  graph.csr_i.resize(n + 1);
  std::unordered_map<node_id, node_id> idx_map;
  for (auto &node : node_pack) {
    int idx = idx_map.size();
    idx_map[node.first] = idx;
  }
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
  handle_->push(construct(node_pack), tag());
}

void GlobalNodeSampler::sample_once(sampleState state) {
  if (state->recvNodes.empty()) {
    // new samples
    auto nodes = rd_.unique(batch_size_, handle_->numGraphNodes());
    for (auto node : nodes) state->query_nodes.emplace(node);
    handle_->getRemote()->queryRemote(std::move(state));
  } else {
    handle_->push(construct(state->recvNodes), tag());
  }
}

void RandomWalkSampler::sample_once(sampleState state_base) {
  auto state = std::static_pointer_cast<_randomWalkState>(state_base);
  if (state->rw_round == rw_length_) {
    // if ready
    handle_->push(construct(state->recvNodes), tag());
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

GraphMiniBatch GraphSageSampler::SageConstruct(sampleState state_base) {
  auto state = std::static_pointer_cast<_graphSageState>(state_base);
  GraphMiniBatch graph;
  graph.tag = tag();
  graph.type = static_cast<int>(type());
  size_t n = state->recvNodes.size();
  graph.f_feat.resize(n * handle_->fLen());
  graph.i_feat.resize(n * handle_->iLen());
  std::unordered_map<node_id, node_id> idx_map;
  for (auto &node : state->recvNodes) {
    int idx = idx_map.size();
    idx_map[node.first] = idx;
  }
  for (auto &node : state->recvNodes) {
    node_id idx = idx_map[node.first];
    std::copy(node.second->f_feat.begin(), node.second->f_feat.end(), &graph.f_feat[idx * handle_->fLen()]);
    std::copy(node.second->i_feat.begin(), node.second->i_feat.end(), &graph.i_feat[idx * handle_->iLen()]);
  }
  graph.csr_i.reserve(state->coo.size());
  graph.csr_j.reserve(state->coo.size());
  for (auto &pair : state->coo) {
    graph.csr_i.push_back(idx_map[pair.first]);
    graph.csr_j.push_back(idx_map[pair.second]);
  }
  return graph;
}

void GraphSageSampler::sample_once(sampleState state_base) {
  auto state = std::static_pointer_cast<_graphSageState>(state_base);
  if (state->expand_round == depth_) {
    // if ready
    auto graph = SageConstruct(state);
    graph.extra.reserve(state->recvNodes.size());
    for (auto &node : state->recvNodes) {
      if (state->core_node.count(node.first)) {
        graph.extra.push_back(1);
      } else {
        graph.extra.push_back(0);
      }
    }
    handle_->push(graph, tag());
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
      state->coo.emplace(nxt_node, node);
      state->coo.emplace(node, nxt_node);
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

void GraphSageSampler::try_build_index(ssize_t index) {
  if (!train_index.empty()) return;
  if (index < 0) {
    train_index.reserve(handle_->nNodes());
    for (node_id i = handle_->offset(); i < handle_->offset() + handle_->nNodes(); i++)
      train_index.push_back(i);
  } else {
    CHECK(index >= 0 && size_t(index) < handle_->iLen());
    for (node_id i = handle_->offset(); i < handle_->offset() + handle_->nNodes(); i++) {
      if (handle_->getNode(i)->i_feat[index] == 1) {
        train_index.push_back(i);
      }
    }
  }
  CHECK(train_index.size() >= batch_size_)
    << "GraphSage index build fails train_index < batch_size " << train_index.size() << "<" << batch_size_;
  PS_VLOG(1) << "Create GraphSage Sampler at index " << index << " with train nodes " << train_index.size();
}

} // namespace ps
