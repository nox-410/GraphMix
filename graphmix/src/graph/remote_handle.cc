#include "graph/remote_handle.h"
#include "graph/graph_handle.h"

namespace ps {

RemoteHandle::RemoteHandle(std::unique_ptr<KVApp<GraphHandle>> &app, GraphHandle* handle) {
  CHECK(app);
  kvapp_ = std::move(app);
  for (int i = 0; i< static_cast<int>(SamplerType::kNumSamplerType); i++) {
    SamplerType tp = static_cast<SamplerType>(i);
    auto ptr = std::make_unique<ThreadsafeQueue<sampleState>>();
    recv_queue_.emplace(tp, std::move(ptr));
    on_flight_[i] = 0;
  }
  handle_ = handle->shared_from_this();
}

void RemoteHandle::initCache(size_t cache_size, cache::policy cache_policy) {
  CHECK(!cache_) << "Already have a cache";
  switch (cache_policy)
  {
  case cache::policy::LRU:
    cache_ = std::make_unique<cache::LRUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFU:
    cache_ = std::make_unique<cache::LFUCache<node_id, NodeData>>(cache_size);
    break;
  case cache::policy::LFUOpt:
    cache_ = std::make_unique<cache::LFUOptCache<node_id, NodeData>>(cache_size);
    break;
  default:
    std::runtime_error("Cache Policy Error");
  }
}

void RemoteHandle::defaultCallback(const sampleState &state) {
  CHECK(state);
  // cache insert
  if (cache_) {
    std::lock_guard lock(cache_mtx_);
    for (node_id node : state->query_nodes) {
      CHECK(state->recvNodes[node]);
      cache_->insert(node, state->recvNodes[node]);
    }
  }
  // If we are receiving too much, we will discard them.
  // This is not likely to happen
  recv_queue_[state->type]->Push(state);
  on_flight_[static_cast<int>(state->type)]--;
}

void RemoteHandle::partialCallback(sampleState state, SArray<node_id> pull_keys, const PSFData<NodePull>::Response &response) {
  CHECK(state);

  auto f_feat = std::get<0>(response);
  auto i_feat= std::get<1>(response);
  auto edge = std::get<2>(response);
  auto offset = std::get<3>(response);
  auto f_len = handle_->fLen(), i_len = handle_->iLen();
  CHECK_EQ(offset[offset.size() - 1], edge.size());
  for (size_t i = 0; i < pull_keys.size(); i++) {
    auto &node = state->recvNodes[pull_keys[i]];
    node->f_feat.resize(f_len);
    node->i_feat.resize(i_len);
    node->edge.resize(offset[i+1]-offset[i]);
    std::copy(&f_feat[i*f_len], &f_feat[(i + 1) * f_len], node->f_feat.data());
    std::copy(&i_feat[i*i_len], &i_feat[(i + 1) * i_len], node->i_feat.data());
    std::copy(&edge[offset[i]], &edge[offset[i+1]], node->edge.data());
  }
  state->mtx.lock();
  int wait_num = state->wait_num--;
  state->mtx.unlock();
  if (wait_num == 1) defaultCallback(state);
}

sampleState RemoteHandle::getSampleState(SamplerType type) {
  sampleState state;
  bool success = false;
  success = recv_queue_[type]->TryPop(&state);
  if (success) {
    return state;
  } else if (on_flight_[static_cast<int>(type)] > handle_->kserverBufferSize) {
    recv_queue_[type]->WaitAndPop(&state);
    return state;
  } else {
    state = makeSampleState(type);
    return state;
  }
}

void RemoteHandle::queryRemote(sampleState state) {
  CHECK(state != nullptr);
  CHECK(state->wait_num == 0);
  on_flight_[static_cast<int>(state->type)]++;
  filterNode(state);
  int nserver = Postoffice::Get()->num_servers();
  std::vector<SArray<node_id>> keys(nserver);
  for (node_id node : state->query_nodes) {
    keys[handle_->getServer(node)].push_back(node);
  }

  for (int server = 0; server < nserver; server++) {
    if (keys[server].size()) state->wait_num++;
  }
  // If no remote pull, push back immediately
  if (state->wait_num == 0) defaultCallback(state);
  for (int server = 0; server < nserver; server++) {
    if (keys[server].size() == 0) continue;
    PSFData<NodePull>::Request request(keys[server]);
    auto cb = std::bind(&RemoteHandle::partialCallback, this, state, keys[server], std::placeholders::_1);
    kvapp_->Request<NodePull>(request, cb, server);
  }
}

void RemoteHandle::filterNode(sampleState &state) {
  size_t local_cnt = 0;
  size_t num_query = state->query_nodes.size();
  state->recvNodes.reserve(state->recvNodes.size() + num_query);
  for (auto iter=state->query_nodes.begin(); iter != state->query_nodes.end();) {
    node_id node = *iter;
    if (handle_->isLocalNode(node)) {
      state->recvNodes[node] = handle_->getNode(node);
      local_cnt++;
    } else if (cache_)  {
      std::lock_guard lock(cache_mtx_);
      cache_->lookup(node, state->recvNodes[node]);
    }
    // create empty slot to avoid write conflict on callback
    if (state->recvNodes[node]) {
      iter = state->query_nodes.erase(iter);
    } else {
      state->recvNodes[node] = makeNodeData();
      iter++;
    }
  }
  // Handle profile data
  cache_mtx_.lock();
  total_cnt_ += num_query;
  nonlocal_cnt_ += num_query - local_cnt;
  cache_miss_cnt_ += state->query_nodes.size();
  cache_mtx_.unlock();
}

} // namespace ps
