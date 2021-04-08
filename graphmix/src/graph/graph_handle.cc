#include "graph/graph_handle.h"

#include "ps/internal/postoffice.h"
#include "ps/internal/env.h"

namespace ps {

void GraphHandle::serve(const PSFData<NodePull>::Request &request, PSFData<NodePull>::Response &response) {
  auto keys = get<0>(request);
  if (keys.empty()) return;
  size_t n = keys.size();
  SArray<size_t> offset(n + 1);
  SArray<graph_float> f_feat(n * meta_.f_len);
  SArray<graph_int> i_feat(n * meta_.i_len);
  offset[0] = 0;
  for (size_t i = 0; i < n; i++) {
    CHECK(keys[i] >= local_offset_ && keys[i] < local_offset_ + num_local_nodes_);
    auto node = nodes_[keys[i] - local_offset_];
    offset[i + 1] = offset[i] + node->edge.size();
  }
  SArray<node_id> edge(offset[n]);
  for (size_t i = 0; i < n; i++) {
    auto node = nodes_[keys[i] - local_offset_];
    std::copy(node->f_feat.begin(), node->f_feat.end(), &f_feat[i * meta_.f_len]);
    std::copy(node->i_feat.begin(), node->i_feat.end(), &i_feat[i * meta_.i_len]);
    std::copy(node->edge.begin(), node->edge.end(), &edge[offset[i]]);
  }
  get<0>(response) = f_feat;
  get<1>(response) = i_feat;
  get<2>(response) = edge;
  get<3>(response) = offset;
}

void GraphHandle::serve(const PSFData<GraphPull>::Request &request, PSFData<GraphPull>::Response &response) {
  CHECK(!graph_queue_.empty()) << "No sampler registered";
  GraphMiniBatch result;
  bool success = false;
  for (auto &queue : graph_queue_) {
    success = queue.second->try_pop(result);
    if (success) break;
  }
  // If all samplers are not ready, use the one with lowest priority
  if (!success) graph_queue_.rbegin()->second->pop(result);
  std::get<0>(response) = result.f_feat;
  std::get<1>(response) = result.i_feat;
  std::get<2>(response) = result.csr_i;
  std::get<3>(response) = result.csr_j;
}

void GraphHandle::initMeta(size_t f_len, size_t i_len, py::array_t<node_id> offset) {
  PYTHON_CHECK_ARRAY(offset);
  meta_.f_len = f_len;
  meta_.i_len = i_len;
  meta_.rank = Postoffice::Get()->my_rank();
  meta_.nrank = Postoffice::Get()->num_servers();
  meta_.offset = std::vector<node_id>(meta_.nrank + 1);
  assert(offset.size() == meta_.nrank + 1);
  for (int i = 0; i < meta_.nrank + 1; i++)
    meta_.offset[i] = offset.at(i);
  num_local_nodes_ = meta_.offset[meta_.rank + 1] - meta_.offset[meta_.rank];
  local_offset_ = meta_.offset[meta_.rank];
  meta_.num_nodes = meta_.offset[meta_.nrank];
  nodes_.reserve(num_local_nodes_);
}

void GraphHandle::initData(py::array_t<graph_float> f_feat, py::array_t<graph_int> i_feat, py::array_t<node_id> edges) {
  PYTHON_CHECK_ARRAY(f_feat);
  PYTHON_CHECK_ARRAY(i_feat);
  PYTHON_CHECK_ARRAY(edges);
  CHECK(f_feat.ndim() == 2 && f_feat.shape(0) == num_local_nodes_ && (size_t)f_feat.shape(1) == meta_.f_len);
  CHECK(i_feat.ndim() == 2 && i_feat.shape(0) == num_local_nodes_ && (size_t)i_feat.shape(1) == meta_.i_len);
  CHECK(edges.ndim() == 2 && edges.shape(0) == 2);
  size_t nedges = edges.shape(1);
  nodes_.resize(num_local_nodes_);
  for (node_id i = 0 ; i < num_local_nodes_; i++) {
    nodes_[i] = makeNodeData();
    nodes_[i]->f_feat.resize(fLen());
    nodes_[i]->i_feat.resize(iLen());
    std::copy(f_feat.data(i, 0), f_feat.data(i, 0) + fLen(), nodes_[i]->f_feat.data());
    std::copy(i_feat.data(i, 0), i_feat.data(i, 0) + iLen(), nodes_[i]->i_feat.data());
  }
  for (size_t i = 0; i < nedges; i++) {
    node_id u = edges.at(0, i), v = edges.at(1, i);
    CHECK(u >= local_offset_ && u < local_offset_ + num_local_nodes_);
    nodes_[u - local_offset_]->edge.push_back(v);
  }
}

int GraphHandle::getServer(node_id idx) {
  int server = 0;
  while (idx >= meta_.offset[server + 1]) server++;
  return server;
}

GraphHandle::~GraphHandle() {
  for (SamplerPTR sampler: samplers_)
    sampler->kill();
  for (SamplerPTR sampler: samplers_)
    sampler->join();
}

void GraphHandle::addLocalNodeSampler(size_t batch_size) {
  if (!graph_queue_.count(SamplerType::kLocalNode)) {
    auto ptr = std::make_unique<rigtorp::MPMCQueue<GraphMiniBatch>>(10);
    graph_queue_.emplace(SamplerType::kLocalNode, std::move(ptr));
  }
  auto sampler = std::make_shared<LocalNodeSampler>(this, batch_size);
  samplers_.push_back(sampler);
  sampler->sample_start();
}

void GraphHandle::addGlobalNodeSampler(size_t batch_size) {
  if (!graph_queue_.count(SamplerType::kGlobalNode)) {
    auto ptr = std::make_unique<rigtorp::MPMCQueue<GraphMiniBatch>>(10);
    graph_queue_.emplace(SamplerType::kGlobalNode, std::move(ptr));
  }
  auto sampler = std::make_shared<GlobalNodeSampler>(this, batch_size);
  samplers_.push_back(sampler);
  sampler->sample_start();
}

void GraphHandle::push(const GraphMiniBatch &graph, SamplerType type) {
  graph_queue_[type]->push(graph);
}

void GraphHandle::initBinding(py::module &m) {
  py::class_<GraphHandle, std::shared_ptr<GraphHandle>>(m, "Graph handle")
    .def("init_meta", &GraphHandle::initMeta)
    .def("init_data", &GraphHandle::initData)
    .def("init_cache", &GraphHandle::initCache)
    .def("add_local_node_sampler", &GraphHandle::addLocalNodeSampler)
    .def("add_global_node_sampler", &GraphHandle::addGlobalNodeSampler);
}

void GraphHandle::createRemoteHandle(std::shared_ptr<KVApp<GraphHandle>> app) {
  CHECK(app != nullptr);
  remote_ = std::make_unique<RemoteHandle>(app, this);
}

void GraphHandle::initCache(double ratio, cache::policy policy) {
  CHECK(num_local_nodes_ != 0);
  size_t cache_size = size_t(ratio * num_local_nodes_);
  if (cache_size > meta_.num_nodes - num_local_nodes_) {
    cache_size = meta_.num_nodes - num_local_nodes_;
    LG << "Server cache set limit too large " << cache_size;
  }
  remote_->initCache(cache_size, policy);
}

std::shared_ptr<GraphHandle> StartServer() {
  CHECK(Postoffice::Get()->is_server());
  static std::shared_ptr<KVApp<GraphHandle>> ptr;
  static std::once_flag oc;
  std::call_once(oc, []() {
    ptr = std::make_shared<KVApp<GraphHandle>>();
    ptr->getHandler()->createRemoteHandle(ptr);
  });
  return ptr->getHandler();
}

} // namespace ps
