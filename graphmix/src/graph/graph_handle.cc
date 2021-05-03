#include "graph/graph_handle.h"

#include "ps/internal/postoffice.h"
#include "ps/internal/env.h"

namespace ps {

void GraphHandle::setReady() {
  std::unique_lock<std::mutex> lock(start_mu_);
  is_ready_ = true;
  cv_.notify_all();
}

void GraphHandle::waitReady() {
  std::unique_lock<std::mutex> lock(start_mu_);
  while (!is_ready_)
    cv_.wait(lock);
}

void GraphHandle::serve(const PSFData<NodePull>::Request& request, PSFData<NodePull>::Response& response) {
  //std::this_thread::sleep_for(std::chrono::milliseconds(100));
  waitReady();
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

void GraphHandle::serve(const PSFData<GraphPull>::Request& request, PSFData<GraphPull>::Response& response) {
  waitReady();
  std::vector<SamplerTag> valid_tag;
  auto request_tag = std::get<0>(request);
  // if no sampler specified, use the default priority
  if (request_tag.empty())
    for (auto &kvs : graph_queue_) request_tag.push_back(kvs.first);
  // filter out not exist tags;
  for (auto val : request_tag)
    if (graph_queue_.count(val)) valid_tag.push_back(val);
  if (valid_tag.empty()) {
    // request might not be correct
    std::get<5>(response) = kInvalidTag;
    std::get<6>(response) = static_cast<int>(SamplerType::kNumSamplerType);
    return;
  }

  GraphMiniBatch result;
  bool success = false;
  for (auto tag : valid_tag) {
    success = graph_queue_[tag]->TryPop(&result);
    if (success) break;
  }

  // If all samplers are not ready, use the one with lowest priority
  if (!success) {
    SamplerTag final_wait_tag = valid_tag.back();
    graph_queue_[final_wait_tag]->WaitAndPop(&result);
  }
  std::get<0>(response) = result.f_feat;
  std::get<1>(response) = result.i_feat;
  std::get<2>(response) = result.csr_i;
  std::get<3>(response) = result.csr_j;
  std::get<4>(response) = result.extra;
  std::get<5>(response) = result.tag;
  std::get<6>(response) = result.type;
}

void GraphHandle::serve(const PSFData<MetaPull>::Request& request, PSFData<MetaPull>::Response& response) {
  waitReady();
  std::string s;
  {
    py::gil_scoped_acquire acquire;
    s = py::str(dict_meta_);
  }
  SArray<char> meta(s.size());
  std::copy(s.data(), s.data() + s.size(), meta.data());
  std::get<0>(response) = meta;
}

void GraphHandle::initMeta(py::dict meta) {
  dict_meta_ = meta;
  // get graph data
  meta_.f_len = meta["float_feature"].cast<size_t>();
  meta_.i_len = meta["int_feature"].cast<size_t>();
  meta_.num_nodes = meta["node"].cast<size_t>();
  meta_.rank = Postoffice::Get()->my_rank();
  meta_.nrank = Postoffice::Get()->num_servers();

  // get partition data
  py::list offset = meta["partition"]["offset"];
  CHECK(int(offset.size()) == meta_.nrank);
  meta_.offset = std::vector<node_id>(meta_.nrank + 1);
  for (int i = 0; i < meta_.nrank; i++)
    meta_.offset[i] = offset[i].cast<node_id>();
  meta_.offset.back() = meta_.num_nodes;

  num_local_nodes_ = meta_.offset[meta_.rank + 1] - meta_.offset[meta_.rank];
  local_offset_ = meta_.offset[meta_.rank];
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
  for (node_id i = 0; i < num_local_nodes_; i++) {
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

void GraphHandle::stopSampling() {
  for (SamplerPTR& sampler : samplers_)
    sampler->kill();
  // Clean the queue so that sampelrs can stop
  GraphMiniBatch temp;
  for (auto& queue : graph_queue_) {
    while (queue.second->TryPop(&temp));
  }
  for (SamplerPTR& sampler : samplers_)
    sampler->join();
  samplers_.clear();
}

void GraphHandle::addSampler(SamplerType type, py::kwargs kwargs) {
  SamplerPTR sampler;
  SamplerTag tag = graph_queue_.size(); // this is the default tag, will be overwritten
  std::unordered_map<std::string, int> kvs;
  for (auto item : kwargs) {
    std::string key = std::string(py::str(item.first));
    if (key == "tag") {
      tag = py::hash(item.second);
      continue;
    }
    int value = item.second.cast<int>();
    kvs.emplace(key, value);
  }
  if (!graph_queue_.count(tag)) {
    auto ptr = std::make_unique<ThreadsafeBoundedQueue<GraphMiniBatch>>(kserverBufferSize);
    graph_queue_.emplace(tag, std::move(ptr));
    remote_->initQueue(tag);
  } else {
    LF << "Sampler tag should not be duplicated.";
  }
  int thread = kvs.count("thread") ? kvs["thread"] : 1;
  for (int i = 0; i < thread; i++) {
    switch (type)
    {
    case SamplerType::kLocalNode:
      CHECK(kvs.count("batch_size"));
      sampler = std::make_unique<LocalNodeSampler>(this, tag, kvs["batch_size"]);
      break;
    case SamplerType::kGlobalNode:
      CHECK(kvs.count("batch_size"));
      sampler = std::make_unique<GlobalNodeSampler>(this, tag, kvs["batch_size"]);
      break;
    case SamplerType::kRandomWalk:
      CHECK(kvs.count("rw_head"));
      CHECK(kvs.count("rw_length"));
      sampler = std::make_unique<RandomWalkSampler>(this, tag, kvs["rw_head"], kvs["rw_length"]);
      break;
    case SamplerType::kGraphSage:
      CHECK(kvs.count("batch_size"));
      CHECK(kvs.count("depth"));
      CHECK(kvs.count("width"));
      if (!kvs.count("index")) kvs["index"] = iLen() - 1;
      sampler = std::make_unique<GraphSageSampler>(this, tag, kvs["batch_size"], kvs["depth"], kvs["width"], kvs["index"]);
      break;
    default:
      LF << "Sampler Not Implemented";
    }
    sampler->sample_start();
    samplers_.push_back(std::move(sampler));
  }
}

void GraphHandle::push(const GraphMiniBatch& graph, SamplerTag tag) {
  graph_queue_[tag]->Push(graph);
}

void GraphHandle::initBinding(py::module& m) {
  py::class_<GraphHandle, std::shared_ptr<GraphHandle>>(m, "Graph handle", py::module_local())
    .def_property_readonly("meta", &GraphHandle::getMeta)
    .def("init_meta", &GraphHandle::initMeta)
    .def("init_data", &GraphHandle::initData)
    .def("init_cache", &GraphHandle::initCache)
    .def("get_perf", &GraphHandle::getProfileData)
    .def("is_ready", &GraphHandle::setReady)
    .def("add_sampler", &GraphHandle::addSampler)
    .def_static("barrier", []() {
      py::gil_scoped_release release;
      Postoffice::Get()->Barrier(0, kServerGroup);
    })
    .def_static("barrier_all", []() {
      py::gil_scoped_release release;
      Postoffice::Get()->Barrier(0, kWorkerGroup | kServerGroup);
    })
    .def_static("rank", []() { return Postoffice::Get()->my_rank(); })
    .def_static("num_worker", []() { return Postoffice::Get()->num_workers(); })
    .def_static("num_server", []() { return Postoffice::Get()->num_servers(); })
    .def_static("ip", []() { return Postoffice::Get()->van()->my_node().hostname; })
    .def_static("port", []() { return Postoffice::Get()->van()->my_node().port; });
}

void GraphHandle::createRemoteHandle(std::unique_ptr<KVApp<GraphHandle>>& app) {
  CHECK(app != nullptr);
  remote_ = std::make_unique<RemoteHandle>(app, this);
}

void GraphHandle::initCache(double ratio, cache::policy policy) {
  CHECK(num_local_nodes_ != 0) << "Data not ready.";
  ratio = std::min(ratio, 1.0);
  ratio = std::max(ratio, 0.0);
  size_t cache_size = size_t(ratio * (meta_.num_nodes - num_local_nodes_));
  if (cache_size > 0) {
    remote_->initCache(cache_size, policy);
  }
}

std::shared_ptr<GraphHandle> StartServer() {
  CHECK(Postoffice::Get()->is_server());
  static std::once_flag oc;
  static std::shared_ptr<GraphHandle> handle;
  std::call_once(oc, []() {
    auto ptr = std::make_unique<KVApp<GraphHandle>>();
    handle = ptr->getHandler();
    handle->createRemoteHandle(ptr);
    });
  return handle;
}

} // namespace ps
