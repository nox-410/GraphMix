#pragma once
#include "graph/graph_type.h"
#include "graph/random.h"
#include <thread>
#include <set>

namespace ps {

enum class SamplerType {
  kGraphSage = 0,
  kRandomWalk,
  kGlobalNode,
  kLocalNode,
  kNumSamplerType,
};

class _sampleState {
public:
  std::mutex mtx;
  int wait_num;
  std::unordered_set<node_id> query_nodes;
  NodePack recvNodes;
  SamplerType type;
  SamplerTag tag;
};

class _randomWalkState : public _sampleState {
public:
  std::unordered_set<node_id> frontier;
  size_t rw_round = 0;
};

class _graphSageState : public _sampleState {
public:
  std::unordered_set<node_id> frontier;
  std::unordered_set<node_id> core_node;
  std::set<std::pair<node_id, node_id>> coo;
  size_t expand_round = 0;
};

typedef std::shared_ptr<_sampleState> sampleState;

sampleState makeSampleState(SamplerType);

class GraphHandle;

class BaseSampler {
public:
  BaseSampler(GraphHandle *handle, SamplerTag tag);
  void sample_start();
  void join() { thread_.join(); }
  void kill() { killed_ = true; }
  virtual ~BaseSampler() = default;
  virtual SamplerType type() = 0;
  SamplerTag tag() { return tag_; }
protected:
  const std::shared_ptr<GraphHandle> handle_;
  GraphMiniBatch construct(const NodePack &node_pack);
  virtual void sample_once(sampleState) = 0;
private:
  std::thread thread_;
  bool killed_ = false;
  const SamplerTag tag_;
};

typedef std::unique_ptr<BaseSampler> SamplerPTR;

class LocalNodeSampler : public BaseSampler {
public:
  LocalNodeSampler(GraphHandle *handle, SamplerTag tag, size_t batch_size)
  : BaseSampler(handle, tag), batch_size_(batch_size) {}
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kLocalNode; }
private:
  RandomIndexSelecter rd_;
  const size_t batch_size_;
};

class GlobalNodeSampler : public BaseSampler {
public:
  GlobalNodeSampler(GraphHandle *handle, SamplerTag tag, size_t batch_size)
  : BaseSampler(handle, tag), batch_size_(batch_size) {}
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kGlobalNode; }
private:
  RandomIndexSelecter rd_;
  const size_t batch_size_;
};

class RandomWalkSampler : public BaseSampler {
public:
  RandomWalkSampler(GraphHandle *handle, SamplerTag tag, size_t rw_head, size_t rw_length)
   : BaseSampler(handle, tag), rw_head_(rw_head), rw_length_(rw_length) {}
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kRandomWalk; }
private:
  RandomIndexSelecter rd_;
  const size_t rw_head_;
  const size_t rw_length_;
};

class GraphSageSampler : public BaseSampler {
public:
  GraphSageSampler(GraphHandle *handle, SamplerTag tag, size_t batch_size, size_t depth, size_t width, ssize_t train_mask_index)
   : BaseSampler(handle, tag), batch_size_(batch_size), depth_(depth), width_(width) {
     try_build_index(train_mask_index);
    }
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kGraphSage; }
private:
  GraphMiniBatch SageConstruct(sampleState);
  RandomIndexSelecter rd_;
  const size_t batch_size_;
  const size_t depth_, width_;
  std::vector<node_id> train_index;
  void try_build_index(ssize_t index);
};

}
