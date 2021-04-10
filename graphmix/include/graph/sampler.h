#pragma once
#include "graph/graph_type.h"
#include "graph/random.h"
#include <thread>

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
  bool stopSampling = false;
};

class _randomWalkState : public _sampleState {
public:
  std::unordered_set<node_id> frontier;
  size_t rw_round = 0;
};

typedef std::shared_ptr<_sampleState> sampleState;

sampleState makeSampleState(SamplerType);

class GraphHandle;

class BaseSampler {
public:
  BaseSampler(GraphHandle *handle);
  void sample_start();
  void join() { thread_.join(); }
  virtual ~BaseSampler() = default;
  virtual SamplerType type() = 0;
protected:
  std::shared_ptr<GraphHandle> handle_;
  GraphMiniBatch construct(const NodePack &node_pack);
  virtual void sample_once(sampleState) = 0;
private:
  std::thread thread_;
};

typedef std::unique_ptr<BaseSampler> SamplerPTR;

class LocalNodeSampler : public BaseSampler {
public:
  LocalNodeSampler(GraphHandle *handle, size_t batch_size);
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kLocalNode; }
private:
  RandomIndexSelecter rd_;
  size_t batch_size_;
};

class GlobalNodeSampler : public BaseSampler {
public:
  GlobalNodeSampler(GraphHandle *handle, size_t batch_size);
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kGlobalNode; }
private:
  RandomIndexSelecter rd_;
  size_t batch_size_;
};

class RandomWalkSampler : public BaseSampler {
public:
  RandomWalkSampler(GraphHandle *handle, size_t rw_head, size_t rw_length)
   : BaseSampler(handle), rw_head_(rw_head), rw_length_(rw_length) {}
  void sample_once(sampleState);
  SamplerType type() { return SamplerType::kRandomWalk; }
private:
  RandomIndexSelecter rd_;
  size_t rw_head_;
  size_t rw_length_;
};

// class GraphSageSampler : public BaseSampler {
// public:
//   GraphSageSampler(GraphHandle *handle, size_t batch_size, int depth, int width);
//   GraphMiniBatch sample_once();
//   SamplerType type() { return SamplerType::kGraphSage; }
// private:
//   RandomIndexSelecter rd_;
//   size_t batch_size_;
// };

}
