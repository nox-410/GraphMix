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

class GraphHandle;

class BaseSampler {
public:
  BaseSampler(GraphHandle *handle);
  void sample_start();
  void kill() { killed_ = true; }
  void join() { thread_.join(); }
  virtual ~BaseSampler() = default;
  virtual SamplerType type() = 0;
protected:
  std::shared_ptr<GraphHandle> handle_;
  GraphMiniBatch construct(const NodePack &node_pack);
  virtual GraphMiniBatch sample_once() = 0;
private:
  bool killed_ = false;
  std::thread thread_;
};

typedef std::unique_ptr<BaseSampler> SamplerPTR;

class LocalNodeSampler : public BaseSampler {
public:
  LocalNodeSampler(GraphHandle *handle, size_t batch_size);
  GraphMiniBatch sample_once();
  SamplerType type() { return SamplerType::kLocalNode; }
private:
  RandomIndexSelecter rd_;
  size_t batch_size_;
};

class GlobalNodeSampler : public BaseSampler {
public:
  GlobalNodeSampler(GraphHandle *handle, size_t batch_size);
  GraphMiniBatch sample_once();
  SamplerType type() { return SamplerType::kGlobalNode; }
private:
  RandomIndexSelecter rd_;
  size_t batch_size_;
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
