#pragma once
#include "graph/graph_type.h"
#include "graph/random.h"
#include <thread>

namespace ps {

enum class SamplerType {
  kLocalNode = 0,
  kGraphSage,
  kRandomWalk,
  kNumSamplerType,
};

class GraphHandle;

class BaseSampler {
public:
  BaseSampler(GraphHandle *handle);
  void sample_loop();
  void sample_start();
  void kill();
  void join();
  virtual GraphMiniBatch sample_once() = 0;
  virtual ~BaseSampler() {}
  virtual SamplerType type() = 0;
  GraphMiniBatch construct(const NodePack &node_pack);
  std::shared_ptr<GraphHandle> handle_;
private:
  bool killed_ = false;
  std::thread thread_;
};

typedef std::shared_ptr<BaseSampler> SamplerPTR;

class LocalNodeSampler : public BaseSampler {
public:
  LocalNodeSampler(GraphHandle *handle, size_t batch_size);
  GraphMiniBatch sample_once();
  SamplerType type() { return SamplerType::kLocalNode; }
private:
  RandomIndexSelecter rd_;
  size_t batch_size_;
};

}
