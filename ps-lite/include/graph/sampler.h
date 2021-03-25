#pragma once
#include "graph/graph_type.h"
#include <thread>

namespace ps {

class GraphHandle;

class BaseSampler {
public:
  BaseSampler(GraphHandle &handle);
  void sample_loop();
  void sample_start();
  virtual GraphMiniBatch sample_once();
  virtual ~BaseSampler() {}
  GraphHandle &handle_;
private:
  std::thread thread_;
};

typedef std::shared_ptr<BaseSampler> SamplerPTR;

class LocalNodeSampler : public BaseSampler {
public:
  LocalNodeSampler(GraphHandle &handle, size_t batch_size);
  GraphMiniBatch sample_once();
private:
  size_t batch_size_;
};

}
