#include "graph/sampler.h"
#include "graph/graph_handle.h"

namespace ps {

BaseSampler::BaseSampler(GraphHandle &handle) : handle_(handle) {}

void BaseSampler::sample_loop() {
  while (true) {
    GraphMiniBatch graph = sample_once();
    handle_.push(graph);
  }
}

void BaseSampler::sample_start() {
  thread_ = std::thread(&BaseSampler::sample_loop, this);
}

LocalNodeSampler::LocalNodeSampler(GraphHandle &handle, size_t batch_size) : BaseSampler(handle) {
  batch_size_ = batch_size;
}

GraphMiniBatch LocalNodeSampler::sample_once() {
  GraphMiniBatch graph;
  return graph;
}

}
