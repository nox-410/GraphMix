#pragma once

#include "PSFunc.h"
#include "graph/graph_type.h"
#include "graph/sampler.h"

namespace ps {

template<> struct PSFData<NodePull> {
  using Request = tuple<
    SArray<node_id> // key
  >;
  using Response = tuple<
    SArray<graph_float>,
    SArray<graph_int>,
    SArray<node_id>,
    SArray<size_t>
  >;
};

template<> struct PSFData<GraphPull> {
  using Request = tuple<
    SArray<SamplerTag> // desired sampler
  >;
  using Response = tuple<
    SArray<graph_float>, // float feature
    SArray<graph_int>, // int feature
    SArray<node_id>, // csr-format graph
    SArray<node_id>, // csr-foramt graph
    SArray<graph_int>, // extra data
    SamplerTag, //sampler tag
    int // sampler type
  >;
};

template<> struct PSFData<MetaPull> {
  using Request = tuple<>;
  using Response = tuple<SArray<char>>;
};

} // namespace ps
