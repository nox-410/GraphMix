#pragma once

#include "PSFunc.h"
#include "graph/graph_type.h"

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
  using Request = tuple<>;
  using Response = tuple<
    SArray<graph_float>, // float feature
    SArray<graph_int>, // int feature
    SArray<node_id>, // csr-format graph
    SArray<node_id>, // csr-foramt graph
    int, // tag
    SArray<graph_int> // extra data
  >;
};

} // namespace ps
