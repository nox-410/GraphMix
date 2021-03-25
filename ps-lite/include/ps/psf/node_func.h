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

} // namespace ps
