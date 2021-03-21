#pragma once

#include "PSFunc.h"
#include "common/graph_type.h"

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

template<> struct PSFData<NodePush> {
  using Request = tuple<
    node_id, // key
    SArray<graph_float>,
    SArray<graph_int>,
    SArray<node_id>
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

} // namespace ps
