#pragma once

#include "PSFunc.h"
#include "common/graph_type.h"

namespace ps {

template<> struct PSFData<NodePull> {
  using Request = tuple<
    Key // key
  >;
  using Response = tuple<
    SArray<graph_float>,
    SArray<graph_int>,
    SArray<node_id>
  >;
  static void _callback(const Response &response, NodeData &n) {
    n.f_feat = get<0>(response);
    n.i_feat = get<1>(response);
    n.edge = get<2>(response);
  }
};

template<> struct PSFData<NodePush> {
  using Request = tuple<
    Key, // key
    SArray<graph_float>,
    SArray<graph_int>,
    SArray<node_id>
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

} // namespace ps
