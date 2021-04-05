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
    SArray<graph_float>,
    SArray<graph_int>,
    SArray<node_id>,
    SArray<node_id>
  >;
  static void _callback(const Response &response, std::shared_ptr<GraphMiniBatch> tgt) {
    tgt->f_feat = std::get<0>(response);
    tgt->i_feat = std::get<1>(response);
    tgt->csr_i = std::get<2>(response);
    tgt->csr_j = std::get<3>(response);
  }
};

} // namespace ps
