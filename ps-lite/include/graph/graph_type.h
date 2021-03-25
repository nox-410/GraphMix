#pragma once

#include "common/sarray.h"
#include <unordered_map>

typedef long node_id;
typedef float graph_float;
typedef int graph_int;

struct NodeData {
    SArray<graph_float> f_feat;
    SArray<graph_int> i_feat;
    SArray<node_id> edge;
};

typedef std::unordered_map<node_id, NodeData> NodePack;

struct GraphMiniBatch {
    SArray<graph_float> f_feat;
    SArray<graph_int> i_feat;
    SArray<node_id> coo_u, coo_v;
};

struct GraphMetaData {
    int rank, nrank;
    size_t f_len, i_len;
    size_t num_nodes;
    std::vector<node_id> offset;
};
