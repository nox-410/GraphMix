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
