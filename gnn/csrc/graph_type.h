#pragma once

#include "common/sarray.h"
#include <unordered_map>

struct NodeData {
    SArray<float> f_feat;
    SArray<int> i_feat;
    SArray<size_t> edge;
};

typedef std::unordered_map<long, NodeData> NodePack;
