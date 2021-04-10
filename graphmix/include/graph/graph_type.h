#pragma once

#include "common/sarray.h"
#include <unordered_map>

typedef long node_id;
typedef float graph_float;
typedef int graph_int;

struct _NodeData {
  std::vector<graph_float> f_feat;
  std::vector<graph_int> i_feat;
  std::vector<node_id> edge;
};

typedef std::shared_ptr<_NodeData> NodeData;

NodeData makeNodeData();

typedef std::unordered_map<node_id, NodeData> NodePack;

struct GraphMiniBatch {
  SArray<graph_float> f_feat;
  SArray<graph_int> i_feat;
  SArray<node_id> csr_i, csr_j;
  SArray<graph_int> extra;
};

struct GraphMetaData {
  int rank, nrank;
  size_t f_len, i_len;
  size_t num_nodes;
  std::vector<node_id> offset;
};
