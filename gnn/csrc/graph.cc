#include "graph.h"

PyGraph makeGraph(py::array_t<node_id> edge_index, size_t num_nodes) {
  assert(edge_index.ndim() == 2 && edge_index.shape(0) == 2);
  size_t num_edges = edge_index.shape(1);
  SArray<node_id> edge_index_u(num_edges), edge_index_v(num_edges);
  memcpy(edge_index_u.data(), edge_index.data(), num_edges * sizeof(node_id));
  memcpy(edge_index_v.data(), edge_index.data(1), num_edges * sizeof(node_id));
  return PyGraph(edge_index_u, edge_index_v, num_nodes);
}

PyGraph::PyGraph(SArray<node_id> edge_index_u, SArray<node_id> edge_index_v, size_t num_nodes) {
  edge_index_u_ = edge_index_u;
  edge_index_v_ = edge_index_v;
  nnodes_ = num_nodes;
}

void PyGraph::addSelfLoop() {
  std::vector<bool> check(nNodes(), false);
  for (size_t i = 0;i < nEdges(); i++) {
    if (edge_index_u_[i] == edge_index_v_[i]) {
      check[edge_index_u_[i]] = true;
    }
  }
  edge_index_u_.reserve(nEdges() + nNodes());
  edge_index_v_.reserve(nEdges() + nNodes());
  for (size_t i = 0;i < nNodes(); i++) {
    if (!check[i]) {
      edge_index_u_.push_back(i);
      edge_index_v_.push_back(i);
    }
  }
}

void PyGraph::removeSelfLoop() {
  SArray<node_id> u, v;
  u.reserve(nEdges());
  v.reserve(nEdges());
  for (size_t i = 0;i < nEdges(); i++) {
    if (edge_index_u_[i] != edge_index_v_[i]) {
      u.push_back(edge_index_u_[i]);
      v.push_back(edge_index_v_[i]);
    }
  }
  edge_index_u_ = u;
  edge_index_v_ = v;
}

double PyGraph::denseEfficiency() {
  return double(nEdges()) / (nNodes() * nNodes());
}

std::vector<long> PyGraph::degree() {
  std::vector<long> deg(nNodes(), 0);
  for (size_t i = 0;i < nEdges(); i++) {
    deg[edge_index_u_[i]]++;
  }
  return deg;
}

py::array_t<float> PyGraph::gcnNorm(bool use_original_gcn_norm) {
  auto deg = degree();
  py::array_t<float> py_norm(nEdges());
  auto norm = py_norm.mutable_data();
  {
    py::gil_scoped_release release;
    if (use_original_gcn_norm) {
      for (size_t i = 0;i < nEdges(); i++) {
        node_id v = edge_index_v_[i], u = edge_index_u_[i];
        norm[i] = sqrt(1.0f / (deg[v] * deg[u]));
      }
    } else {
      for (size_t i = 0;i < nEdges(); i++) {
        node_id v = edge_index_v_[i];
        norm[i] = 1.0f / deg[v];
      }
    }
  }

  return py_norm;
}

std::vector<idx_t> PyGraph::partition(idx_t nparts, bool balance_edge) {
  assert(nparts >= 1);
  if (nparts == 1) {
    return std::vector<idx_t>(nNodes(), 0);
  }
  std::vector<idx_t> indices(nEdges()), indptr(nNodes() + 1);
  auto deg = degree();
  indptr[0] = 0;
  for (size_t i = 1; i <= nNodes(); i++) {
    indptr[i] = deg[i-1] + indptr[i-1];
  }
  auto temp = indptr;
  for (size_t i = 0; i < nEdges(); i++) {
    node_id u = edge_index_u_[i], v = edge_index_v_[i];
    indices[temp[u]] = v;
    temp[u]++;
  }
  //Start metis API
  idx_t num_nodes = nNodes();
  idx_t ncon = 1, edgecut;
  std::vector<idx_t> parts(nNodes());

  auto partition_function = METIS_PartGraphKway;
  if (nparts > 8) {
    partition_function = METIS_PartGraphRecursive;
  }
  // Decide whether to balance edge
  // two constraint, number of nodes and nodes' degree
  std::vector<idx_t> vwgt;
  idx_t *vwgt_data = NULL;
  if (balance_edge) {
    ncon = 2;
    vwgt.resize(2 * nNodes());
    for (size_t i = 0;i < nNodes(); i++) {
      vwgt[i * ncon] = deg[i];
      vwgt[i * ncon + 1] = 1;
    }
    vwgt_data = vwgt.data();
  }
  int info = partition_function(
    &num_nodes, /* number of nodes */
    &ncon,
    indptr.data(),
    indices.data(),
    vwgt_data,    /* weight of nodes */
    NULL,    /* The size of the vertices for computing the total communication volume */
    NULL,    /* weight of edges */
    &nparts, /* num parts */
    NULL,    /* the desired weight for each partition and constraint */
    NULL,    /* an array of size ncon that specifies the allowed load imbalance tolerance for each constraint */
    NULL,    /* options */
    &edgecut,  /* store number of edge cut */
    parts.data() /* store partition result */
  );
  switch (info) {
  case METIS_OK:
    break;
  case METIS_ERROR_INPUT:
    printf("Metis error input");
    break;
  case METIS_ERROR_MEMORY:
    printf("Metis error memory");
    break;
  case METIS_ERROR:
  default:
    printf("Metis error");
    break;
  }
  assert(info == METIS_OK);
  return parts;
}

py::array_t<idx_t> PyGraph::PyPartition(idx_t nparts) {
  auto x = partition(nparts, true);
  return bind::vec(x);
}

py::list PyGraph::part_graph(int nparts, bool balance_edge) {
  auto parts = partition((idx_t)nparts, balance_edge);
  std::vector<std::vector<node_id>> edges_u(nparts), edges_v(nparts);
  std::vector<std::vector<node_id>> nodes(nparts);
  for (size_t i = 0; i < nEdges(); i++) {
    auto u = edge_index_u_[i], v = edge_index_v_[i];
    auto belong = parts[u];
    edges_u[belong].emplace_back(u);
    edges_v[belong].emplace_back(v);
  }
  for (size_t i = 0; i < nNodes(); i++) {
    auto belong = parts[i];
    nodes[belong].emplace_back(i);
  }
  py::list result;
  for (int i = 0; i < nparts; i++) {
    auto tup = std::make_tuple(bind::vec(nodes[i]), bind::vec(edges_u[i]), bind::vec(edges_v[i]));
    result.append(tup);
  }
  return result;
}
