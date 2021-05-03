#include "graph/graph.h"

#include "graph/random.h"

std::shared_ptr<PyGraph> makeGraph(py::array_t<node_id> edge_index, size_t num_nodes) {
  assert(edge_index.ndim() == 2 && edge_index.shape(0) == 2);
  size_t num_edges = edge_index.shape(1);
  SArray<node_id> edge_index_u(num_edges), edge_index_v(num_edges);
  memcpy(edge_index_u.data(), edge_index.data(), num_edges * sizeof(node_id));
  memcpy(edge_index_v.data(), edge_index.data(1), num_edges * sizeof(node_id));
  return std::make_shared<PyGraph>(edge_index_u, edge_index_v, num_nodes);
}

PyGraph::PyGraph(SArray<node_id> edge_index_u, SArray<node_id> edge_index_v, size_t num_nodes, std::string format) {
  edge_index_u_ = edge_index_u;
  edge_index_v_ = edge_index_v;
  nnodes_ = num_nodes;
  format_ = format;
  if (format != "coo" && format != "csr") {
    throw std::invalid_argument("PyGraph format should be coo or csr");
  }
  if (format == "coo") {
    CHECK_EQ(edge_index_u.size(), edge_index_v.size());
  } else {
    CHECK_EQ(edge_index_u.size(), nnodes_ + 1);
  }
}

py::array_t<graph_float> PyGraph::getFloatFeat() {
  py::array_t<graph_float> result = binding::svec_nocp(f_feat_);
  size_t feat_len = f_feat_.size() / nNodes();
  result.resize({nNodes(), feat_len});
  return result;
}

py::array_t<graph_int> PyGraph::getIntFeat() {
  py::array_t<graph_int> result = binding::svec_nocp(i_feat_);
  size_t feat_len = i_feat_.size() / nNodes();
  result.resize({nNodes(), feat_len});
  return result;
}

py::array_t<graph_int> PyGraph::getExtra() {
  py::array_t<graph_int> result = binding::svec_nocp(extra_);
  size_t extra_len = extra_.size() / nNodes();
  result.resize({nNodes(), extra_len});
  return result;
}

void PyGraph::setExtra(SArray<graph_int> extra) {
  if (extra.size() % nNodes() != 0)
    throw std::invalid_argument("extra size not met");
  extra_ = extra;
}

void PyGraph::setFeature(SArray<graph_float> f_feat, SArray<graph_int> i_feat) {
  if (f_feat.size() % nNodes() != 0 || i_feat.size() % nNodes() != 0)
    throw std::invalid_argument("feature length not met");
  f_feat_ = f_feat;
  i_feat_ = i_feat;
}

void PyGraph::addSelfLoop() {
  convert2coo();
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
  convert2coo();
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
  if (format_ == "csr") {
    for (size_t i = 0;i < nNodes(); i++) {
      deg[i] = edge_index_u_[i + 1] - edge_index_u_[i];
    }
  } else {
    for (size_t i = 0;i < nEdges(); i++) {
      deg[edge_index_u_[i]]++;
    }
  }
  return deg;
}

void PyGraph::convert2coo() {
  if (format_ == "coo") return;
  SArray<node_id> coo_u(nEdges());
  for (size_t i = 0; i < nNodes(); i++) {
    for (node_id j = edge_index_u_[i]; j < edge_index_u_[i + 1]; j++)
      coo_u[j] = i;
  }
  edge_index_u_ = coo_u;
  format_ = "coo";
}

void PyGraph::convert2csr() {
  if (format_ == "csr") return;
  SArray<node_id> indices(nEdges()), indptr(nNodes() + 1);
  auto deg = degree();
  indptr[0] = 0;
  for (size_t i = 1; i <= nNodes(); i++) {
    indptr[i] = deg[i-1] + indptr[i-1];
  }

  SArray<node_id> temp;
  temp.CopyFrom(indptr);
  for (size_t i = 0; i < nEdges(); i++) {
    node_id u = edge_index_u_[i], v = edge_index_v_[i];
    indices[temp[u]] = v;
    temp[u]++;
  }
  edge_index_u_ = indptr;
  edge_index_v_ = indices;
  format_ = "csr";
}

py::array_t<float> PyGraph::gcnNorm(bool use_original_gcn_norm) {
  auto deg = degree();
  py::array_t<float> py_norm(nEdges());
  auto norm = py_norm.mutable_data();
  {
    py::gil_scoped_release release;
    if (use_original_gcn_norm) {
      if (format_ == "csr") {
        node_id u = 0;
        for (size_t i = 0; i < nEdges(); i++) {
          node_id v = edge_index_v_[i];
          while (size_t(edge_index_u_[u + 1]) <= i) u++;
          norm[i] = sqrt(1.0f / (deg[v] * deg[u]));
        }
      } else {
        for (size_t i = 0; i < nEdges(); i++) {
          node_id v = edge_index_v_[i], u = edge_index_u_[i];
          norm[i] = sqrt(1.0f / (deg[v] * deg[u]));
        }
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
  convert2coo();
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

py::array_t<idx_t> PyGraph::PyPartition(int nparts) {
  auto x = partition(nparts, true);
  return binding::vec(x);
}

py::list PyGraph::part_graph(int nparts, bool balance_edge, bool random) {
  std::vector<idx_t> parts;
  if (random) {
    RandomIndexSelecter rd;
    parts.resize(nnodes_);
    for (size_t i = 0; i < nnodes_; i++) parts[i] = rd.randInt(nparts);
  } else
    parts = partition((idx_t)nparts, balance_edge);

  // compute new index and offset for each node
  std::vector<node_id> reindex(nNodes()), counting(nparts, 0), offset(nparts, 0);
  for (size_t i = 0;i < nNodes(); i++) {
    reindex[i] = counting[parts[i]];
    counting[parts[i]]++;
  }
  for (int i = 1; i < nparts; i++) offset[i] = offset[i-1] + counting[i-1];
  for (size_t i = 0;i < nNodes(); i++) reindex[i] += offset[parts[i]];
  offset.push_back(nNodes());

  // reindex edges
  std::vector<std::vector<node_id>> edges_u(nparts), edges_v(nparts);
  for (size_t i = 0; i < nEdges(); i++) {
    auto u = edge_index_u_[i], v = edge_index_v_[i];
    auto belong = parts[u];
    edges_u[belong].emplace_back(reindex[u]);
    edges_v[belong].emplace_back(reindex[v]);
  }

  std::vector<std::vector<node_id>> nodes(nparts);
  for (size_t i = 0; i < nNodes(); i++) nodes[parts[i]].emplace_back(i);

  py::list result;
  for (int i = 0; i < nparts; i++) {
    py::dict part_dict;
    part_dict["offset"] = offset[i];
    part_dict["orig_index"] = binding::vec(nodes[i]);
    part_dict["edges"] = std::make_tuple(binding::vec(edges_u[i]), binding::vec(edges_v[i]));
    result.append(part_dict);
  }
  return result;
}

void PyGraph::initBinding(py::module &m) {
  py::class_<PyGraph, std::shared_ptr<PyGraph>>(m, "Graph", py::module_local(), py::module_local())
    .def(py::init(&makeGraph), py::arg("edge_index"), py::arg("num_nodes"))
    .def_property_readonly("edge_index", &PyGraph::getEdgeIndex)
    .def_property_readonly("num_nodes", &PyGraph::nNodes)
    .def_property_readonly("num_edges", &PyGraph::nEdges)
    .def_property_readonly("format", &PyGraph::getFormat)
    .def_property_readonly("f_feat", &PyGraph::getFloatFeat)
    .def_property_readonly("i_feat", &PyGraph::getIntFeat)
    .def_property_readonly("type", &PyGraph::getType)
    .def_property_readonly("tag", &PyGraph::getTag)
    .def_property_readonly("extra", &PyGraph::getExtra)
    .def("part_graph", &PyGraph::part_graph, py::arg("nparts"), py::arg("balance_edge")=true, py::arg("random")=false)
    .def("partition", &PyGraph::PyPartition)
    .def("gcn_norm", &PyGraph::gcnNorm)
    .def("add_self_loop", &PyGraph::addSelfLoop)
    .def("remove_self_loop", &PyGraph::removeSelfLoop)
    .def("convert2csr", &PyGraph::convert2csr)
    .def("convert2coo", &PyGraph::convert2coo)
    .def("__repr__", [](PyGraph &g) {
          std::stringstream ss;
          ss << "<PyGraph Object, nodes=" << g.nNodes() << ",";
          ss << "edges=" << g.nEdges() << ">";
          return ss.str();
        });
}

NodeData makeNodeData() {
  return NodeData(new _NodeData());
}
