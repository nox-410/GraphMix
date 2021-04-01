#include "ps/kvapp.h"
#include "graph/graph_client.h"
#include "graph/graph_handle.h"
#include "common/binding.h"
#include "graph/graph.h"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(NodePack);

std::shared_ptr<GraphHandle> StartServer() {
  auto server = new KVApp<GraphHandle>();
  Postoffice::Get()->RegisterExitCallback([server]() { delete server; });
  return server->getHandler();
}

PYBIND11_MODULE(libc_graphmix, m) {
  m.doc() = "graphmix graph server C++ backend";

  m.def("init", []() {
    if (Postoffice::Get()->van()) return;
    Postoffice::Get()->Start(0, nullptr, false);
    });
  m.def("finalize", []() { Postoffice::Get()->Finalize(0, true); });

  m.def("ip", []() { return Postoffice::Get()->van()->my_node().hostname; });
  m.def("port", []() { return Postoffice::Get()->van()->my_node().port; });
  m.def("rank", []() { return Postoffice::Get()->my_rank(); });
  m.def("num_worker", []() { return Postoffice::Get()->num_workers(); });
  m.def("num_server", []() { return Postoffice::Get()->num_servers(); });
  m.def("barrier", []() { Postoffice::Get()->Barrier(0, kWorkerGroup); });
  m.def("barrier_all", []() { Postoffice::Get()->Barrier(0, kWorkerGroup | kServerGroup); });

  m.def("start_server", StartServer);

  py::bind_map<NodePack>(m, "NodePack");
  py::class_<NodeData>(m, "NodeData")
    .def_property_readonly("f", [](NodeData &n){ return bind::svec_nocp(n.f_feat); } )
    .def_property_readonly("i", [](NodeData &n){ return bind::svec_nocp(n.i_feat); } )
    .def_property_readonly("e", [](NodeData &n){ return bind::svec_nocp(n.edge); } );

  py::class_<PyGraph, std::shared_ptr<PyGraph>>(m, "Graph")
    .def(py::init(&makeGraph), py::arg("edge_index"), py::arg("num_nodes"))
    .def_property_readonly("edge_index", &PyGraph::getEdgeIndex)
    .def_property_readonly("num_nodes", &PyGraph::nNodes)
    .def_property_readonly("num_edges", &PyGraph::nEdges)
    .def("part_graph", &PyGraph::part_graph, py::arg("nparts"), py::arg("balance_edge")=true)
    .def("partition", &PyGraph::PyPartition)
    .def("gcn_norm", &PyGraph::gcnNorm)
    .def("add_self_loop", &PyGraph::addSelfLoop)
    .def("remove_self_loop", &PyGraph::removeSelfLoop)
    .def("__repr__", [](PyGraph &g) {
          std::stringstream ss;
          ss << "<PyGraph Object, nodes=" << g.nNodes() << ",";
          ss << "edges=" << g.nEdges() << ">";
          return ss.str();
        });

  GraphClient::initBinding(m);
  GraphHandle::initBinding(m);
} // PYBIND11_MODULE
