#include "graph/graph_client.h"
#include "graph/graph_handle.h"
#include "common/binding.h"
#include "graph/graph.h"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(NodePack);

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
  py::class_<_NodeData, NodeData>(m, "NodeData")
    .def_property_readonly("f", [](NodeData &n){ return binding::vec_nocp(n->f_feat); } )
    .def_property_readonly("i", [](NodeData &n){ return binding::vec_nocp(n->i_feat); } )
    .def_property_readonly("e", [](NodeData &n){ return binding::vec_nocp(n->edge); } );

  py::enum_<cache::policy>(m, "cache")
    .value("LRU", cache::policy::LRU)
    .value("LFU", cache::policy::LFU)
    .value("LFUOpt", cache::policy::LFUOpt);

  GraphClient::initBinding(m);
  GraphHandle::initBinding(m);
  PyGraph::initBinding(m);
} // PYBIND11_MODULE
