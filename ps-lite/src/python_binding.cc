#include "ps/kvserver.h"
#include "graph/graph_client.h"
#include "graph/graph_handle.h"
#include "common/binding.h"

GraphHandle& StartServer() {
  auto server = new KVServer(0, GraphHandle());
  Postoffice::Get()->RegisterExitCallback([server]() { delete server; });
  return server->handler;
}

PYBIND11_MODULE(libc_PS, m) {
  m.doc() = "parameter server C++ plugin";

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

  m.def("start_server", StartServer, py::return_value_policy::reference);

  GraphClient::initBinding(m);
  GraphHandle::initBinding(m);
} // PYBIND11_MODULE
