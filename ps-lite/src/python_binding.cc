#include "ps/worker/worker.h"
#include "ps/server/kvserver.h"
#include "common/binding.h"

void StartServer() {
  auto server = new KVServer(0);
  Postoffice::Get()->RegisterExitCallback([server]() { delete server; });
}

PYBIND11_MODULE(libc_PS, m) {
  m.doc() = "parameter server C++ plugin";

  m.def("rank", []() { return Postoffice::Get()->my_rank(); });
  m.def("nrank", []() { return Postoffice::Get()->num_workers(); });
  m.def("init", []() {
    if (Postoffice::Get()->van()) return;
    Postoffice::Get()->Start(0, nullptr, false);
    });
  m.def("finalize", []() { Postoffice::Get()->Finalize(0, true); });
  m.def("barrier", []() { Postoffice::Get()->Barrier(0, kWorkerGroup); });

  m.def("start_server", StartServer);

  Worker::initBinding(m);
} // PYBIND11_MODULE
