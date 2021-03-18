#include "ps/worker/worker.h"
#include "ps/ps.h"
#include "ps/server/kvserver.h"
#include "common/binding.h"

PYBIND11_MODULE(ps, m) {
  m.doc() = "parameter server C++ plugin";

  m.def("rank", []() { return Postoffice::Get()->my_rank(); });
  m.def("nrank", []() { return Postoffice::Get()->num_workers(); });
  m.def("init", []() { if (Postoffice::Get()->van()) return; Start(0); });
  m.def("finalize", []() { Finalize(0, true); });
  m.def("barrier", []() { Postoffice::Get()->Barrier(0,kWorkerGroup); });

} // PYBIND11_MODULE

void StartServer() {
    auto server = new KVServer(0);
    RegisterExitCallback([server]() { delete server; });
}
