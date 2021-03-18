#include "ps/worker/worker.h"

Worker::Worker() : _kvworker(0, 0) {}

/*
    wait_data waits until a query success
*/
void Worker::waitData(query_t query) {
  data_mu.lock();
  auto iter = query2timestamp.find(query);
  if (iter == query2timestamp.end()) {
    data_mu.unlock();
    LG << "Wait on empty query " << query;
    return;
  } else {
    auto timestamps = std::move(iter->second);
    query2timestamp.erase(iter);
    data_mu.unlock();
    for (int t : timestamps) {
      waitTimestamp(t);
    }
  }
}

void Worker::initBinding(py::module& m) {
  py::class_<Worker>(m, "graph worker")
    .def("push_data_float", &Worker::pushData<float>)
    .def("pull_data_float", &Worker::pullData<float>)
    .def("push_data_int", &Worker::pushData<int>)
    .def("pull_data_int", &Worker::pullData<int>)
    .def("wait", &Worker::waitData);
  m.def("get_handle", Worker::Get, py::return_value_policy::reference);
}

Worker& Worker::Get() {
  static Worker w;
  return w;
}
