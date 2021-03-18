#pragma once

#include "ps/worker/kvworker.h"
#include "common/binding.h"

using namespace ps;

class Worker {
public:
  static Worker& Get();
  // for data push&pull
  typedef uint64_t query_t;

  query_t pushData(py::array_t<long> indices, py::array_t<float> data, const py::array_t<long> lengths);
  query_t pullData(py::array_t<long> indices, py::array_t<float> data, const py::array_t<long> lengths);
  /*
    wait_data waits until a query success
  */
  void waitData(query_t query);
  static void initBinding(py::module &m);

private:
  Worker();
  query_t pushData_impl(const long* indices, int index_size, float* data, const long* lengths);
  query_t pullData_impl(const long* indices, int index_size, float* data, const long* lengths);
  void _pushData(Key idx, float *vals, int len, std::vector<int>& timestamp);
  void _pullData(Key idx, float *vals, int len, std::vector<int>& timestamp);
  void waitTimestamp(int timestamp) { _kvworker.Wait(timestamp); }
  // used this hold to thread_pool return object
  std::unordered_map<query_t, std::vector<int>> query2timestamp;
  // data_pull & data_push query, increase 1 each call
  query_t next_query = 0;
  // protect query2timestamp and next_query
  std::mutex data_mu;
  KVWorker _kvworker;
};
