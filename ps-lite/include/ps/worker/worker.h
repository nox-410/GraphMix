#pragma once

#include "ps/worker/kvworker.h"
#include "common/binding.h"

using namespace ps;

class Worker {
public:
  static Worker& Get();
  // for data push&pull
  typedef uint64_t query_t;
  template<typename T>
  query_t pushData(py::array_t<long> indices, py::array_t<T> data, const py::array_t<long> lengths) {
    PYTHON_CHECK_ARRAY(indices);
    PYTHON_CHECK_ARRAY(lengths);
    return pushData_impl(indices.data(), indices.size(), data.mutable_data(), lengths.data());
  }
  template<typename T>
  query_t pullData(py::array_t<long> indices, py::array_t<T> data, const py::array_t<long> lengths) {
    PYTHON_CHECK_ARRAY(indices);
    PYTHON_CHECK_ARRAY(lengths);
    return pullData_impl(indices.data(), indices.size(), data.mutable_data(), lengths.data());
  }
  /*
    wait_data waits until a query success
  */
  void waitData(query_t query);
  static void initBinding(py::module &m);

private:
  Worker();
  Key mapWkeyToSkey(Key idx) {
    const std::vector<Range>& server_range = Postoffice::Get()->GetServerKeyRanges();
    int server = idx % server_range.size();
    Key k = server_range[server].begin() + idx;
    return k;
  }
  template<typename T>
  query_t pushData_impl(const long* indices, int index_size, T* data, const long* lengths) {
    py::gil_scoped_release release;
    data_mu.lock();
    query_t cur_query = next_query++;
    auto& timestamps = query2timestamp[cur_query];
    data_mu.unlock();
    auto cb = getCallBack<DensePush>();
    for (int i = 0; i < index_size; i++) {
      Key idx = (Key)indices[i];
      auto len = lengths[i];
      PSFData<DensePush>::Request request(mapWkeyToSkey(idx), SArray<char>(SArray<T>(data, len)));
      int ts = _kvworker.Request<DensePush>(request, cb);
      timestamps.push_back(ts);
      data += len;
    }
    return cur_query;
  }
  template<typename T>
  query_t pullData_impl(const long* indices, int index_size, T* data, const long* lengths) {
    py::gil_scoped_release release;
    data_mu.lock();
    query_t cur_query = next_query++;
    auto& timestamps = query2timestamp[cur_query];
    data_mu.unlock();

    for (int i = 0; i < index_size; i++) {
      Key idx = (Key)indices[i];
      auto len = lengths[i];
      auto cb = getCallBack<DensePull>(SArray<char>(SArray<T>(data, len)));
      PSFData<DensePull>::Request request(mapWkeyToSkey(idx));
      int ts = _kvworker.Request<DensePull>(request, cb);
      timestamps.push_back(ts);
      data += len;
    }
    return cur_query;
  }

  void waitTimestamp(int timestamp) { _kvworker.Wait(timestamp); }
  // used this hold to thread_pool return object
  std::unordered_map<query_t, std::vector<int>> query2timestamp;
  // data_pull & data_push query, increase 1 each call
  query_t next_query = 0;
  // protect query2timestamp and next_query
  std::mutex data_mu;
  KVWorker _kvworker;
};
