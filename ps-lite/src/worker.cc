#include "ps/worker/worker.h"

Worker::Worker() : _kvworker(0, 0) {}

Worker::query_t Worker::pushData(const long* indices, int index_size, float* data, const long* lengths) {
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();

  for (int i = 0; i < index_size; i++) {
    Key idx = (Key)indices[i];
    auto len = lengths[i];
    _pushData(idx, data, len, timestamps);
    data += len;
  }
  return cur_query;
}

// this is almost the same as push_data
Worker::query_t Worker::pullData(const long* indices, int index_size, float* data, const long* lengths) {
  data_mu.lock();
  query_t cur_query = next_query++;
  auto& timestamps = query2timestamp[cur_query];
  data_mu.unlock();

  for (int i = 0; i < index_size; i++) {
    Key idx = (Key)indices[i];
    auto len = lengths[i];
    _pullData(idx, data, len, timestamps);
    data += len;
  }
  return cur_query;
}

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

Key mapWkeyToSkey(Key idx) {
  const std::vector<Range>& server_range = Postoffice::Get()->GetServerKeyRanges();
  int server = idx % server_range.size();
  Key k = server_range[server].end() - idx - 1;
  return k;
}

void Worker::_pushData(Key idx, float* vals, int len, std::vector<int>& timestamp) {
  auto cb = getCallBack<DensePush>();
  PSFData<DensePush>::Request request(mapWkeyToSkey(idx), len, SArray<float>(vals, len));
  int ts = _kvworker.Request<DensePush>(request, cb);
  timestamp.push_back(ts);
}

void Worker::_pullData(Key idx, float* vals, int len, std::vector<int>& timestamp) {
  auto cb = getCallBack<DensePull>(SArray<float>(vals, len));
  PSFData<DensePull>::Request request(mapWkeyToSkey(idx), len);
  int ts = _kvworker.Request<DensePull>(request, cb);
  timestamp.push_back(ts);
}

Worker worker;
