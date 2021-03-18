#pragma once

#include "ps/ps.h"
#include "ps/worker/PSAgent.h"

using namespace ps;

class Worker {
public:
  Worker();

  // for data push&pull
  typedef uint64_t query_t;
  /*
    for each indice, call PSAgent::PushData to launch a thread
    hold the return handle in the global map
    immediately return
    user should guaruntee value unchanged until waitdata
    returns:
      an query_t which is a long
      use waitdata(query_t) to wait for its success
  */
  query_t push_data(const long *indices, int index_size, float *data, const long *lengths);
  // this is almost the same as push_data
  query_t pull_data(const long *indices, int index_size, float *data, const long *lengths);
  /*
    wait_data waits until a query success
  */
  void wait_data(query_t query);

private:
  // used this hold to thread_pool return object
  std::unordered_map<query_t, std::vector<int>> query2timestamp;
  // data_pull & data_push query, increase 1 each call
  query_t next_query = 0;
  // protect query2timestamp and next_query
  std::mutex data_mu;

  int _thread_num = 3;
};

extern Worker worker;
