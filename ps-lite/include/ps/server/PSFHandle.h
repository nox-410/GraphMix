#pragma once

#include "ps/psf/PSFunc.h"

#include "common/thread_safe_hash_map.h"
#include <algorithm>
#include <utility>
#include <mutex>
#include <omp.h>
#include <random>
#include <fstream>

namespace ps {
/**
 * \brief used in ML part for sparse/dense pull, push.
 *        keys is used for the key of one partition.
 *        lens is used as the offset of the keys.
 *        vals is vals.
 *        One key (two keys for binary op) per request in Athena.
 *        Is it ok in a lock-free manner? By @Zhipeng
 */

class KVServerMatrixHandle {
public:
  KVServerMatrixHandle() {}
  KVServerMatrixHandle(const KVServerMatrixHandle& handle) {}

  void serve(const PSFData<DensePull>::Request &request, PSFData<DensePull>::Response &response) {
    Key k = get<0>(request);
    SArray<char> &pull_vals = get<0>(response);
    auto iter = const_store.find(k);
    if (iter != const_store.end()) {
      pull_vals.CopyFrom(iter->second);
    } else {
      LF << "Key does not exist on PS in DensePull:" << k;
    }
  }

  void serve(const PSFData<DensePush>::Request &request, PSFData<DensePush>::Response &response) {
    Key k = get<0>(request);
    SArray<char> vals = get<1>(request);
    if (const_store.find(k) == const_store.end()) {
      store[k] = vals;
    } else {
      LF << "Key already exist on PS in DensePush:" << k;
    }
  }

private:
  typedef threadsafe_unordered_map<Key, SArray<char>> tmap;
  tmap store;
  const tmap& const_store = store; // const reference to force compiler to use read lock
};

} // namespace ps
