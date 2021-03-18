#pragma once

#include "ps/psf/PSFunc.h"

#include "common/thread_safe_hash_map.h"
#include "param.h"
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
    size_t len = get<1>(request);
    SArray<float> &pull_vals = get<0>(response);

    auto iter = const_store.find(k);
    if (iter != const_store.end()) {
      auto &value_set_ = *iter->second;
      size_t data_size = value_set_.size();
      CHECK_EQ(len, data_size) << " size mismatch in DensePull " << k << " " << len << " " << data_size;
      pull_vals.resize(data_size);
      auto read_lock = value_set_.read_guard();
      std::copy(value_set_.begin(), value_set_.end(), pull_vals.begin());
    } else {
      LG << "Key does not exist on PS in DensePull" << k;
    }
  }

  void serve(const PSFData<DensePush>::Request &request, PSFData<DensePush>::Response &response) {
    Key k = get<0>(request);
    size_t len = get<1>(request);
    SArray<float> vals = get<2>(request);

    if (const_store.find(k) == const_store.end()) {
      store[k] = std::make_shared<Param<float>>(len);
    }
    auto iter = const_store.find(k);
    if (iter != const_store.end()) {
      CHECK_EQ(len, iter->second->size()) << k << " " << len <<" " << iter->second->size() <<" size mismatch in DensePush";
      // write, discard const qualifier
      auto &value_set_ = *const_cast<typename tmap::mapped_type&>(iter->second);
      auto write_lock = value_set_.write_guard();
      #pragma omp parallel for num_threads(4)
      for (size_t j = 0; j < value_set_.size(); j++)
        value_set_[j] += vals[j];
    } else {
      LG << "Key does not exist on PS in DensePull" << k;
    }
  }

private:
  typedef threadsafe_unordered_map<Key, std::shared_ptr<Param<float>>> tmap;
  tmap store;
  const tmap& const_store = store; // const reference to force compiler to use read lock
};

} // namespace ps
