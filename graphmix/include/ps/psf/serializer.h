#pragma once

#include "common/sarray.h"

#include <tuple>
#include <vector>
using std::tuple;
using std::vector;

namespace ps {

// decide whether a data is scalar type or SArray
// isScalar<int>::value -> true
template<typename T> class isScalar {
public:
  constexpr static bool value = std::is_integral<T>::value || std::is_floating_point<T>::value;
};

// Helper class to serialize Tuples recursively
template<typename Tuple, int N>
class tupleSerializer {
public:
  // encode a tuple from back to front
  static void encode(const Tuple &tup, vector<SArray<char>> &target) {
    if constexpr (N > 0) {
      auto &t = std::get<N-1>(tup);
      typedef typename std::decay<decltype(t)>::type dtype;
      if constexpr (isScalar<dtype>::value) {
        // encode scalar type, put it in target[0]
        size_t cur_size = target[0].size();
        target[0].resize(cur_size + sizeof(dtype));
        dtype* ptr = reinterpret_cast<dtype*>(target[0].data() + cur_size);
        *ptr = t;
      } else {
        // encode sarray type, append it to target(no copy)
        SArray<char> bytes(t);
        target.push_back(bytes);
      }
      tupleSerializer<Tuple, N-1>::encode(tup, target);
    }
  }
//---------------------------------Decode---------------------------------------
  // scalar_hint, array_hint, tell where to take the data from target
  static void decode(Tuple &tup, const vector<SArray<char>> &target, size_t scalar_hint, size_t array_hint) {
    // When decode, from front to back
    if constexpr (N > 0) {
      auto &t = std::get<std::tuple_size<Tuple>::value - N>(tup);
      typedef typename std::decay<decltype(t)>::type dtype;
      if constexpr (isScalar<dtype>::value) {
        dtype* ptr = reinterpret_cast<dtype*>(target[0].data() + scalar_hint - sizeof(dtype));
        t = *ptr;
        scalar_hint -= sizeof(dtype);
      } else {
        t = target[array_hint-1];
        array_hint--;
      }
      tupleSerializer<Tuple, N-1>::decode(tup, target, scalar_hint, array_hint);
    }
  }
};

// ------------------------------ Exported APIs ------------------------------------------------
template <typename Tuple>
void tupleEncode(const Tuple &tup, vector<SArray<char>> &dest) {
  dest.clear();
  dest.push_back(SArray<char>()); // Reserve for scalar types
  dest[0].reserve(sizeof(Tuple));
  tupleSerializer<Tuple, std::tuple_size<Tuple>::value>::encode(tup, dest);
}

template <typename Tuple>
void tupleDecode(Tuple &tup, const vector<SArray<char>> &dest) {
  tupleSerializer<Tuple, std::tuple_size<Tuple>::value>::decode(tup, dest, dest[0].size(), dest.size());
}

} // namespace ps
