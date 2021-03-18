#pragma once

#include "PSFunc.h"

namespace ps {

template<> struct PSFData<DensePull> {
  using Request = tuple<
    Key // key
  >;
  using Response = tuple<
    SArray<char> // data
  >;
  static void _callback(const Response &response, SArray<char> tgt) {
    auto val = get<0>(response);
    CHECK_EQ(val.size(), tgt.size()) << val.size() << " " << tgt.size();
    std::copy(val.begin(), val.end(), tgt.begin());
  }
};

template<> struct PSFData<DensePush> {
  using Request = tuple<
    Key, // key
    SArray<char> // data
  >;
  using Response = tuple<>;
  static void _callback(const Response &response) {}
};

} // namespace ps
