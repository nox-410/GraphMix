#pragma once

#include "common/sarray.h"
#include "ps/base.h"

#include <tuple>
#include <functional>
using std::tuple;
using std::get;
using std::function;

namespace ps {

enum PsfType {
  NodePull,
  GraphPull,
  MetaPull,
  kNumPSfunction
};

template <PsfType> struct PSFData;
/*
    To define a new PSFunc, we need 3 parts : Request, Response, _callback
    * Request and Response are tuple-like object, and must only use
      scalar types like int, float or Sarray
    * _callback is a function having format void(const Response&, args...)
      where args are some target memory space to write back
    * See examples in dense.h sparse.h ...
*/


/*
  getCallBack, use this to bind _callback to the get the real callback which can be stored
  example: getCallBack<DensePull>(target);
*/
template<PsfType ftype, typename ...Args>
function<void(const typename PSFData<ftype>::Response&)> getCallBack(Args&&... args) {
  return std::bind(PSFData<ftype>::_callback, std::placeholders::_1, std::forward<Args>(args)...);
}

}

#include "node_func.h"
