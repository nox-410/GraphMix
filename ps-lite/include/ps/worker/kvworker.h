#pragma once

#include "ps/psf/PSFunc.h"
#include "ps/psf/serializer.h"
#include "callback_store.h"
#include "ps/kvapp.h"
#include <vector>
#include <memory>

namespace ps {

template<PsfType> struct KVWorkerRegisterHelper;

class KVWorker : private KVApp {
public:
  /**
   * \brief constructor
   *
   * \param app_id the app id, should match with \ref KVServer's id
   * \param customer_id the customer id which is unique locally
   */
  explicit KVWorker(int app_id, int customer_id) : KVApp(app_id) {
    KVAppRegisterHelper<PsfType(0), KVWorker>::init(this);
  }

  ~KVWorker() {}

  /**
   * \brief Waits until a Request has been finished
   *
   * Sample usage:
   * \code
   *   _kvworker.Wait(ts);
   * \endcode
   *
   * \param timestamp the timestamp returned by kvworker.Request
   */
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); }
  /**
   * \brief make a new Request
   *
   * Sample usage:
   * \code
   *   int ts = _kvworker.Request<DensePush>(request, callback);
   * \endcode
   *
   * \param request create request by PSFData<PsfType>::Request
   * \param cb the callback returned by getCallback<PSfType>(args...)
   */
  template<PsfType ftype, typename Tuple, typename CallBack>
  int Request(const Tuple &request, const CallBack &cb, int target_server_id) {
    int timestamp = obj_->NewRequest(kServerGroup);
    CallbackStore<ftype>::Get()->store(timestamp, cb);
    // Create message
    Message msg;
    tupleEncode(request, msg.data);
    msg.meta.app_id = obj_->app_id();
    msg.meta.customer_id = obj_->customer_id();
    msg.meta.timestamp = timestamp;
    msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
    msg.meta.psftype = ftype;
    msg.meta.request = true;
    Postoffice::Get()->van()->Send(msg);
    return timestamp;
  }
private:
  template<PsfType ftype>
  void onReceive(const Message &msg) {
    typename PSFData<ftype>::Response response;
    tupleDecode(response ,msg.data);
    int timestamp = msg.meta.timestamp;
    CallbackStore<ftype>::Get()->run(timestamp, response);
  }
  template<PsfType, typename> friend struct KVAppRegisterHelper;
};

} // namespace ps
