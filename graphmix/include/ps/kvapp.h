#pragma once

#include "ps/psf/serializer.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "ps/internal/message.h"
#include "callback_store.h"
#include <memory>
#include <vector>
namespace ps {

class EmptyHandler {};

// Recursively register receive message handler (from 0 to kNumPSfunction)
template<PsfType ftype, typename app>
struct KVAppRegisterHelper {
  static void init(app* ptr) {
    if constexpr (ftype != kNumPSfunction) {
      ptr->message_handlers[ftype] = std::bind(&app::template onReceive<ftype>, ptr, std::placeholders::_1);
      KVAppRegisterHelper<PsfType(ftype+1), app>::init(ptr);
    }
  }
};

template<class Handler>
class KVApp {
public:
  explicit KVApp(int app_id, Handler handler) {
    KVAppRegisterHelper<PsfType(0), KVApp>::init(this);
    obj_.reset(new Customer(
      app_id, app_id,
      std::bind(&KVApp::Process, this, std::placeholders::_1)
    ));
  }
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); }
  template<PsfType ftype, typename CallBack>
  int Request(const typename PSFData<ftype>::Request &request, const CallBack &cb, int target_server_id) {
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
  Handler handler;
  std::unique_ptr<Customer> obj_;
private:
  template<PsfType ftype>
  void onReceive(const Message &msg) {
    if (msg.meta.request) {
      if constexpr (std::is_same<Handler, EmptyHandler>()) {
        LF << "Request should not be sent to this node. "
          << Postoffice::Get()->van()->my_node().DebugString();
      } else {
        typename PSFData<ftype>::Request request;
        typename PSFData<ftype>::Response response;
        tupleDecode(request ,msg.data);
        handler.serve(request, response);
        Message rmsg;
        tupleEncode(response, rmsg.data);
        rmsg.meta = msg.meta;
        rmsg.meta.recver = msg.meta.sender;
        rmsg.meta.request = false;
        Postoffice::Get()->van()->Send(rmsg);
      }
    } else {
      typename PSFData<ftype>::Response response;
      tupleDecode(response, msg.data);
      int timestamp = msg.meta.timestamp;
      CallbackStore<ftype>::Get()->run(timestamp, response);
    }
  }
  void Process(const Message &msg) {
    CHECK_LT(msg.meta.psftype, kNumPSfunction) << "Unknown PS Function Received";
    message_handlers[msg.meta.psftype](msg);
  }

  typedef std::function<void(const Message&)> MessageHandle;
  MessageHandle message_handlers[kNumPSfunction];
  template<PsfType, typename> friend struct KVAppRegisterHelper;
};

} // namespace ps
