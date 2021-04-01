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

template<class Handler>
class KVApp {
public:
  explicit KVApp(int app_id, Handler handler) {
    _init_message_handlers<PsfType(0)>();
    obj_.reset(new Customer(
      app_id, app_id,
      std::bind(&KVApp::_process, this, std::placeholders::_1)
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

  void _process(const Message &msg) {
    CHECK_LT(msg.meta.psftype, kNumPSfunction) << "Unknown PS Function Received";
    _message_handlers[msg.meta.psftype](msg);
  }

  // Recursively register receive message handler (from 0 to kNumPSfunction)
  template<PsfType ftype>
  void _init_message_handlers() {
    if constexpr (ftype != kNumPSfunction) {
      _message_handlers[ftype] = std::bind(&KVApp::template onReceive<ftype>, this, std::placeholders::_1);
      _init_message_handlers<PsfType(ftype+1)>();
    }
  }

  typedef std::function<void(const Message&)> MessageHandle;
  MessageHandle _message_handlers[kNumPSfunction];
};

} // namespace ps
