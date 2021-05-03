#pragma once

#include "ps/psf/serializer.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "ps/internal/message.h"
#include "callback_store.h"
#include "client.h"
#include <memory>
#include <vector>
namespace ps {

// Use EmptyHandler if not server
class EmptyHandler {
public:
  template<class A, class B>
  void serve(A request, B response) {
    LF << "Worker Error: Request should not be sent to worker. ";
  }
};

template<class Handler>
class KVApp {
public:
  explicit KVApp(int app_id=0, int customer_id=0, std::shared_ptr<Handler> handler=nullptr, int port = -1) {
    stand_alone_ = port > 0;
    handler_ = handler ? handler : std::make_shared<Handler>();
    _init_message_handlers<PsfType(0)>();
    customer_.reset(new Customer(
      app_id, customer_id,
      std::bind(&KVApp::_process, this, std::placeholders::_1),
      stand_alone_
    ));
    if (stand_alone_)
      cli_ = std::make_unique<graphmix::Client>(port, customer_);
  }
  ~KVApp() {
    // Free customer first, so that recv threads can end safely
    customer_.reset();
  }
  void Wait(int timestamp) { customer_->WaitRequest(timestamp); }
  template<PsfType ftype, typename CallBack>
  int Request(const typename PSFData<ftype>::Request &request, const CallBack &cb, int target_server_id) {
    int timestamp = customer_->NewRequest(kServerGroup);
    CallbackStore<ftype>::Get()->store(timestamp, cb);
    // Create message
    Message msg;
    tupleEncode(request, msg.data);
    msg.meta.app_id = customer_->app_id();
    msg.meta.customer_id = customer_->customer_id();
    msg.meta.timestamp = timestamp;
    msg.meta.psftype = ftype;
    msg.meta.request = true;
    if (stand_alone_) {
      msg.meta.recver = 0;
      cli_->SendMsg(msg);
    } else {
      msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
      Postoffice::Get()->van()->Send(msg);
    }
    return timestamp;
  }
  std::shared_ptr<Handler> getHandler() { return handler_; }
private:
  template<PsfType ftype>
  void onReceive(const Message &msg) {
    if (msg.meta.request) {
      typename PSFData<ftype>::Request request;
      typename PSFData<ftype>::Response response;
      tupleDecode(request ,msg.data);
      handler_->serve(request, response);
      Message rmsg;
      tupleEncode(response, rmsg.data);
      rmsg.meta = msg.meta;
      rmsg.meta.recver = msg.meta.sender;
      rmsg.meta.request = false;
      if (Postoffice::Get()->van()->IsReady())
        Postoffice::Get()->van()->Send(rmsg);
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
#if __cplusplus < 201703L
    KVAppRegisterHelper<PsfType(0), KVApp<Handler>>::init(this);
#else
    if constexpr (ftype != kNumPSfunction) {
      _message_handlers[ftype] = std::bind(&KVApp::template onReceive<ftype>, this, std::placeholders::_1);
      _init_message_handlers<PsfType(ftype+1)>();
    }
#endif
  }
#if __cplusplus < 201703L
  // Recursively register receive message handler (from 0 to kNumPSfunction)
  template<PsfType ftype, typename app>
  struct KVAppRegisterHelper {
    static void init(app* ptr) {
      ptr->_message_handlers[ftype] = std::bind(&app::template onReceive<ftype>, ptr, std::placeholders::_1);
      KVApp::KVAppRegisterHelper<PsfType(ftype+1), app>::init(ptr);
    }
  };
  template<typename app>
  struct KVAppRegisterHelper<kNumPSfunction, app> {
    static void init(app* ptr) {}
  };
  template<PsfType, typename> friend struct KVAppRegisterHelper;
#endif
  std::shared_ptr<Handler> handler_;
  std::shared_ptr<Customer> customer_;
  std::unique_ptr<graphmix::Client> cli_;
  Customer::RecvHandle _message_handlers[kNumPSfunction];
  bool stand_alone_;
};

} // namespace ps
