#pragma once

#include <random>
#include <thread>
#include "ps/internal/van.h"
#include "ps/internal/customer.h"

namespace graphmix {

using ps::Message;

class Client {
private:
  std::unique_ptr<std::thread> recv_thread_;
  std::shared_ptr<ps::Customer> customer_;
  size_t recv_bytes_ = 0;
  void *senders[2] = {nullptr, nullptr};
  void *receiver_ = nullptr;
  void *context_ = nullptr;
  void *loop_ = nullptr;
  int recv_port_;
  int bind(int max_retry=40);
  void connect(int target);
  void sendArriveMessage();
  void Receiving();
  int RecvMsg(Message* msg);
  void stop();
public:
  Client(int target, std::shared_ptr<ps::Customer>);
  ~Client() { this->stop(); }
  int SendMsg(Message& msg, bool send_to_self=false);
};

} // namespace graphmix
