/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/customer.h"
#include "ps/internal/postoffice.h"

namespace ps {

const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();

Customer::Customer(int app_id, int customer_id, const Customer::RecvHandle& recv_handle, bool stand_alone)
    : app_id_(app_id), customer_id_(customer_id), recv_handle_(recv_handle) {
  cur_timestamp = 0;
  stand_alone_ = stand_alone;
  int num_threads = GetEnv("GRAPHMIX_WORKER_RECV_THREAD", 1);
  if (!stand_alone_) {
    Postoffice::Get()->AddCustomer(this);
    if (Postoffice::Get()->is_server()) {
      num_threads = GetEnv("GRAPHMIX_SERVER_RECV_THREAD", 10);
      CHECK(num_threads >= 2) << "Graph Server Requires at leat 2 receive thread";
    }
  }
  for(int i = 0; i < num_threads; i++) {
    // only allow half of the thread to handle blocking requests for graph
    bool handle_graph = i & 1;
    recv_threads_.emplace_back(new std::thread(&Customer::Receiving, this, handle_graph));
  }
}

Customer::~Customer() {
  if (!stand_alone_)
    Postoffice::Get()->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  for(auto& thread: recv_threads_)
      thread->join();
}

int Customer::NewRequest(int recver) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  assert (recver == kServerGroup);
  // int num = Postoffice::Get()->GetNodeIDs(recver).size();
  tracker_[cur_timestamp] = false;
  return cur_timestamp++;
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp];
    });
  tracker_.erase(timestamp);
}

// int Customer::NumResponse(int timestamp) {
//   std::lock_guard<std::mutex> lk(tracker_mu_);
//   return tracker_[timestamp].second;
// }

// void Customer::AddResponse(int timestamp, int num) {
//   std::lock_guard<std::mutex> lk(tracker_mu_);
//   tracker_[timestamp].second += num;
// }

void Customer::Receiving(bool handle_graph) {
  while (true) {
    Message recv;
    // thread safe
    recv_queue_.WaitAndPop(&recv);
    if (!recv.meta.control.empty() &&
        recv.meta.control.cmd == Control::TERMINATE) {
      recv_queue_.Push(recv);
      break;
    }
    if (!handle_graph && recv.meta.request && recv.meta.psftype == GraphPull) {
      recv_queue_.Push(recv);
      continue;
    }
    recv_handle_(recv);
    if (!recv.meta.request) {
      std::lock_guard<std::mutex> lk(tracker_mu_);
      tracker_[recv.meta.timestamp] = true;
      tracker_cond_.notify_all();
    }
  }
}

}  // namespace ps
