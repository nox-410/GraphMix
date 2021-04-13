#pragma once

#include <zmq.h>
#include <random>
#include <thread>
#include "ps/internal/van.h"

namespace graphmix {

using ps::Message;

/**
 * \brief be smart on freeing recved data
 */
inline void FreeData(void *data, void *hint) {
  if (hint == NULL) {
    delete[] static_cast<char*>(data);
  } else {
    delete static_cast<SArray<char>*>(hint);
  }
}

class Client
{
private:
  std::unique_ptr<std::thread> recv_thread_;
  std::shared_ptr<ps::ThreadsafeQueue<Message>> recv_queue_;
  size_t recv_bytes_ = 0;
  void *senders[2] = {nullptr, nullptr};
  void *receiver_ = nullptr;
  void *context_ = nullptr;
  void *loop_ = nullptr;
  int recv_port_;
public:
  Client(int target) {
    // start zmq
    context_ = zmq_ctx_new();
    CHECK(context_ != NULL) << "create 0mq context failed";
    zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 65536);
    int zmq_threads = ps::GetEnv("ZMQ_WORKER_THREAD", 1);
    zmq_ctx_set(context_, ZMQ_IO_THREADS, zmq_threads);
    recv_port_ = this->bind();
    std::cout << "Client bind to " << recv_port_ << std::endl;
    recv_thread_ =
        std::unique_ptr<std::thread>(new std::thread(&Client::Receiving, this));
    this->connect(target);
    sendArriveMessage();
    Message msg;
    msg.meta.timestamp = 0;
    msg.meta.psftype = ps::GraphPull;
    PSFData<GraphPull>::Request request;
    tupleEncode(request, msg.data);
    SendMsg(msg);
  }

  ~Client() { this->stop(); }

  void stop() {
    // stop receive thread
    Message msg;
    msg.meta.control.cmd = Control::LEAVE;
    SendMsg(msg);
    recv_thread_->join();

    // close sockets
    int linger = 0;
    int rc = zmq_setsockopt(receiver_, ZMQ_LINGER, &linger, sizeof(linger));
    CHECK(rc == 0 || errno == ETERM);
    CHECK_EQ(zmq_close(receiver_), 0);
    for (int i = 0; i < 2; i++) {
      int rc = zmq_setsockopt(senders[i], ZMQ_LINGER, &linger, sizeof(linger));
      CHECK(rc == 0 || errno == ETERM);
      CHECK_EQ(zmq_close(senders[i]), 0);
    }
    zmq_ctx_destroy(context_);
    context_ = nullptr;
  }

  int bind(int max_retry=40) {
    receiver_ = zmq_socket(context_, ZMQ_ROUTER);
    CHECK(receiver_ != NULL)
        << "create receiver socket failed: " << zmq_strerror(errno);
    std::string addr = "ipc:///tmp/";

    std::mt19937_64 rd;
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    rd.seed(seed);
    int port;
    for (int i = 0; i < max_retry + 1; ++i) {
      port = 10000 + rd() % 40000;
      auto address = addr + std::to_string(port);
      if (zmq_bind(receiver_, address.c_str()) == 0) break;
      if (i == max_retry) {
        throw std::runtime_error("Client bind fail max_retry");
      }
    }
    return port;
  }

  void connect(int target) {
    for (int i = 0; i < 2; i++) {
      senders[i] = zmq_socket(context_, ZMQ_DEALER);
      CHECK(senders[i] != NULL) << zmq_strerror(errno);
      std::string my_id = "wk" + std::to_string(recv_port_);
        zmq_setsockopt(senders[i], ZMQ_IDENTITY, my_id.data(), my_id.size());
      // connect
      // sender[0] connect to server, sender[1] connect to self
      std::string addr;
      if (i == 0)
        addr = "tcp://127.0.0.1:" + std::to_string(target);
      else
        addr = "ipc:///tmp/" + std::to_string(recv_port_);
      if (zmq_connect(senders[i], addr.c_str()) != 0) {
        LOG(FATAL) <<  "connect to " + addr + " failed: " + zmq_strerror(errno);
      }
    }
  }

  void sendArriveMessage() {
    Message msg;
    msg.meta.control.cmd = Control::ARRIVE;
    SendMsg(msg);
  }

  void Receiving() {
    while (true) {
      Message msg;
      int recv_bytes = RecvMsg(&msg);
      CHECK_NE(recv_bytes, -1);
      if (msg.meta.control.cmd == Control::TERMINATE) break;
      recv_bytes_ += recv_bytes;
    }
  }

  int SendMsg(Message& msg, bool send_to_self=false) {
    // send meta
    void * sender = send_to_self ? senders[1] : senders[0];
    msg.meta.sender = recv_port_;
    msg.meta.app_id = 0;
    msg.meta.recver = 0;
    msg.meta.request = true;
    msg.meta.customer_id = 0;
    int meta_size; char* meta_buf;
    PackMeta(msg.meta, &meta_buf, &meta_size);
    int tag = ZMQ_SNDMORE;
    int n = msg.data.size();
    if (n == 0) tag = 0;
    zmq_msg_t meta_msg;
    zmq_msg_init_data(&meta_msg, meta_buf, meta_size, FreeData, NULL);
    while (true) {
      if (zmq_msg_send(&meta_msg, sender, tag) == meta_size) break;
      if (errno == EINTR) continue;
      return -1;
    }
    // zmq_msg_close(&meta_msg);
    int send_bytes = meta_size;
    // send data
    for (int i = 0; i < n; ++i) {
      zmq_msg_t data_msg;
      SArray<char>* data = new SArray<char>(msg.data[i]);
      int data_size = data->size();
      zmq_msg_init_data(&data_msg, data->data(), data->size(), FreeData, data);
      if (i == n - 1) tag = 0;
      while (true) {
        if (zmq_msg_send(&data_msg, sender, tag) == data_size) break;
        if (errno == EINTR) continue;
        LOG(WARNING) << "failed to send message to node error: " << errno << " " << zmq_strerror(errno)
                     << ". " << i << "/" << n;
        return -1;
      }
      // zmq_msg_close(&data_msg);
      send_bytes += data_size;
    }
    return send_bytes;
  }

  int RecvMsg(Message* msg) {
    msg->data.clear();
    size_t recv_bytes = 0;
    for (int i = 0; ; ++i) {
      zmq_msg_t* zmsg = new zmq_msg_t;
      CHECK(zmq_msg_init(zmsg) == 0) << zmq_strerror(errno);
      while (true) {
        if (zmq_msg_recv(zmsg, receiver_, 0) != -1) break;
        if (errno == EINTR) {
          std::cout << "interrupted";
          continue;
        }
        LOG(WARNING) << "failed to receive message. errno: "
                     << errno << " " << zmq_strerror(errno);
        return -1;
      }
      char* buf = CHECK_NOTNULL((char *)zmq_msg_data(zmsg));
      size_t size = zmq_msg_size(zmsg);
      recv_bytes += size;

      if (i == 0) {
        // identify
        CHECK(zmq_msg_more(zmsg));
        zmq_msg_close(zmsg);
        delete zmsg;
      } else if (i == 1) {
        // task
        UnpackMeta(buf, size, &(msg->meta));
        zmq_msg_close(zmsg);
        bool more = zmq_msg_more(zmsg);
        delete zmsg;
        if (!more) break;
      } else {
        // zero-copy
        SArray<char> data;
        data.reset(buf, size, [zmsg, size](char* buf) {
          zmq_msg_close(zmsg);
          delete zmsg;
        });
        msg->data.push_back(data);
        if (!zmq_msg_more(zmsg)) {
          break;
        }
      }
    }
    return recv_bytes;
  }


};

} // namespace graphmix



