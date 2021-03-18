#include "ps/worker/worker.h"

Worker::Worker() {}

Worker::query_t Worker::push_data(const long *indices, int index_size, float *data, const long *lengths) {
    data_mu.lock();
    query_t cur_query = next_query++;
    auto &timestamps = query2timestamp[cur_query];
    data_mu.unlock();

    for(int i = 0; i < index_size; i++) {
        Key idx = (Key)indices[i];
        auto len = lengths[i];
        PSAgent::Get()->PushData(idx, data, len, timestamps);
        data += len;
    }
    return cur_query;
}

// this is almost the same as push_data
Worker::query_t Worker::pull_data(const long *indices, int index_size, float *data, const long *lengths) {
    data_mu.lock();
    query_t cur_query = next_query++;
    auto &timestamps = query2timestamp[cur_query];
    data_mu.unlock();

    for(int i = 0; i < index_size; i++) {
        Key idx = (Key)indices[i];
        auto len = lengths[i];
        PSAgent::Get()->PullData(idx, data, len, timestamps);
        data += len;
    }
    return cur_query;
}

/*
    wait_data waits until a query success
*/
void Worker::wait_data(query_t query) {
    data_mu.lock();
    auto iter = query2timestamp.find(query);
    if (iter == query2timestamp.end()) {
        data_mu.unlock();
        LG << "Wait on empty query " << query;
        return;
    } else {
        auto timestamps = std::move(iter->second);
        query2timestamp.erase(iter);
        data_mu.unlock();
        for(int t: timestamps) {
            PSAgent::Get()->waitTimestamp(t);
        }
    }
}

Worker worker;
