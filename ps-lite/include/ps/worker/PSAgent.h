#pragma once

#include "ps/ps.h"
#include "ps/worker/kvworker.h"
#include "ps/psf/PSFunc.h"
#include "ps/server/param.h"
#include "common/logging.h"

namespace ps {

class PSAgent {

private:
    /* The KVWorker used to make requests. */
    KVWorker _kvworker;

    PSAgent() : _kvworker(0, 0) {}

public:

    static PSAgent *Get() {
        static PSAgent e;
        return &e;
    }

    void waitTimestamp(int timestamp) { _kvworker.Wait(timestamp); }

    /*
        A simple key mapping for multiple server case
    */
    Key mapWkeyToSkey(Key idx) {
        const std::vector<Range> &server_range = Postoffice::Get()->GetServerKeyRanges();
        int server = idx % server_range.size();
        Key k = server_range[server].end() - idx - 1;
        return k;
    }

    /*
        Enqueue the Zpush request for PushData
    */
    void PushData(Key idx, float *vals, int len, std::vector<int>& timestamp) {
        auto cb = getCallBack<DensePush>();
        PSFData<DensePush>::Request request(mapWkeyToSkey(idx), len, SArray<float>(vals, len));
        int ts = _kvworker.Request<DensePush>(request, cb);
        timestamp.push_back(ts);
    }

    // This is almost the same as PushData
    void PullData(Key idx, float *vals, int len, std::vector<int>& timestamp) {
        auto cb = getCallBack<DensePull>(SArray<float>(vals, len));
        PSFData<DensePull>::Request request(mapWkeyToSkey(idx), len);
        int ts = _kvworker.Request<DensePull>(request, cb);
        timestamp.push_back(ts);
    }
};

} // namespace ps
