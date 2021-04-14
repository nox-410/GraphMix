import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import graphmix

max_thread = 5

def test(args):
    rank = graphmix._C.rank()
    nrank = graphmix._C.num_worker()
    if rank != 0:
        return
    comm = graphmix._C.get_client()
    t = ThreadPoolExecutor(max_workers=max_thread)
    item_count = 0
    def pull_data():
        while True:
            indices = np.random.randint(0, comm.meta["node"], 1000)
            pack = graphmix._C.NodePack()
            query = comm.pull(indices, pack)
            comm.wait(query)
            nonlocal item_count
            item_count += len(indices)

    def watch():
        nonlocal item_count
        start = time.time()
        while True:
            time.sleep(1)
            speed = item_count / (time.time() - start)
            print("speed : {} item/s".format(speed))
    task_list = [None for i in range(max_thread)]
    threading.Thread(target=watch).start()
    for i in range(max_thread):
        task_list[i] = t.submit(pull_data)
    time.sleep(1000)

def server_init(server):
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    import os
    os.environ["PS_WORKER_THREAD"]=str(max_thread)
    graphmix.launcher(test, args, server_init=server_init)
