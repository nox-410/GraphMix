import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import DistGNN

max_thread = 5

def test(args):
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.num_worker()
    shard = DistGNN.distributed.Shard(args.path, rank)
    if rank != 0:
        return
    comm = DistGNN._PS.get_handle()
    t = ThreadPoolExecutor(max_workers=max_thread)
    indices = np.arange(shard.meta["node"])
    item_count = 0
    def pull_data():
        while True:
            pack = DistGNN._C.NodePack()
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

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    DistGNN.launcher(test, args)
