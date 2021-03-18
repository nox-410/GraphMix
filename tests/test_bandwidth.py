import time, os
import argparse
import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading
import DistGNN

nitem = 2000
item_len = 1000
max_thread = 10

def test(args):
    ps = DistGNN._PS
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.nrank()
    if rank > 0:
        return
    arr = np.random.rand(nitem, item_len).astype(np.float32)

    push_indices = np.arange(nitem)
    print(push_indices)
    push_length = np.repeat(item_len, repeats=nitem)
    worker_communicate = ps.get_handle()
    query = worker_communicate.push_data(push_indices, arr, push_length)
    worker_communicate.wait(query)
    print("data_pushed")
    t = ThreadPoolExecutor(max_workers=max_thread)
    byte_count = 0
    arr2 = np.random.rand(nitem, item_len).astype(np.float32)
    def pull_data():
        query = worker_communicate.pull_data(push_indices, arr2, push_length)
        worker_communicate.wait(query)
        # print( np.all(arr.asnumpy() == arr2.asnumpy()) )
        nonlocal byte_count
        byte_count += nitem * item_len * 4
    def watch():
        nonlocal byte_count
        start = time.time()
        while True:
            time.sleep(1)
            speed = byte_count / (time.time() - start)
            print("speed : {} MB/s".format(speed / 2**20))
    task_list = [None for i in range(max_thread)]
    threading.Thread(target=watch).start()
    while True:
        for i in range(max_thread):
            if task_list[i] is None or task_list[i].done():
                task_list[i] = t.submit(pull_data)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    DistGNN.launcher(args, test)
