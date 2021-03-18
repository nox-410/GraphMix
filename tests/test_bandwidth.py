import time, os, sys
import yaml
import multiprocessing
import argparse
import signal
import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading
import DistGNN

nitem = 2000
item_len = 1000
max_thread = 10

def test():
    ps = DistGNN._PS
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
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


def start_process(settings, args):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        DistGNN._PS.init()
        DistGNN._PS.start_server()
        DistGNN._PS.finalize()
    elif os.environ['DMLC_ROLE'] == "worker":
        DistGNN._PS.init()
        test()
        DistGNN._PS.finalize()
    elif os.environ['DMLC_ROLE'] == "scheduler":
        DistGNN._PS.init()
        DistGNN._PS.finalize()
    else:
        raise ValueError("Unknown role", os.environ['DMLC_ROLE'])

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    process_list = []
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(target=start_process, args=[value, args])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
