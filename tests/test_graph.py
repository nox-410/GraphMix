import time, os, sys
import yaml
import multiprocessing
import argparse
import signal
import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading
import DistGNN

def test(args):
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    DistGNN.distributed.ps_init()
    with DistGNN.distributed.DistributedGraphSageSampler(args.path, 128, 2, 2,
        rank=rank, nrank=nrank , cache_size_factor=1, reduce_nonlocal_factor=0, num_sample_thread=4) as sampler:
        print(sampler.sample())

def start_process(settings, args):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        DistGNN._PS.init()
        DistGNN._PS.start_server()
        DistGNN._PS.finalize()
    elif os.environ['DMLC_ROLE'] == "worker":
        DistGNN._PS.init()
        test(args)
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
    parser.add_argument("--path", "-p", required=True)
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
