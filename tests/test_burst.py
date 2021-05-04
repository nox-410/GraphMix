import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import random
import graphmix

def test(args):
    cora_dataset = graphmix.dataset.load_dataset("Cora")
    comm = graphmix.Client()
    repeats = 1000
    querys = []
    for i in range(repeats):
        querys.append(comm.pull_graph())
    for i in range(repeats):
        graph = comm.wait(querys[i])
    print("CHECK OK")

def server_init(server):
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=64, depth=5, width=2)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
