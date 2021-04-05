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
    graph, query = comm.pull_graph()
    comm.wait(query)
    print(graph, query)

def server_init(server):
    server.add_local_node_sampler(128)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
