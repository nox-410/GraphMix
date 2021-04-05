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
    query = comm.pull_graph()
    graph = comm.resolve(query)
    print(graph.f_feat, graph.i_feat, graph.edge_index)

def server_init(server):
    server.add_local_node_sampler(128)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
