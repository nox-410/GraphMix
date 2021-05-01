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
    rank = comm.rank()
    nrank = comm.num_worker()

    def check(graph):
        if graph.type == graphmix.sampler.GraphSage:
            assert np.all(graph.extra[:,0] <= graph.i_feat[:,-1])
        for f, i in zip(graph.f_feat, graph.i_feat):
            idx = i[-2]
            assert np.all(f==cora_dataset.x[idx])
            assert i[0] == cora_dataset.y[idx]
        all_edge = np.array(cora_dataset.graph.edge_index).T
        for u,v in zip(graph.edge_index[0], graph.edge_index[1]):
            assert (index[u], index[v]) in all_edge
    samplers = [0,1,2,3]
    for i in range(20):
        random.shuffle(samplers)
        query = comm.pull_graph(*samplers)
        graph = comm.wait(query)
        graph.convert2coo()
        index = graph.i_feat[:,-1]
        check(graph)
    print("CHECK OK")

def server_init(server):
    if server.rank() == 0:
        server.init_cache(0.3, graphmix.cache.LFUOpt)
    elif server.rank() == 1:
        server.init_cache(0.3, graphmix.cache.LFU)
    elif server.rank() == 2:
        server.init_cache(0.3, graphmix.cache.LRU)
    server.add_sampler(graphmix.sampler.GlobalNode, batch_size=512)
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=512)
    server.add_sampler(graphmix.sampler.RandomWalk, rw_head=256, rw_length=2)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=16, depth=2, width=2)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
