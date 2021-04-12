import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import graphmix

def test(args):
    graphmix._C.barrier_all()
    cora_dataset = graphmix.dataset.load_dataset("Cora")
    rank = graphmix._C.rank()
    nrank = graphmix._C.num_worker()
    comm = graphmix._C.get_client()

    def check(graph):
        if graph.tag == graphmix.sampler.GraphSage:
            assert np.all(graph.extra[:,0] <= graph.i_feat[:,1])
        for f, i in zip(graph.f_feat, graph.i_feat):
            idx = i[-1]
            assert np.all(f==cora_dataset.x[idx])
            assert i[0] == cora_dataset.y[idx]
        all_edge = np.array(cora_dataset.graph.edge_index).T
        for u,v in zip(graph.edge_index[0], graph.edge_index[1]):
            assert (index[u], index[v]) in all_edge
    for i in range(100):
        query = comm.pull_graph()
        graph = comm.resolve(query)
        graph.convert2coo()
        index = graph.i_feat[:,-1]
        check(graph)
    print("CHECK OK")

def server_init(server):
    if graphmix._C.rank() == 0:
        server.init_cache(0.3, graphmix.cache.LFUOpt)
    elif graphmix._C.rank() == 1:
        server.init_cache(0.3, graphmix.cache.LFU)
    elif graphmix._C.rank() == 2:
        server.init_cache(0.3, graphmix.cache.LRU)
    server.add_sampler(graphmix.sampler.GlobalNode, batch_size=512)
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=512)
    server.add_sampler(graphmix.sampler.RandomWalk, rw_head=256, rw_length=2)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=16, depth=2, width=2)
    graphmix._C.barrier_all()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
