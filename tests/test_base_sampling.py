import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import graphmix

max_thread = 5

def test(args):
    comm = graphmix.Client()
    rank = comm.rank()
    nrank = comm.num_worker()
    query = comm.pull_graph()
    graph = comm.wait(query)
    graph.convert2coo()
    cora_dataset = graphmix.dataset.load_dataset("Cora")
    index = graph.i_feat[:,-2]
    for f, i in zip(graph.f_feat, graph.i_feat):
        idx = i[-2]
        assert np.all(f==cora_dataset.x[idx])
        assert i[0] == cora_dataset.y[idx]
    all_edge = np.array(cora_dataset.graph.edge_index).T
    for u,v in zip(graph.edge_index[0], graph.edge_index[1]):
        assert (index[u], index[v]) in all_edge
    print("CHECK OK")

def server_init(server):
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=128)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
