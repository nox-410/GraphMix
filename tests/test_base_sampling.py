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
    comm = graphmix._C.get_client()
    query = comm.pull_graph()
    graph = comm.resolve(query)
    graph.convert2coo()
    cora_dataset = graphmix.dataset.load_dataset("Cora")
    index = graph.i_feat[:,-1]
    for f, i in zip(graph.f_feat, graph.i_feat):
        idx = i[-1]
        assert np.all(f==cora_dataset.x[idx])
        assert i[0] == cora_dataset.y[idx]
    all_edge = np.array(cora_dataset.graph.edge_index).T
    for u,v in zip(graph.edge_index[0], graph.edge_index[1]):
        assert (index[u], index[v]) in all_edge
    print("CHECK OK")

def server_init(server):
    server.add_local_node_sampler(128)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
