import numpy as np
import argparse

import graphmix


def test(args):
    comm = graphmix.Client()
    rank = comm.rank()
    nrank = comm.num_worker()
    dataset = graphmix.dataset.load_dataset("Cora")
    num_nodes = dataset.graph.num_nodes
    query = comm.pull_node(np.arange(num_nodes))
    pack = comm.wait(query)
    assert len(pack) == num_nodes
    reindex = {}
    for i, node in pack.items() :
        idx = node.i[-2]
        reindex[idx] = i
        assert np.all(dataset.x[idx] == node.f)
        assert dataset.y[idx] == node.i[0]
    for u, v in zip(dataset.graph.edge_index[0], dataset.graph.edge_index[1]):
        assert(reindex[v] in pack[reindex[u]].e)

    print("Check OK")

def server_init(server):
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
