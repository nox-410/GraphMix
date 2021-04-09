import numpy as np
import argparse

import graphmix


def test(args):
    rank = graphmix._C.rank()
    nrank = graphmix._C.num_worker()
    comm = graphmix._C.get_client()
    pack = graphmix._C.NodePack()
    dataset = graphmix.dataset.load_dataset("Cora")
    num_nodes = dataset.graph.num_nodes
    task = comm.pull(np.arange(num_nodes), pack)
    comm.wait(task)
    assert len(pack) == num_nodes
    reindex = {}
    for i, node in pack.items() :
        idx = node.i[-1]
        reindex[idx] = i
        assert np.all(dataset.x[idx] == node.f)
        assert dataset.y[idx] == node.i[0]
    for u, v in zip(dataset.graph.edge_index[0], dataset.graph.edge_index[1]):
        assert(reindex[v] in pack[reindex[u]].e)

    print("Check OK")

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args)