import numpy as np
import argparse

import DistGNN


def test(args):
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.num_worker()
    comm = DistGNN._PS.get_client()
    pack = DistGNN._C.NodePack()
    dataset = DistGNN.dataset.load_dataset("Cora")
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
    parser.add_argument("config")
    args = parser.parse_args()
    DistGNN.launcher(test, args)
