import numpy as np
import argparse

import DistGNN


def test(args):
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.nrank()
    shard = DistGNN.distributed.Shard(args.path, rank)
    shard.upload()
    comm = DistGNN._PS.get_handle()
    pack = DistGNN._C.NodePack()
    task = comm.pull(shard.index, pack)
    comm.wait(task)
    if len(pack) != len(shard.index):
        print(rank, len(pack), len(shard.index))
        return
    for i, node in enumerate(shard.index) :
        assert np.all(pack[node].i == shard.i_feat[i])
        assert np.all(pack[node].f == shard.f_feat[i])
    for u, v in zip(shard.edges[0], shard.edges[1]):
        assert v in pack[u].e
    print("Check OK")

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    DistGNN.launcher(test, args)
