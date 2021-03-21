import numpy as np
import argparse

import DistGNN

def test(args):
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.nrank()
    shard = DistGNN.distributed.Shard(args.path, rank)
    print("load")
    shard.upload()
    if rank != 0:
        return
    comm = DistGNN._PS.get_handle()
    pack = DistGNN._C.NodePack()
    ts = comm.pull([1,2,3], pack)
    comm.wait(ts)
    print(pack[1].i, pack[1].f, pack[1].e)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    DistGNN.launcher(test, args)
