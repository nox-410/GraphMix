import numpy as np
import argparse

import DistGNN

def test(args):
    rank = DistGNN._PS.rank()
    nrank = DistGNN._PS.nrank()
    DistGNN.distributed.ps_init()
    with DistGNN.distributed.DistributedGraphSageSampler(args.path, 128, 2, 2,
        rank=rank, nrank=nrank , cache_size_factor=1, reduce_nonlocal_factor=0, num_sample_thread=4) as sampler:
        graph, mask = sampler.sample()
        print(graph)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    DistGNN.launcher(args, test)
