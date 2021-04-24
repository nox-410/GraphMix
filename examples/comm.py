import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import graphmix
import torch
import torch.distributed as dist

def torch_sync_data(*args):
    # all-reduce train stats
    t = torch.tensor(args, dtype=torch.float64)
    dist.barrier()  # synchronizes all processes
    dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t

def test(args):
    rank = graphmix._C.rank()
    nrank = graphmix._C.num_worker()
    comm = graphmix._C.get_client()
    num_node = comm.meta["node"]
    num_batch = num_node // args.batch_size
    print("{} batch per epoch".format(num_batch))
    for epoch in range(20):
        for i in range(num_batch):
            query = comm.pull_graph()
            graph = comm.resolve(query)
        if epoch == 10:
            time.sleep(1)
            graphmix._C.barrier_all()
    time.sleep(1)
    graphmix._C.barrier_all()

def server_init(server):
    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=graphmix._C.num_server(),
        rank=graphmix._C.rank()
    )
    batch_size = args.batch_size
    label_rate = server.meta["train_node"] / server.meta["node"]
    server.init_cache(args.cache_size, eval("graphmix.cache.{}".format(args.cache_method)))
    #server.add_sampler(graphmix.sampler.LocalNode, batch_size=batch_size)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=int(batch_size * label_rate), depth=2, width=5, thread=4, index=100)
    #server.add_sampler(graphmix.sampler.RandomWalk, rw_head=int(batch_size/3), rw_length=2, thread=4)
    #server.add_sampler(graphmix.sampler.GlobalNode, batch_size=batch_size)
    server.is_ready()
    graphmix._C.barrier_all()
    perf = server.get_perf()
    graphmix._C.barrier_all()
    perf2 = server.get_perf()
    value = []
    for i in range(len(perf)):
        value.append(perf2[i] - perf[i])
    value = torch_sync_data(*value)
    if graphmix._C.rank() == 0:
        print("Locallity : {:.3f}".format(1 - value[1]/value[2]))
        print("Hit rate : {:.3f}".format(1 -value[0]/value[1]))
        print("Miss node per epoch {:.1f}".format(value[0]/3))
        print(value.numpy())

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    parser.add_argument("--batch_size", default=300, type=int)
    parser.add_argument("--cache_size", default=0.1, type=float)
    parser.add_argument("--cache_method", default="LFUOpt", type=str)
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
