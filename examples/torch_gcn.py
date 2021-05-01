import numpy as np
import argparse
import os
import yaml
import time
import random

import graphmix
import torch
import torch.distributed as dist

from torch_model import torch_sync_data, Net

def worker_main(args):
    comm = graphmix.Client()
    meta = comm.meta
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=comm.num_worker(),
        rank=comm.rank()
    )
    device = args.local_rank
    torch.cuda.set_device(device)
    model = Net(meta["float_feature"], meta["class"], args.hidden).cuda(device)
    DDPmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    optimizer = torch.optim.Adam(DDPmodel.parameters(), 1e-3)

    query = comm.pull_graph()
    batch_num = meta["node"] // (args.batch_size * dist.get_world_size())
    times = []
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        count, total, wait_time = 0, 0, 0
        for i in range(batch_num):
            wait_start = time.time()
            graph = comm.wait(query)
            wait_time += time.time() - wait_start
            graph.add_self_loop()
            query = comm.pull_graph()
            x = torch.Tensor(graph.f_feat).to(device)
            y = torch.Tensor(graph.i_feat).to(device, torch.long)
            if graph.type == graphmix.sampler.GraphSage:
                train_mask = torch.Tensor(graph.extra[:, 0]).to(device, torch.long)
            else:
                train_mask = y[ : , -1] == 1
            label = y[ : , : -1]
            out = DDPmodel(x, graph)
            loss = model.loss(out, label, train_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total += train_mask.sum()
            acc = model.metrics(out, label, train_mask)
            count += int(train_mask.sum() * acc)
        epoch_end_time = time.time()
        times.append(epoch_end_time-epoch_start_time)
        count, total, wait_time = torch_sync_data(count, total, wait_time)
        if args.local_rank == 0:
            print("epoch {} time {:.3f} acc={:.3f}".format(epoch, np.array(times).mean(), count/total))
            print("wait time total : {:.3f}sec".format(wait_time / dist.get_world_size()))

def server_init(server):
    batch_size = args.batch_size
    label_rate = server.meta["train_node"] / server.meta["node"]
    server.init_cache(args.cache_size, graphmix.cache.LFUOpt)
    worker_per_server = server.num_worker() // server.num_server()
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=int(batch_size * label_rate), depth=2, width=2, thread=4 * worker_per_server)
    #server.add_sampler(graphmix.sampler.RandomWalk, rw_head=int(batch_size/3), rw_length=2, thread=4 * worker_per_server)
    #server.add_sampler(graphmix.sampler.LocalNode, batch_size=batch_size, thread=4 * worker_per_server)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--hidden", default=256, type=int)
    parser.add_argument("--cache_size", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    # parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
