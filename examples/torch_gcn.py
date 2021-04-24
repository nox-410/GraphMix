import numpy as np
import argparse
import os
import yaml
import time
import random

import graphmix
from graphmix.torch import SageConv, mp_matrix
from graphmix.dataset import load_dataset
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden):
        super(Net, self).__init__()
        self.conv1 = SageConv(dim_in, hidden, activation="relu", dropout=0.1)
        self.conv2 = SageConv(2*hidden, hidden, activation="relu", dropout=0.1)
        self.classifier = torch.nn.Linear(2*hidden, dim_out)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, graph):
        edge_norm = mp_matrix(graph, x.device)
        x = self.conv1(x, edge_norm)
        x = self.conv2(x, edge_norm)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

def torch_sync_data(*args):
    # all-reduce train stats
    t = torch.tensor(args, dtype=torch.float64, device='cuda')
    dist.barrier()  # synchronizes all processes
    dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t

def worker_main(args):
    comm = graphmix._C.get_client()
    meta = comm.meta
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=graphmix._C.num_worker(),
        rank=graphmix._C.rank()
    )
    device = args.local_rank
    torch.cuda.set_device(device)
    model = Net(meta["float_feature"], meta["class"], 128).cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    query = comm.pull_graph()
    batch_num = meta["node"] // (args.batch_size * dist.get_world_size())
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        count, total, wait_time = 0, 0, 0
        for i in range(batch_num):
            wait_start = time.time()
            graph = comm.resolve(query)
            wait_time += time.time() - wait_start
            graph.add_self_loop()
            query = comm.pull_graph()
            x = torch.Tensor(graph.f_feat).to(device)
            y = torch.Tensor(graph.i_feat).to(device, torch.long)
            if graph.tag == graphmix.sampler.GraphSage:
                train_mask = torch.Tensor(graph.extra[:, 0]).to(device, torch.long)
            else:
                train_mask = y[:,1]==1
            label = y[:,0]
            out = model(x, graph)
            loss = F.cross_entropy(out, label, reduction='none')
            loss = loss * train_mask
            loss = loss.sum() / train_mask.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total += train_mask.sum()
            count += int(((out.argmax(axis=1) == label)*train_mask).sum())
        epoch_end_time = time.time()
        count, total, wait_time = torch_sync_data(count, total, wait_time)
        if args.local_rank == 0:
            print("epoch {} time {:.3f} acc={:.3f}".format(epoch, epoch_end_time-epoch_start_time, count/total))
            print("wait time total : {:.3f}sec".format(wait_time / dist.get_world_size()))

def server_init(server):
    batch_size = args.batch_size
    label_rate = server.meta["train_node"] / server.meta["node"]
    server.init_cache(args.cache_size, graphmix.cache.LFUOpt)
    worker_per_server = graphmix._C.num_worker() // graphmix._C.num_server()
    #server.add_sampler(graphmix.sampler.LocalNode, batch_size=batch_size, thread=4 * worker_per_server)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=int(batch_size * label_rate), depth=2, width=2, thread=4 * worker_per_server)
    #server.add_sampler(graphmix.sampler.RandomWalk, rw_head=int(batch_size/3), rw_length=2, thread=4 * worker_per_server)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--batch_size", default=4000, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--cache_size", default=0.1, type=float)
    # parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
