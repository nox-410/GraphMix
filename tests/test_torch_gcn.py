import numpy as np
import argparse
import os
import yaml
import time

import graphmix
from graphmix.torch import GCN, mp_matrix
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden):
        super(Net, self).__init__()
        self.conv1 = GCN(dim_in, hidden, activation="relu", dropout=0.1)
        self.conv2 = GCN(hidden, hidden, activation="relu", dropout=0.1)
        self.classifier = torch.nn.Linear(hidden, dim_out)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, graph):
        edge_norm = mp_matrix(graph, x.device)
        x = self.conv1(x, edge_norm)
        x = self.conv2(x, edge_norm)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

def worker_main(args):
    graphmix._C.barrier_all()
    meta = args.meta
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

    comm = graphmix._C.get_client()
    query = comm.pull_graph()
    start = time.time()

    from graphmix.dataset import load_dataset
    dataset = load_dataset("Cora")
    best_result = 0
    def eval_data():
        nonlocal best_result
        with torch.no_grad():
            model.eval()
            x = torch.Tensor(dataset.x).to(device)
            label = torch.Tensor(dataset.y).to(device, torch.long)
            out = model(x, dataset.graph)
            eval_mask = torch.Tensor(dataset.train_mask==0).to(device)
            count = int(((out.argmax(axis=1) == label)*eval_mask).sum())
            total = eval_mask.sum()
            best_result = max(best_result, float(count/total))
            print(float(count/total), best_result)
            model.train()

    for i in range(100):
        graph = comm.resolve(query)
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
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        eval_mask = y[:,1]==0
        count = int(((out.argmax(axis=1) == label)*eval_mask).sum())
        total = eval_mask.sum()

        # all-reduce train stats
        t = torch.tensor([count, total, float(loss) / dist.get_world_size()], dtype=torch.float64, device='cuda')
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            print("epoch={} loss={:.5f} acc={:.5f}".format(i, t[2], t[0]/t[1]))
            eval_data()

def server_init(server):
    #server.init_cache(1, graphmix.cache.LFUOpt)
    #server.add_sampler(graphmix.sampler.LocalNode, batch_size=512)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=16, depth=2, width=2)
    #server.add_sampler(graphmix.sampler.RandomWalk, rw_head=256, rw_length=2)
    #server.add_sampler(graphmix.sampler.GlobalNode, batch_size=2708)
    graphmix._C.barrier_all()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    # parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
