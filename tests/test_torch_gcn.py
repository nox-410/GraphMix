import numpy as np
import argparse
import os
import yaml

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
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    comm = graphmix._C.get_client()

    for i in range(100):
        query = comm.pull_graph()
        graph = comm.resolve(query)
        x = torch.Tensor(graph.f_feat).to(device)
        y = torch.Tensor(graph.i_feat[:,0]).to(device, torch.long)
        out = model(x, graph)
        loss = F.cross_entropy(out, y)
        count = int((out.argmax(axis=1) == y).sum())
        total = y.shape[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # all-reduce train stats
        t = torch.tensor([count, total, float(loss) / dist.get_world_size()], dtype=torch.float64, device='cuda')
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            print("epoch={} loss={:.5f} acc={:.5f}".format(i, t[2], t[0]/t[1]))

def server_init(server):
    server.add_local_node_sampler(256)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    # parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
