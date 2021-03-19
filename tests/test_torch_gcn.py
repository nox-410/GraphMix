import numpy as np
import argparse
import os
import yaml

import DistGNN
from DistGNN.layer import torch_GCN
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden):
        super(Net, self).__init__()
        self.conv1 = torch_GCN(dim_in, hidden, activation="relu", dropout=0.1)
        self.conv2 = torch_GCN(hidden, hidden, activation="relu", dropout=0.1)
        self.classifier = torch.nn.Linear(hidden, dim_out)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, graph):
        edge_norm = DistGNN.graph.mp_matrix(graph, x.device, system="Pytorch")
        x = self.conv1(x, edge_norm)
        x = self.conv2(x, edge_norm)
        x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

def test(args):
    with open(os.path.join(args.path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    DistGNN.distributed.ps_init()
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=DistGNN._PS.nrank(),
    	rank=DistGNN._PS.rank()
    )
    device = args.local_rank
    torch.cuda.set_device(device)
    model = Net(meta["feature"], meta["class"], 128).cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    with DistGNN.distributed.DistributedGraphSageSampler(args.path, 128, 2, 2,
        rank=DistGNN._PS.rank(), nrank=DistGNN._PS.nrank() , cache_size_factor=1, reduce_nonlocal_factor=0, num_sample_thread=4) as sampler:
        for i in range(100):
            g_sample, mask = sampler.sample()
            x = torch.Tensor(g_sample.x).to(device)
            y = torch.Tensor(g_sample.y).to(device, torch.long)
            out = model(x, g_sample)
            loss = F.cross_entropy(out, y)
            acc = int((out.argmax(axis=1) == y).sum()) / y.shape[0]
            print("epoch={} loss={:.5f} acc={:.5f}".format(i, float(loss), acc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    DistGNN.launcher(test, args)
