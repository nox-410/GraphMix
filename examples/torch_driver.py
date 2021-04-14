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

class PytorchTrain():
    def __init__(self, args):
        comm = graphmix._C.get_client()
        self.meta = comm.meta
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=graphmix._C.num_worker(),
            rank=graphmix._C.rank()
        )
        self.device = args.local_rank
        torch.cuda.set_device(self.device)
        if dist.get_rank() == 0:
            self.dataset = load_dataset(self.meta["name"])

    def train_once(self, samplers):
        meta = self.meta
        device = self.device
        model = Net(meta["float_feature"], meta["class"], args.hidden).cuda(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        num_nodes, num_epoch = 0, 0
        comm = graphmix._C.get_client()
        query = comm.pull_graph()
        start = time.time()
        best_result = 0
        converge_epoch = 0

        while True:
            sampler = samplers[random.randrange(len(samplers))]
            graph = comm.resolve(query)
            graph.add_self_loop()
            query = comm.pull_graph(sampler)
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
            total = train_mask.sum()
            loss = loss.sum() / total
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            count = int(((out.argmax(axis=1) == label)*train_mask).sum())
            count, total, loss = self.torch_sync_data(count, total, loss)
            num_nodes += total
            if num_nodes >= meta["train_node"]:
                num_epoch += 1
                num_nodes = 0
                if num_epoch == args.num_epoch:
                    break
                if dist.get_rank() == 0:
                    cur_result = self.eval_data(model)
                    if cur_result > best_result:
                        best_result = cur_result
                        converge_epoch = num_epoch
        return best_result, converge_epoch

    # eval full-batch data on model once, return accuracy
    def eval_data(self, model):
        dataset = self.dataset
        device = self.device
        with torch.no_grad():
            model.eval()
            x = torch.Tensor(dataset.x).to(device)
            label = torch.Tensor(dataset.y).to(device, torch.long)
            out = model(x, dataset.graph)
            eval_mask = torch.Tensor(dataset.train_mask==0).to(device)
            count = int(((out.argmax(axis=1) == label)*eval_mask).sum())
            total = eval_mask.sum()
            model.train()
            return float(count/total)

    @staticmethod
    def torch_sync_data(*args):
        # all-reduce train stats
        t = torch.tensor(args, dtype=torch.float64, device='cuda')
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return t

def worker_main(args):
    driver = PytorchTrain(args)
    mapping = {
        "G" : graphmix.sampler.GraphSage,
        "R" : graphmix.sampler.RandomWalk,
        "L" : graphmix.sampler.LocalNode,
        "F" : graphmix.sampler.GlobalNode,
    }
    tests = ("G", "R", "L", "F", "GR", "GL", "GF",
    "RL", "RF", "LF", "GRL", "GRF", "GLF", "GRLF")
    if graphmix._C.rank() == 0:
        log_file = open("log.txt", "w")
    for test in tests:
        samplers = list(map(mapping.get, test))
        accs, epochs = [], []
        for i in range(10):
            acc, epoch = driver.train_once(samplers)
            accs.append(acc)
            epochs.append(epoch)
        if graphmix._C.rank() == 0:
            printstr = "\t{:.3f}+-{:.5f}\t{:.1f}+-{:.3f}".format(
                np.mean(accs), np.std(accs), np.mean(epochs), np.std(epochs))
            print(test, printstr)
            print(test, printstr, file=log_file, flush=True)

def server_init(server):
    batch_size = args.batch_size
    label_rate = server.meta["train_node"] / server.meta["node"]
    server.init_cache(1, graphmix.cache.LFUOpt)
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=batch_size)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=int(batch_size * label_rate), depth=2, width=2)
    server.add_sampler(graphmix.sampler.RandomWalk, rw_head=int(batch_size/3), rw_length=2)
    server.add_sampler(graphmix.sampler.GlobalNode, batch_size=batch_size)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--batch_size", default=300, type=int)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
