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

from torch_model import torch_sync_data, Net

class PytorchTrain():
    def __init__(self, args):
        comm = graphmix.Client()
        self.meta = comm.meta
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=comm.num_worker(),
            rank=comm.rank()
        )
        self.device = args.local_rank
        self.eval_dataset = []
        torch.cuda.set_device(self.device)
        if dist.get_rank() == 0:
            self.dataset = load_dataset(self.meta["name"])

    def train_once(self, samplers):
        self.train_info = []
        meta = self.meta
        device = self.device
        model = Net(meta["float_feature"], meta["class"], args.hidden).cuda(device)
        DDPmodel = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        optimizer = torch.optim.Adam(DDPmodel.parameters(), args.lr)
        num_nodes, num_epoch = 0, 0
        comm = graphmix.Client()
        query = comm.pull_graph()
        start = time.time()
        best_result = 0
        test_acc_result = 0
        converge_epoch = 0

        while True:
            sampler = samplers[random.randrange(len(samplers))]
            graph = comm.wait(query)
            graph.add_self_loop()
            query = comm.pull_graph(sampler)
            x = torch.Tensor(graph.f_feat).to(device)
            y = torch.Tensor(graph.i_feat).to(device, torch.long)
            if graph.type == graphmix.sampler.GraphSage:
                train_mask = torch.Tensor(graph.extra[:, 0]).to(device, torch.long)
            else:
                train_mask = y[ : , -1] == 1
            out = DDPmodel(x, graph)
            label = y[ : , : -1]
            loss = model.loss(out, label, train_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total = train_mask.sum()
            total = torch_sync_data(total)
            num_nodes += total
            if num_nodes >= meta["train_node"]:
                num_epoch += 1
                num_nodes = 0
                if num_epoch == args.num_epoch:
                    break
                if dist.get_rank() == 0:
                    eval_acc, test_acc = self.eval_data(model)
                    self.train_info.append((eval_acc, test_acc))
                    if eval_acc > best_result:
                        best_result = eval_acc
                        test_acc_result = test_acc
                        converge_epoch = num_epoch
        return best_result, test_acc_result, converge_epoch

    # eval full-batch data on model once, return accuracy
    def eval_data(self, model):
        if not self.eval_dataset:
            device = self.device
            self.eval_dataset = [
                torch.Tensor(self.dataset.x).to(device),
                torch.Tensor(self.dataset.y).to(device, torch.long),
                torch.Tensor(self.dataset.train_mask).to(device),
                self.dataset.graph
            ]
        eval_acc, test_acc = model.evaluate(*self.eval_dataset)
        return eval_acc, test_acc

def worker_main(args):
    from graphmix.utils import powerset
    driver = PytorchTrain(args)
    comm = graphmix.Client()
    mapping = {
        "G" : graphmix.sampler.GraphSage,
        "R" : graphmix.sampler.RandomWalk,
        "L" : graphmix.sampler.LocalNode,
    }
    tests = powerset("GRL")
    train_dict = {}
    if comm.rank() == 0:
        log_file = open("log.txt", "w")
    for test in tests:
        test = "".join(test)
        samplers = list(test)
        eval_accs, test_accs, epochs = [], [], []
        for i in range(args.rerun):
            eval_acc, test_acc, epoch = driver.train_once(samplers)
            eval_accs.append(eval_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)
        if comm.rank() == 0:
            printstr = "\t{:.3f}+-{:.4f}\t{:.3f}+-{:.4f}\t{:.1f}+-{:.3f}".format(
                np.mean(eval_accs), np.std(eval_accs), np.mean(test_accs), np.std(test_accs), np.mean(epochs), np.std(epochs))
            print(test, printstr)
            print(test, printstr, file=log_file, flush=True)
            train_dict["{}{}".format(test, i)]=driver.train_info
    if comm.rank() == 0:
        with open("train_data.yml", "w") as f:
            f.write(yaml.dump(train_dict))

def server_init(server):
    batch_size = args.batch_size
    label_rate = server.meta["train_node"] / server.meta["node"]
    server.init_cache(1, graphmix.cache.LFUOpt)
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=int(batch_size * label_rate), depth=2, width=5, thread=4, tag="G")
    server.add_sampler(graphmix.sampler.RandomWalk, rw_head=int(batch_size/3), rw_length=2, thread=4, tag="R")
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=batch_size, thread=4, tag="L")
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--hidden", default=256, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--rerun", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()
    graphmix.launcher(worker_main, args, server_init=server_init)
