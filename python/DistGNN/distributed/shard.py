import libc_GNN as _C
import libc_PS as _PS

import numpy as np
import os, yaml

def _load_graph_shard(path, shard_idx):
    path = os.path.join(path, "part{}".format(shard_idx))
    with open(os.path.join(path, "graph.npy"), 'rb') as f:
        edges = np.load(f)
    with open(os.path.join(path, "float_feature.npy"), 'rb') as f:
        float_feature = np.load(f)
    with open(os.path.join(path, "int_feature.npy"), 'rb') as f:
        int_feature = np.load(f)
    return edges, float_feature, int_feature

def _load_meta(path):
    with open(os.path.join(path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    return meta

class Shard():
    def __init__(self, path, shard_idx):
        self.edges, self.f_feat, self.i_feat = _load_graph_shard(path, shard_idx)
        self.meta = _load_meta(path)
        self.offset = self.meta["partition"]["offset"][shard_idx]
        self.num_nodes = self.meta["partition"]["nodes"][shard_idx]
        self.index = np.arange(self.offset, self.offset + self.num_nodes)

    def upload(self, sync=True):
        comm = _PS.get_handle()
        task = comm.push(self.index, self.f_feat, self.i_feat, self.edges)
        comm.wait(task)
        if sync:
            _PS.barrier()
