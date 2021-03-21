import libc_GNN as _C
import libc_PS as _PS

import numpy as np
import os, yaml

def _load_graph_shard(path, shard_idx):
    path = os.path.join(path, "part{}".format(shard_idx))
    with open(os.path.join(path, "graph.npz"), 'rb') as f:
        data = np.load(f)
        edges = data["edge"]
        index = data["index"]
    with open(os.path.join(path, "data.npz"), 'rb') as f:
        data = np.load(f)
        int_feature = data["i"]
        float_feature = data["f"]
    return index, edges, float_feature, int_feature

def _load_meta(path):
    with open(os.path.join(path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    return meta

class Shard():
    def __init__(self, path, shard_idx):
        self.index, self.edges, self.f_feat, self.i_feat = _load_graph_shard(path, shard_idx)
        self.meta = _load_meta(path)

    def upload(self, sync=True):
        comm = _PS.get_handle()
        task = comm.push(self.index, self.f_feat, self.i_feat, self.edges)
        comm.wait(task)
        if sync:
            _PS.barrier()
