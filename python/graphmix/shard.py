import libc_graphmix as _PS

import numpy as np
import os, yaml

class Shard():
    def __init__(self, path):
        self.path = path
        self._load_meta()

    def _load_meta(self):
        with open(os.path.join(self.path, "meta.yml"), 'rb') as f:
            self.meta = yaml.load(f.read(), Loader=yaml.FullLoader)

    def load_graph_shard(self, shard_idx):
        assert shard_idx >= 0
        path = os.path.join(self.path, "part{}".format(shard_idx))
        with open(os.path.join(path, "graph.npy"), 'rb') as f:
            self.edges = np.load(f)
        with open(os.path.join(path, "float_feature.npy"), 'rb') as f:
            self.f_feat = np.load(f)
        with open(os.path.join(path, "int_feature.npy"), 'rb') as f:
            self.i_feat = np.load(f)
