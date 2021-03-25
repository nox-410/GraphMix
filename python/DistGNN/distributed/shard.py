import libc_PS as _PS

import numpy as np
import os, yaml

class Shard():
    def __init__(self, path, shard_idx=-1):
        self.path = path
        self._load_meta()
        self.offset = self.meta["partition"]["offset"][shard_idx]
        self.num_nodes = self.meta["partition"]["nodes"][shard_idx]
        self.index = np.arange(self.offset, self.offset + self.num_nodes)
        if shard_idx >= 0:
            self._load_graph_shard(shard_idx)

    def _load_meta(self):
        with open(os.path.join(self.path, "meta.yml"), 'rb') as f:
            self.meta = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _load_graph_shard(self, shard_idx):
        assert shard_idx >= 0
        path = os.path.join(self.path, "part{}".format(shard_idx))
        with open(os.path.join(path, "graph.npy"), 'rb') as f:
            self.edges = np.load(f)
        with open(os.path.join(path, "float_feature.npy"), 'rb') as f:
            self.f_feat = np.load(f)
        with open(os.path.join(path, "int_feature.npy"), 'rb') as f:
            self.i_feat = np.load(f)

    def create_server(self):
        handler = _PS.start_server()
        offset = self.meta["partition"]["offset"]
        offset.append(self.meta["node"])
        handler.init_meta(self.meta["float_feature"], self.meta["int_feature"], offset)
        handler.init_data(self.f_feat, self.i_feat, self.edges)
        print("Server {} data initialized at {}:{}".format(_PS.rank(), _PS.ip(), _PS.port()))
        return handler

    def init_worker(self, target_server):
        handler = _PS.get_handle()
        offset = self.meta["partition"]["offset"]
        offset.append(self.meta["node"])
        handler.init_meta(self.meta["float_feature"], self.meta["int_feature"], offset, target_server)
        print("Worker {} start at {}:{} === Server {}".format(_PS.rank(), _PS.ip(), _PS.port(), target_server))
