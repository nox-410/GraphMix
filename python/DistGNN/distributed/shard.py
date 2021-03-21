import libc_GNN as _C
import libc_PS as _PS

import numpy as np


class Shard():
    def __init__(self, path, shard_id):
        pass

    def upload(self, sync=True):
        ps_upload(
            self.graph.x, self.graph.y,
            self._internal.indptr, self._internal.indices, self._internal.nodes_from
        )
        if sync:
            _PS.barrier()

def _load_graph_shard(path, rank, nrank):
    import os, yaml
    with open(os.path.join(path, "meta.yml"), 'rb') as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    path = os.path.join(path, "part{}".format(rank))
    with open(os.path.join(path, "edge.npz"), 'rb') as f:
        data = np.load(f)
        edges = []
        for i in range(nrank):
            edges.append(data.get("edge_"+str(i)))
    with open(os.path.join(path, "data.npz"), 'rb') as f:
        data = np.load(f)
        x = data.get("x")
        y = data.get("y")
    return x, y, edges, meta
