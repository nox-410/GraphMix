import libc_graphmix as _C

import os
import numpy as np
import pickle

from .utils import download_url, process_graph

# class PlanetoidDataset():
#     def __init__(self, root, name):
#         from torch_geometric.datasets import Planetoid
#         dataset = Planetoid(root=root, name=name)
#         data = dataset[0]
#         self.graph = _C.Graph(
#             edge_index=data.edge_index.numpy(),
#             num_nodes=data.num_nodes
#         )
#         self.x = data.x.numpy()
#         self.y = data.y.numpy()
#         self.train_mask = data.train_mask.numpy()
#         self.num_classes = dataset.num_classes
#         self.dt = PlanetoidDataset1(root, name)

class PlanetoidDataset():
    def __init__(self, root, dataset_name):
        dataset_name = dataset_name.lower()
        assert dataset_name in ["cora", "pubmed", "citeseer"]
        super().__init__()
        # url = 'https://gitee.com/TMACzza/planetoid_datasets/raw/master'
        url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        items = []
        for name in names:
            file_url = "{}/ind.{}.{}".format(url, dataset_name, name)
            file = download_url(file_url, root)
            if name == "test.index":
                with open(file, "r") as f:
                    idx = f.read().split('\n')[:-1]
                    idx = list(map(int, idx))
                    items.append(idx)
            else:
                with open(file, "rb") as f:
                    out = pickle.load(f, encoding='latin1')
                    items.append(out)
        x, tx, allx, y, ty, ally, graph, test_index = items
        sorted_test_index = sorted(test_index)
        self.x = np.concatenate([allx.todense(), tx.todense()])
        self.y = np.concatenate([ally, ty]).argmax(1)
        self.x[test_index] = self.x[sorted_test_index]
        self.y[test_index] = self.y[sorted_test_index]
        self.graph = _C.Graph(
            edge_index=process_graph(graph),
            num_nodes=len(self.y)
        )
        self.train_mask = np.zeros(len(self.y), dtype=np.long)
        self.train_mask[0:len(y)] = 1 # train
        # self.mask[-1000:0] = 2 # test
        self.num_classes = y.shape[1]

class RedditDataset():
    def __init__(self, root):
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root=root)
        data = dataset[0]
        self.graph = _C.Graph(
            edge_index=data.edge_index.numpy(),
            num_nodes=data.num_nodes
        )
        self.x = data.x.numpy()
        self.y = data.y.numpy()
        self.train_mask = data.train_mask.numpy()
        self.num_classes = dataset.num_classes

class OGBDataset():
    def __init__(self, root, name):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=root)
        data = dataset[0]
        self.graph = _C.Graph(
            edge_index=data.edge_index.numpy(),
            num_nodes=data.num_nodes
        )
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        self.x = data.x.numpy()
        self.y = data.y.numpy().squeeze()
        self.train_mask = np.zeros(data.num_nodes, np.bool)
        self.train_mask[train_idx] = True
        self.num_classes = dataset.num_classes

def get_dataset_path(dataset_name):
    if "HOME" not in os.environ.keys():
        raise Exception("$HOME environ not set, cannot find datasetroot.")
    home_path = os.environ["HOME"]
    dataset_root = os.path.join(home_path, ".graphmix_dataset")
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)
    dataset_path = os.path.join(dataset_root, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    return dataset_path

#building the dataset using numpy
def load_dataset(name):
    root = get_dataset_path(name)
    if name=="Cora" or name=="PubMed":
        dataset = PlanetoidDataset(root, name)
    elif name=="Reddit":
        dataset = RedditDataset(root)
    elif name=="ogbn-products" or name=="ogbn-papers100M":
        dataset = OGBDataset(root, name)
    else:
        raise NotImplementedError
    return dataset

__all__ = ['load_dataset']
