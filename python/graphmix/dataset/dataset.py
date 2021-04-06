import libc_graphmix as _C

import os
import numpy as np

class PlanetoidDataset():
    def __init__(self, root, name):
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        self.graph = _C.Graph(
            edge_index=data.edge_index.numpy(),
            num_nodes=data.num_nodes
        )
        self.x = data.x.numpy()
        self.y = data.y.numpy()
        self.train_mask = data.train_mask.numpy()
        self.num_classes = dataset.num_classes

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
