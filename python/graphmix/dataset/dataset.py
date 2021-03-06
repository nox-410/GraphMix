import libc_graphmix as _C

import os
import numpy as np
import scipy.sparse as sp
import pickle
import json
from sklearn.preprocessing import StandardScaler

from .utils import download_url, process_graph, extract_zip

class PlanetoidDataset():
    def __init__(self, root, dataset_name, public_split=False):
        assert dataset_name in ["Cora", "PubMed"]
        self.name = dataset_name
        dataset_name = dataset_name.lower()
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
        self.train_mask = np.zeros(len(self.y), dtype=np.int32)
        self.train_mask[0:len(y)] = 1 # train
        if not public_split:
            eval_index = np.arange(len(y), len(y) + 500)
            self.train_mask[:] = 1
            self.train_mask[eval_index] = 0
            self.train_mask[sorted_test_index] = 2
        self.num_classes = y.shape[1]

class RedditDataset():
    def __init__(self, root):
        self.name = "Reddit"
        url = 'https://data.dgl.ai/dataset/reddit.zip'
        npz_file = os.path.join(root, 'reddit_data.npz')
        npz_graph_file = os.path.join(root, 'reddit_graph.npz')
        if not os.path.exists(npz_file) or not os.path.exists(npz_graph_file):
            zip_file = download_url(url, root)
            extract_zip(zip_file, root)
            os.unlink(zip_file)
        data = np.load(npz_file)

        self.x = data['feature']
        self.y = data['label']
        split = data['node_types']
        adj = sp.load_npz(npz_graph_file)
        edge_index = np.stack([adj.row, adj.col])
        self.num_classes = int(self.y.max() + 1)
        self.train_mask = np.zeros(len(self.y), dtype=np.int32)
        self.train_mask[split == 1] = 1
        self.train_mask[split == 3] = 2
        self.graph = _C.Graph(
            edge_index=edge_index,
            num_nodes=len(self.y)
        )

class OGBDataset():
    def __init__(self, root, name):
        self.name = name
        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset(name=name, root=root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        num_nodes=data[1].shape[0]
        edge = data[0]["edge_index"]
        if name == "ogbn-arxiv":
            #convert ogbn-arxiv to undirected graph
            edge = np.concatenate([edge, edge[[1, 0]]], axis=1)
        self.graph = _C.Graph(
            edge_index=edge,
            num_nodes=num_nodes
        )
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        self.x = data[0]["node_feat"]
        self.y = data[1].squeeze()
        self.train_mask = np.zeros(num_nodes, np.int32)
        self.train_mask[train_idx] = 1
        self.train_mask[test_idx] = 2
        self.num_classes = dataset.num_classes

class GraphSaintDataset():
    def __init__(self, root, name):
        self.name = name
        if name=="Yelp":
            adj_full_id = '1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA'
            feats_id = '1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM'
            class_map_id = '1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU'
            role_id = '1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_'
        elif name=="Flickr":
            adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
            feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
            class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
            role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'
        elif name=="Reddit2":
            adj_full_id = '1sncK996BM5lpuDf75lDFqCiDZyErc1c2'
            feats_id = '1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3'
            class_map_id = '1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u'
            role_id = '1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A'

        from google_drive_downloader import GoogleDriveDownloader as gdd

        path = os.path.join(root, 'adj_full.npz')
        gdd.download_file_from_google_drive(adj_full_id, path)
        path = os.path.join(root, 'feats.npy')
        gdd.download_file_from_google_drive(feats_id, path)
        path = os.path.join(root, 'class_map.json')
        gdd.download_file_from_google_drive(class_map_id, path)
        path = os.path.join(root, 'role.json')
        gdd.download_file_from_google_drive(role_id, path)

        preprocess_class_map = os.path.join(root, 'processed.npy')
        if os.path.exists(preprocess_class_map):
            class_arr = np.load(preprocess_class_map)
        else:
            class_map = json.load(open(os.path.join(root, 'class_map.json')))
            class_arr = [-1] * len(class_map)
            for i in range(len(class_map)):
                class_arr[i] = class_map[str(i)]
            class_arr = np.array(class_arr, dtype=np.int32)
            np.save(preprocess_class_map, class_arr)
        num_nodes = class_arr.shape[0]
        role_map = json.load(open(os.path.join(root, 'role.json')))
        f = np.load(os.path.join(root, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        self.graph = _C.Graph(
            edge_index=np.vstack([adj.row, adj.col]),
            num_nodes=num_nodes
        )
        self.train_mask = np.zeros(num_nodes, dtype=np.int32)
        self.train_mask[role_map['tr']] = 1
        self.train_mask[role_map['te']] = 2
        self.y = class_arr
        # follow graphsaint data preprocessing
        feats = np.load(os.path.join(root, 'feats.npy'))
        scaler = StandardScaler()
        scaler.fit(feats[role_map['tr']])
        self.x = scaler.transform(feats)
        if len(class_arr.shape) == 2:
            self.num_classes = class_arr.shape[1]
        else:
            self.num_classes = int(class_arr.max() + 1)

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
    elif name=="Yelp" or name=="Flickr" or name=="Reddit2":
        dataset = GraphSaintDataset(root, name)
    elif name=="ogbn-products" or name=="ogbn-papers100M" or name=='ogbn-arxiv':
        dataset = OGBDataset(root, name)
    else:
        raise NotImplementedError
    return dataset

__all__ = ['load_dataset']
