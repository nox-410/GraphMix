import graphmix

import argparse
import sys, os
import numpy as np
import scipy.sparse as sp
import time
import yaml

def to_inductive(dataset):
    indices = np.where(dataset.train_mask == 1)[0]
    dataset.x = dataset.x[indices]
    dataset.y = dataset.y[indices]
    dataset.train_mask = dataset.train_mask[indices]
    adj_mat = sp.coo_matrix(
        (np.ones(dataset.graph.num_edges) ,dataset.graph.edge_index),
        shape=(dataset.graph.num_nodes, dataset.graph.num_nodes)
    ).tocsr()
    adj_mat = adj_mat.T[indices].T[indices].tocoo()
    dataset.graph = graphmix.Graph(np.vstack([adj_mat.row, adj_mat.col]), len(indices))
    return dataset

def part_graph(dataset_name, nparts, output_path):
    os.makedirs(os.path.expanduser(os.path.normpath(output_path)))
    start = time.time()
    dataset = graphmix.dataset.load_dataset(dataset_name)
    if args.inductive:
        dataset = to_inductive(dataset)
    print("step1: load_dataset complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    partition = dataset.graph.part_graph(nparts, random=args.random)
    print("step2: partition graph complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    float_feature = dataset.x.astype(np.float32)

    # process labels and train_mask
    int_feature = dataset.y
    if len(int_feature.shape) == 1:
        int_feature = int_feature.reshape(-1, 1)
    if args.nodeid:
        int_feature = np.concatenate([int_feature, np.arange(dataset.graph.num_nodes).reshape(-1, 1)], axis=1)
    int_feature = np.concatenate([int_feature, dataset.train_mask.reshape(-1, 1)], axis=1).astype(np.int32)

    for i in range(nparts):
        part_dict = partition[i]
        part_dir = os.path.join(output_path, "part{}".format(i))
        os.mkdir(part_dir)
        edge_path = os.path.join(part_dir, "graph.npy")
        float_feature_path = os.path.join(part_dir, "float_feature.npy")
        int_feature_path = os.path.join(part_dir, "int_feature.npy")
        with open(edge_path, 'wb') as f:
            np.save(f, np.vstack(part_dict["edges"]))
        with open(float_feature_path, 'wb') as f:
            np.save(f, float_feature[part_dict["orig_index"]])
        with open(int_feature_path, 'wb') as f:
            np.save(f, int_feature[part_dict["orig_index"]])
    print("step3: save partitioned graph, time cost {:.3f}s".format(time.time()-start))
    part_meta = {
        "nodes" : [len(part_dict["orig_index"]) for part_dict in partition],
        "edges" : [len(part_dict["edges"][0]) for part_dict in partition],
        "offset" : [part_dict["offset"] for part_dict in partition],
    }
    meta = {
        "name": dataset_name,
        "node": dataset.graph.num_nodes,
        "edge": dataset.graph.num_edges,
        "float_feature": float_feature.shape[1],
        "int_feature": int_feature.shape[1],
        "class": dataset.num_classes,
        "num_part": nparts,
        "partition": part_meta,
        "random" : args.random,
        "train_node" : int((dataset.train_mask==1).sum()),
        "eval_node" : int((dataset.train_mask==0).sum()),
        "test_node" : int((dataset.train_mask==2).sum()),
    }
    edge_path = os.path.join(output_path, "meta.yml")
    with open(edge_path, 'w') as f:
        yaml.dump(meta, f, sort_keys=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--nparts", "-n", required=True)
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--nodeid", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--inductive", action="store_true")
    args = parser.parse_args()
    output_path = str(args.path)
    nparts = int(args.nparts)
    dataset = str(args.dataset)
    output_path = os.path.join(output_path, dataset)
    part_graph(dataset, nparts, output_path)
