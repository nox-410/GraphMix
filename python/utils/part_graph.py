import argparse
import sys, os
import numpy as np
import time
import yaml

from DistGNN.dataset import load_dataset

def part_graph(dataset_name, nparts, output_path):
    if os.path.exists(output_path):
        os.rmdir(output_path)
    os.mkdir(output_path)
    start = time.time()
    dataset = load_dataset(dataset_name)
    print("step1: load_dataset complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    partition = dataset.graph.part_graph(nparts)
    print("step2: partition graph complete, time cost {:.3f}s".format(time.time()-start))
    start = time.time()
    for i in range(nparts):
        part_dir = os.path.join(output_path, "part{}".format(i))
        os.mkdir(part_dir)
        edge_path = os.path.join(part_dir, "graph.npz")
        data_path = os.path.join(part_dir, "data.npz")
        float_feature = dataset.x.astype(np.float32)
        int_feature = np.vstack([dataset.y, dataset.train_mask]).T.astype(np.int32)

        with open(edge_path, 'wb') as f:
            np.savez(file=f, index=partition[i][0], edge=np.vstack(partition[i][1:3]))
        with open(data_path, 'wb') as f:
            np.savez(file=f, f=float_feature, i=int_feature)
    print("step3: save partitioned graph, time cost {:.3f}s".format(time.time()-start))
    part_meta = {
        "nodes" : [len(g[0]) for g in partition],
        "edges" : [len(g[1]) for g in partition],
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
    }
    edge_path = os.path.join(output_path, "meta.yml")
    with open(edge_path, 'w') as f:
        yaml.dump(meta, f, sort_keys=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--nparts", "-n", required=True)
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    output_path = str(args.path)
    nparts = int(args.nparts)
    dataset = str(args.dataset)
    output_path = os.path.join(output_path, dataset)
    part_graph(dataset, nparts, output_path)
