import numpy as np

import libc_graphmix as _C
from libc_graphmix import Graph

def shuffle(graph):
    perm = np.random.permutation(graph.num_nodes)
    rperm = np.argsort(perm) # reversed permutation
    return Graph(
        (perm[graph.edge_index[0]], perm[graph.edge_index[1]]),
        graph.num_nodes
    )

def dense_efficient(graph):
    return graph.num_edges / (graph.num_nodes ** 2)

def mp_matrix(graph, device, use_original_gcn_norm=False):
    norm = graph.gcn_norm(use_original_gcn_norm)
    import torch
    indices = np.vstack((graph.edge_index[1], graph.edge_index[0]))
    mp_mat = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.FloatTensor(norm),
        size=(graph.num_nodes, graph.num_nodes),
        device=device,
    )
    return mp_mat

def pick_edges(edge_index, index):
    return edge_index[0][index], edge_index[1][index]
