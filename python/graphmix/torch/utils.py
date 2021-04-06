import torch
import numpy as np

def mp_matrix(graph, device, use_original_gcn_norm=False):
    graph.convert2coo()
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
