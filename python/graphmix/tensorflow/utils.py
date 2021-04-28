import tensorflow as tf
import numpy as np

def mp_matrix(graph, use_original_gcn_norm=False):
    graph.convert2coo()
    norm = graph.gcn_norm(use_original_gcn_norm)
    indices = np.vstack((graph.edge_index[1], graph.edge_index[0])).T
    shape = np.array([graph.num_nodes, graph.num_nodes], dtype=np.int64)
    mp_val = tf.compat.v1.SparseTensorValue(indices, norm, shape)
    return mp_val
