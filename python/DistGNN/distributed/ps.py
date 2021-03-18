import os
import ctypes
import numpy as np
import threading
import libc_PS as _PS

class PS:
    rank = None
    nrank = None
    offset = 0
    feature_len = None
    communicator = None

def ps_init():
    PS.rank = _PS.rank()
    PS.nrank = _PS.nrank()
    PS.communicator = _PS.get_handle()

#-------------------------------------------------------------------------------
# args can be both integer or array
def ps_node_id(node, node_from):
    return PS.nrank * node + node_from

def ps_node_feat_id(node, node_from):
    node_id = ps_node_id(node, node_from)
    return PS.offset + 2 * node_id

def ps_node_edge_id(node, node_from):
    node_id = ps_node_id(node, node_from)
    return PS.offset + 2 * node_id + 1

#-------------------------------------------------------------------------------
def ps_upload(x, y, indptr, indices, nodes_from):
    num_nodes = x.shape[0]
    PS.feature_len = x.shape[1]
    all_pushed_id = []
    #upload feature, label, degree (aggregated into one)
    degree = indptr[1:] - indptr[:-1]
    #feat_id_arr = np.empty(num_nodes, dtype=np.long)
    feat_length_arr = np.repeat(x.shape[1] + 2, num_nodes)
    feat_data_arr = np.concatenate([x, y.reshape(-1,1), degree.reshape(-1,1)], axis=1).astype(np.float32)
    #upload edge info (2 * degree)
    #edge_id_arr = np.empty(num_nodes, dtype=np.long)
    edge_length_arr = degree * 2
    edge_data_arr = np.concatenate([indices, nodes_from]).reshape(2, -1).T
    edge_data_arr = np.ascontiguousarray(edge_data_arr).astype(np.float32)

    feat_id_arr = ps_node_feat_id(np.arange(num_nodes), PS.rank)
    edge_id_arr = ps_node_edge_id(np.arange(num_nodes), PS.rank)
    query1 = PS.communicator.push_data(feat_id_arr, feat_data_arr, feat_length_arr)
    query2 = PS.communicator.push_data(edge_id_arr, edge_data_arr, edge_length_arr)
    PS.communicator.wait(query1)
    PS.communicator.wait(query2)

def ps_download(nodes_id, nodes_from, feature_only=False):
    num_nodes = len(nodes_id)
    feat_id_arr = ps_node_feat_id(nodes_id, nodes_from)
    feat_id_arr = np.array(list(feat_id_arr), dtype=np.long)
    feat_data_arr = np.empty(shape=[num_nodes, PS.feature_len + 2]).astype(np.float32)
    feat_length_arr = np.repeat(PS.feature_len + 2, num_nodes)
    #from time import time
    #start = time()
    query = PS.communicator.pull_data(feat_id_arr, feat_data_arr, feat_length_arr)
    PS.communicator.wait(query)
    #print("Pull_Data", time() - start, num_nodes)
    feature = feat_data_arr[:,:-2]
    label = feat_data_arr[:,-2:-1].reshape(-1).astype(np.int32)
    degree = feat_data_arr[:,-1:].reshape(-1).astype(np.long)
    if feature_only:
        return feature, label, degree
    edge_id_arr = ps_node_edge_id(nodes_id, nodes_from)
    edge_id_arr = np.array(list(edge_id_arr), dtype=np.long)
    edge_length_arr = degree * 2
    edge_data_arr = np.empty(shape=[edge_length_arr.sum()]).astype(np.float32)
    query = PS.communicator.pull_data(edge_id_arr, edge_data_arr, edge_length_arr)
    PS.communicator.wait(query)
    #print("Pull_Data2", time() - start)
    edge_data_arr = edge_data_arr.reshape(-1, 2).T.astype(np.long)
    edge_data_arr = np.ascontiguousarray(edge_data_arr)
    return feature, label, degree, edge_data_arr
