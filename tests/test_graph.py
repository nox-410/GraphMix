import graphmix
import numpy as np
import scipy.sparse as sp

if __name__ =='__main__':
    cora = graphmix.dataset.load_dataset("Cora")
    cora.y = cora.y.reshape([-1, 1])
    graph = cora.graph
    graph.f_feat = cora.x
    graph.i_feat = cora.y
    print("Node : ", graph.num_nodes, " Edge : ", graph.num_edges)
    graph.tag = 1
    graph.type = graphmix.sampler.GraphSage
    assert graph.tag == 1 and graph.type == graphmix.sampler.GraphSage
    print("Check type and tag ok")

    graph.f_feat = np.empty([graph.num_nodes, 128])
    graph.i_feat = np.empty([graph.num_nodes, 128])
    graph.extra = np.empty([graph.num_nodes, 128])
    assert graph.f_feat.shape == (graph.num_nodes, 128)
    assert graph.i_feat.shape == (graph.num_nodes, 128)
    graph.f_feat = cora.x
    graph.i_feat = cora.y
    assert np.all(cora.x == graph.f_feat)
    assert np.all(cora.y == graph.i_feat)
    print("Check feature ok")

    for i in range(5):
        graph.convert2csr()
        graph.convert2coo()
    adj = sp.coo_matrix((np.ones(graph.num_edges), graph.edge_index), shape=(graph.num_nodes, graph.num_nodes))
    csradj = adj.tocsr()
    for i in range(5):
        graph.convert2coo()
        graph.convert2csr()
    assert np.all(csradj.indptr==graph.edge_index[0])
    assert np.all(csradj.indices==graph.edge_index[1])

    csradj += sp.identity(graph.num_nodes)
    graph.add_self_loop()
    graph.convert2csr()
    assert np.all(csradj.indptr==graph.edge_index[0])
    assert np.all(sorted(csradj.indices)==sorted(graph.edge_index[1]))
    print("Check edge ok")
