import libc_graphmix as _C

import os

# when launch an async server function, a waitobject is returned
    # use result = Client.wait(waitobj) to wait and get the data
class _WaitObject():
    def __init__(self, query, is_graph_query=True, pack=None):
        self.query = query
        self.is_graph_query = is_graph_query
        self.pack = pack

# wrapper of the C++ class
class Client():
    def __init__(self, port=None):
        # detect whether in stand alone mode
        nw = "GRAPHMIX_NUM_WORKER"
        if nw not in os.environ or os.environ[nw] == '0':
            self.stand_alone = True
        else:
            self.stand_alone = False
        if self.stand_alone:
            if port is None:
                raise RuntimeError("A port should be specified in standalone mode, \
                                    try 'graphmix.default_server_port'")
            self.comm = _C.creat_client(port)
        else:
            self.comm = _C.get_client()

    # Get the information dict of graph, which is the same as meta.yml file
    @property
    def meta(self):
        return self.comm.meta

    def rank(self):
        if self.stand_alone:
            self._handle_error("rank")
        else:
            return _C.rank()

    # get the number of graph servers
    def num_server(self):
        return self.comm.meta["num_part"]

    def num_worker(self):
        if self.stand_alone:
            self._handle_error("num_worker")
        else:
            return _C.num_worker()

    # Barrier all the worker
    def barrier(self):
        if self.stand_alone:
            self._handle_error("barrier")
        else:
            return _C.barrier()

    # Barrier both workers and servers
    # Server should call this too
    def barrier_all(self):
        if self.stand_alone:
            self._handle_error("barrier_all")
        else:
            return _C.barrier_all()

    def _handle_error(self, method):
        raise RuntimeError(
            "method Client.'{}' is not supported in standalone client mode".format(method)
        )

    def pull_graph(self, *sampler):
        if len(sampler) == 1 and type(sampler[0]) in (list, tuple):
            sampler = sampler[0]
        query = self.comm.pull_graph(*sampler)
        waitobj = _WaitObject(query)
        return waitobj

    # get a list of node data
    # can only be used in none-standalone mode
    def pull_node(self, node_ids):
        if self.stand_alone:
            _handle_error("pull_node")
        pack = _C.NodePack()
        query = self.comm.pull_node(node_ids, pack)
        waitobj = _WaitObject(query, False, pack)
        return waitobj

    def wait(self, waitobj):
        assert type(waitobj) is _WaitObject
        if waitobj.is_graph_query:
            graph = self.comm.resolve(waitobj.query)
            return graph
        else:
            self.comm.wait(waitobj.query)
            return waitobj.pack
