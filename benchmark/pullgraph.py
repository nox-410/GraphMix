import numpy as np
import argparse
import threading
import time
import graphmix

max_thread = 1

def test(args):
    graphmix._C.barrier_all()
    rank = graphmix._C.rank()
    nrank = graphmix._C.num_worker()
    comm = graphmix._C.get_client()
    item_count = 0
    def pull_graph():
        while True:
            query = comm.pull_graph()
            graph = comm.resolve(query)
            nonlocal item_count
            item_count += graph.num_nodes

    def watch():
        nonlocal item_count
        start = time.time()
        while True:
            time.sleep(1)
            speed = item_count / (time.time() - start)
            print("speed : {} item/s".format(speed))
    task_list = [None for i in range(max_thread)]
    threading.Thread(target=watch).start()
    for i in range(max_thread):
        threading.Thread(target=pull_graph).start()
    time.sleep(1000)

def server_init(server):
    #server.init_cache(1, graphmix.cache.LFUOpt)
    for i in range(args.num_local_worker):
        server.add_sampler(graphmix.sampler.LocalNode, batch_size=500)
    #server.add_sampler(graphmix.sampler.RandomWalk, rw_head=128, rw_length=2)
    graphmix._C.barrier_all()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/test_config.yml")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)
