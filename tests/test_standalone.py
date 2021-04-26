import numpy as np
import argparse
import time
import multiprocessing
import graphmix

def arrive_and_leave():
    for i in range(4):
        comm = graphmix.Client(graphmix.default_server_port + i)

def test(args):
    arrive_and_leave()
    comm = graphmix.Client(graphmix.default_server_port)
    query = comm.pull_graph()
    graph = comm.wait(query)
    print(comm.meta)
    print(graph)
    print("CHECK OK")

def server_init(server):
    server.add_sampler(graphmix.sampler.LocalNode, batch_size=128)
    server.is_ready()
    time.sleep(1)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/stand_alone.yml")
    args = parser.parse_args()

    proc = multiprocessing.Process(target=test, args=[args])
    proc.start()
    graphmix.launcher(test, args, server_init=server_init)
