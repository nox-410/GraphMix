import libc_graphmix as _C
import os
import signal
import yaml
import multiprocessing

from .shard import Shard

envvar = [
"GRAPHMIX_VERBOSE",
"GRAPHMIX_WORKER_RECV_THREAD",
"GRAPHMIX_SERVER_RECV_THREAD",
"GRAPHMIX_WORKER_ZMQ_THREAD",
"GRAPHMIX_SERVER_ZMQ_THREAD",
"GRAPHMIX_SERVER_PORT",
"GRAPHMIX_PS_VAN_TYPE",
"GRAPHMIX_NUM_WORKER",
"GRAPHMIX_NUM_SERVER",
"GRAPHMIX_ROLE",
"GRAPHMIX_ROOT_URI",
"GRAPHMIX_ROOT_PORT",
"GRAPHMIX_NODE_HOST",
"GRAPHMIX_INTERFACE",
"GRAPHMIX_LOCAL",
]

default_server_port = 27777

def start_server(graph_data_path, server_init):
    os.environ['GRAPHMIX_ROLE'] = "server"
    _C.init()
    rank = _C.rank()
    server = Shard(graph_data_path, rank).create_server()
    server_init(server)
    _C.finalize()

def start_scheduler():
    os.environ['GRAPHMIX_ROLE'] = "scheduler"
    _C.init()
    _C.finalize()

def start_worker(func, args, graph_data_path):
    os.environ['GRAPHMIX_ROLE'] = "worker"
    _C.init()
    args.local_rank = _C.rank() % args.num_local_worker
    func(args)
    _C.finalize()

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

process_list = []

def launcher(target, args, server_init):
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for key, value in settings["env"].items():
        os.environ[str(key)] = str(value)
    if "GRAPHMIX_SERVER_PORT" not in os.environ.keys():
        server_port = default_server_port
    else:
        server_port = int(os.environ["GRAPHMIX_SERVER_PORT"])

    graph_data_path = os.path.abspath(settings["launch"]["data"])
    args.num_local_worker = int(settings["launch"]["worker"])
    args.num_local_server = int(settings["launch"]["server"])
    for i in range(args.num_local_worker):
        proc = multiprocessing.Process(target=start_worker, args=[target, args, graph_data_path])
        proc.start()
        process_list.append(proc)

    for i in range(args.num_local_server):
        # if launch multiple server on one node, use different ports
        os.environ["GRAPHMIX_SERVER_PORT"] = str(server_port + i)
        proc = multiprocessing.Process(target=start_server, args=[graph_data_path, server_init])
        proc.start()
        process_list.append(proc)

    if settings["launch"]["scheduler"] != 0:
        proc = multiprocessing.Process(target=start_scheduler)
        proc.start()
        process_list.append(proc)

    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
