import libc_graphmix as _C
import os
import os.path as osp
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

def start_server(shard, server_init, server_port=None):
    os.environ['GRAPHMIX_ROLE'] = "server"
    if server_port:
        os.environ["GRAPHMIX_SERVER_PORT"] = str(server_port)
    _C.init()
    shard.load_graph_shard(_C.rank())
    server = _C.start_server()
    server.init_meta(shard.meta)
    server.init_data(shard.f_feat, shard.i_feat, shard.edges)
    del shard
    print("GraphMix Server {} : data initialized at {}:{}".format(_C.rank(), _C.ip(), _C.port()))
    _C.barrier_all()
    server_init(server)
    _C.finalize()

def env_set_default(key, default_value):
    if key not in os.environ:
        os.environ[key] = str(default_value)
    return os.environ[key]

def start_scheduler():
    os.environ['GRAPHMIX_ROLE'] = "scheduler"
    _C.init()
    _C.finalize()

def start_worker(func, args):
    os.environ['GRAPHMIX_ROLE'] = "worker"
    _C.init()
    args.local_rank = _C.rank() % args.num_local_worker
    _C.barrier_all()
    func(args)
    _C.finalize()

def start_worker_standalone(func, args, local_rank):
    args.local_rank = local_rank
    func(args)

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

process_list = []

def launcher(target, args, server_init):
    # open setting file
    file_path = osp.abspath(osp.expanduser(osp.normpath(args.config)))
    with open(file_path) as setting_file:
        settings = yaml.load(setting_file.read(), Loader=yaml.FullLoader)

    # write environment variables
    for key, value in settings["env"].items():
        os.environ[str(key)] = str(value)

    # Set the server port to default if not specified
    if "GRAPHMIX_SERVER_PORT" not in os.environ.keys():
        server_port = default_server_port
    else:
        server_port = int(os.environ["GRAPHMIX_SERVER_PORT"])
    total_worker_num = int(env_set_default("GRAPHMIX_NUM_WORKER", 0))

    # the graph data path is relative to the setting file path
    raw_graph_data_path = settings["launch"]["data"]
    graph_data_path = osp.relpath(osp.expanduser(osp.normpath(raw_graph_data_path)))
    graph_data_path = osp.abspath(osp.join(osp.dirname(file_path), graph_data_path))
    print("GraphMix launcher : Using Graph Data from ", graph_data_path)

    # load graph and set the server number equal to the number of graph parts
    shard = Shard(graph_data_path)
    os.environ['GRAPHMIX_NUM_SERVER'] = str(shard.meta["num_part"])

    # get local job number
    args.num_local_worker = int(settings["launch"]["worker"])
    args.num_local_server = int(settings["launch"]["server"])
    if args.num_local_server > shard.meta["num_part"]:
        raise ValueError("Launcher server number should not be larger than graph partition")

    # launch workers
    for i in range(args.num_local_worker):
        if total_worker_num == 0:
            proc = multiprocessing.Process(target=start_worker_standalone, args=[target, args, i])
        else:
            proc = multiprocessing.Process(target=start_worker, args=[target, args])
        process_list.append(proc)

    # launch servers
    for i in range(args.num_local_server):
        # if launch multiple server on one node, use different ports
        proc = multiprocessing.Process(target=start_server, args=[shard, server_init, server_port + i])
        process_list.append(proc)

    # launch scheduler
    if settings["launch"]["scheduler"] != 0:
        proc = multiprocessing.Process(target=start_scheduler)
        process_list.append(proc)

    # wait until all process finish
    for proc in process_list:
        proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
