import libc_graphmix as _C
import os
import signal
import yaml
import multiprocessing

from .shard import Shard

envvar = [
"PS_HEARTBEAT_TIMEOUT",
"PS_HEARTBEAT_INTERVAL",
"PS_RESEND",
"PS_VERBOSE",
"PS_WORKER_THREAD",
"PS_SERVER_THREAD",
"PS_DROP_MSG",
"DMLC_PS_VAN_TYPE",
"DMLC_NUM_WORKER",
"DMLC_ROLE",
"DMLC_PS_ROOT_URI",
"DMLC_PS_ROOT_PORT",
"DMLC_NODE_HOST",
"DMLC_INTERFACE",
"DMLC_LOCAL",
"DMLC_USE_KUBERNETES",
"DMLC_PS_WATER_MARK",
]

def start_server(graph_data_path):
    os.environ['DMLC_ROLE'] = "server"
    _C.init()
    rank = _C.rank()
    Shard(graph_data_path, rank).create_server()
    _C.barrier_all()
    _C.finalize()

def start_scheduler():
    os.environ['DMLC_ROLE'] = "scheduler"
    _C.init()
    _C.finalize()

def start_worker(func, args, num_local_worker, graph_data_path):
    os.environ['DMLC_ROLE'] = "worker"
    _C.init()
    local_rank = _C.rank() % num_local_worker
    target_server = _C.rank() // num_local_worker
    args.local_rank = local_rank
    shard = Shard(graph_data_path, -1)
    shard.init_worker(target_server)
    args.meta = shard.meta
    _C.barrier_all()
    func(args)
    _C.finalize()

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

process_list = []

def launcher(target, args):
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for key, value in settings["env"].items():
        os.environ[str(key)] = str(value)

    graph_data_path = os.path.abspath(settings["launch"]["data"])

    num_local_worker = int(settings["launch"]["worker"])
    for i in range(num_local_worker):
        proc = multiprocessing.Process(target=start_worker, args=[target, args, num_local_worker, graph_data_path])
        process_list.append(proc)

    for i in range(int(settings["launch"]["server"])):
        proc = multiprocessing.Process(target=start_server, args=[graph_data_path])
        process_list.append(proc)

    if settings["launch"]["scheduler"] != 0:
        proc = multiprocessing.Process(target=start_scheduler)
        process_list.append(proc)

    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.start()
    for proc in process_list:
        proc.join()
