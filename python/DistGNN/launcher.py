import libc_PS as _PS
import os
import signal
import yaml
import multiprocessing

def start_process(settings, args, func):
    for key, value in settings.items():
        os.environ[key] = str(value)
    _PS.init()
    if os.environ['DMLC_ROLE'] == "server":
        _PS.start_server()
    elif os.environ['DMLC_ROLE'] == "worker":
        _PS.init()
        func(args)
    elif os.environ['DMLC_ROLE'] == "scheduler":
        _PS.init()
    else:
        raise ValueError("Unknown role", os.environ['DMLC_ROLE'])
    _PS.finalize()

def start_server():
    os.environ['DMLC_ROLE'] = "server"
    _PS.init()
    _PS.start_server()
    _PS.finalize()

def start_scheduler():
    os.environ['DMLC_ROLE'] = "scheduler"
    _PS.init()
    _PS.finalize()

def start_worker(func, args, num_local_worker):
    os.environ['DMLC_ROLE'] = "worker"
    _PS.init()
    local_rank = _PS.rank() % num_local_worker
    args.local_rank = local_rank
    func(args)
    _PS.finalize()

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

    num_local_worker = int(settings["launch"]["worker"])
    for i in range(num_local_worker):
        proc = multiprocessing.Process(target=start_worker, args=[target, args, num_local_worker])
        process_list.append(proc)

    for i in range(int(settings["launch"]["server"])):
        proc = multiprocessing.Process(target=start_server)
        process_list.append(proc)

    if settings["launch"]["scheduler"] != 0:
        proc = multiprocessing.Process(target=start_scheduler)
        process_list.append(proc)

    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.start()
    for proc in process_list:
        proc.join()
