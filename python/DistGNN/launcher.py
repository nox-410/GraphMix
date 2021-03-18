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

def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)

process_list = []

def launcher(args, target):
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(target=start_process, args=[value, args, target])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
