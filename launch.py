#!/usr/bin/python3
import os
import sys
import socket
import random
import argparse
import subprocess
import time

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _get_rand_port():
    hour = time.time() // 3600
    random.seed(int(hour))
    return random.randrange(40000, 60000)


def init_workdir():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher')
    parser.add_argument('--launch', type=str, default='projects/UNINEXT/train_net.py',
                        help='Specify launcher script.')
    parser.add_argument('--dist', type=int, default=1,
                        help='Whether start by torch.distributed.launch.')
    parser.add_argument('--np', type=int, default=8,
                        help='number of (GPU) processes per node')
    parser.add_argument('--nn', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--port', type=int, default=-1,
                        help='master port for communication')
    parser.add_argument('--worker_rank', type=int, default=0)
    parser.add_argument('--master_address', type=str)
    args, other_args = parser.parse_known_args()

    # change to current dir
    prj_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(prj_dir)
    init_workdir()

    # Get training info
    master_address = args.master_address
    num_processes_per_worker = args.np
    num_workers = args.nn
    worker_rank = args.worker_rank

    # Get port
    if args.port > 0:
        master_port = args.port
    elif num_workers == 1:
        master_port = _find_free_port()
    else:  # This reduce the conflict possibility, but the port availablity is not guaranteed.
        master_port = _get_rand_port()


    if args.dist >= 1:
        print(f'Start {args.launch} by torch.distributed.launch with port {master_port}!', flush=True)
        cmd = f'python3 {args.launch}\
                --num-gpus={num_processes_per_worker}'
        if num_workers > 1:
            # multi-machine
            assert master_address is not None
            dist_url = "tcp://" + str(master_address) + ":" + str(master_port)
            cmd += f" --num-machines={num_workers}\
                    --machine-rank={worker_rank}\
                    --dist-url={dist_url}"
    else:
        print(f'Start {args.launch}!', flush=True)
        cmd = f'python3 {args.launch}'
        # $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}'
    for argv in other_args:
        cmd += f' {argv}'
    print("==> Run command: " + cmd)
    exit_code = subprocess.call(cmd, shell=True)
    sys.exit(exit_code)
