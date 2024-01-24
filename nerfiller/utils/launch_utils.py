import threading
import time
import socket
from typing import List, Optional

import GPUtil

from nerfstudio.utils.scripts import run_command

def is_port_open(port: int):
    """Returns True if the port is open.

    Args:
        port: Port to check.

    Returns:
        True if the port is open, False otherwise.
    """
    try:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _ = sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False

def get_free_port(default_port: Optional[int] = None):
    """Returns a free port on the local machine. Try to use default_port if possible.

    Args:
        default_port: Port to try to use.

    Returns:
        A free port on the local machine.
    """
    if default_port is not None:
        if is_port_open(default_port):
            return default_port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port


def get_free_ports(num: int) -> List[int]:
    """Fund num free ports."""
    ports = set()
    while len(ports) < num:
        port = get_free_port()
        if port not in ports:
            ports.add(port)
    return list(ports)


def launch_experiments(
    jobs,
    dry_run: bool = False,
    gpu_ids: Optional[List[int]] = None,
    gpus_per_job: int = 1,
    delay_seconds: float = 5.0,
):
    """Launch the experiments.
    Args:
        jobs: list of commands to run
        dry_run: if True, don't actually run the commands
        gpu_ids: list of gpu ids that we can use. If none, we can use any
        delay_seconds: How long to wait before dispatching the next command.
    """

    start_time = time.time()

    num_jobs = len(jobs)
    while jobs:
        # get GPUs that capacity to run these jobs
        gpu_devices_available = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1)
        print("-" * 80)
        print("Available GPUs: ", gpu_devices_available)
        if gpu_ids:
            print("Restricting to subset of GPUs: ", gpu_ids)
            gpu_devices_available = [gpu for gpu in gpu_devices_available if gpu in gpu_ids]
            print("Using GPUs: ", gpu_devices_available)
        print("-" * 80)

        if len(gpu_devices_available) == 0:
            print("No GPUs available, waiting 10 seconds...")
            time.sleep(10)
            continue

        # thread list
        threads = []
        while gpu_devices_available and jobs and len(gpu_devices_available) >= gpus_per_job:
            gpu_str = ""
            for i in range(gpus_per_job):
                gpu = gpu_devices_available.pop(0)
                if i == 0:
                    gpu_str += str(gpu)
                else:
                    gpu_str += "," + str(gpu)
            command = f"CUDA_VISIBLE_DEVICES={gpu_str} " + jobs.pop(0)

            def task():
                print(f"Command:\n{command}\n")
                if not dry_run:
                    _ = run_command(command, verbose=False)
                # print("Finished command:\n", command)

            threads.append(threading.Thread(target=task))
            threads[-1].start()

            if not dry_run:
                # wandb/tensorboard naming or ports might be messed up w/o this
                time.sleep(delay_seconds)

        # wait for all threads to finish
        for t in threads:
            t.join()

        # print("Finished all threads")
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"Finished all {num_jobs} jobs in {minutes} minutes")
