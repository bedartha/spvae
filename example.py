#! /usr/bin/env /home/bedartha/miniconda3/envs/sciprog/bin/python 
##! /usr/bin/env python

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def set_ddp(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    """set up DDP"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return None


def main(rank, world_size, nname):
    """runs the main code for training and validation"""
    set_ddp(rank, world_size)
    a = [0]
    for i in range(10000):
        a.append(a[i] + 1)
    dist.destroy_process_group()
    print(f"done with process {rank} of {world_size} on node {nname}")
    return None


if __name__ == "__main__":
    alloc_nodes = os.environ["SLURM_NODELIST"]
    print(f"List of allocated nodes: {alloc_nodes}")
    ngpus = torch.accelerator.device_count()
    ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    print("ngpus, ncpus per gpu", ngpus, ncpus)
    nname = os.environ["SLURMD_NODENAME"]
    nname = os.environ["SLURMD_NODENAME"]
    world_size = torch.cuda.device_count()
    print(f"Spawning {world_size} processes")
    mp.spawn(main, args=(world_size, nname), nprocs=world_size)
