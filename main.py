"""
Train model using multi-node GPU parallelism

(c) 2026 Bedartha Goswami, bedartha.goswami@iiserpune.ac.in

"""


import os
import argparse

import train
from params import Params

import torch
import torch.multiprocessing as mp


def _parser():
    """Parse arguments"""
    doc_split = __doc__.split("(c) ")
    helpdoc, epilog = doc_split[0], doc_split[-1]
    parser = argparse.ArgumentParser(
                    prog=f'python {__file__.split('/')[-1]}',
                    description=helpdoc,
                    formatter_class=argparse.RawTextHelpFormatter,
                    epilog=f"(c) {epilog}")
    parser.add_argument("-i", "--in-file",
                        help="path to input data file"
                        )
    parser.add_argument("-o", "--out-dir",
                        help="path to output directory"
                        )
    parser.add_argument("-c", "--config-file",
                        help="path to config file"
                        )
    parser.add_argument("-n", "--name",
                        help="name of model run",
                        default="spvae_run"
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Control verbosity",
                        default=True
                        )
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    params = Params(args.config_file)

    alloc_nodes = os.environ["SLURM_NODELIST"]
    nodename = os.environ["SLURMD_NODENAME"]
    nprocs = torch.cuda.device_count()
    global_rank = int(os.environ["SLURM_PROCID"])

    print(f"spawning {nprocs} processes on node {nodename} " + \
            f"with global rank {global_rank}")

    jid = os.environ["SLURM_JOB_ID"]
    name = f"{args.name}_jobid_{jid}"
    dirname = f"{args.out_dir}{name}"
    if  global_rank == 0:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        print(f"created output directory: {dirname}")
        os.system(f"cp -v {args.config_file} {dirname}")


    mp.spawn(train.main,
             args=(nprocs, params, args, nodename, dirname),
             nprocs=nprocs
             )
