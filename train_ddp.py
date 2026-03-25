#! /usr/bin/env /home/bedartha/miniconda3/envs/sciprog/bin/python 
##! /usr/bin/env python

import os
import time
import datetime as dt

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from dataset import WeatherBenchDataset
from model import SPVAE



# general params
verbose = False
# file info
HOMEDIR = "/home/bedartha/"
DATADIR = "public/datasets/for_model_development/weatherbench2/era5/"
# ARRNAME = "1979-2022_01_10-6h-64x32_equiangular_conservative_MWE.zarr"
ARRNAME = "1979-2022_01_10-6h-240x121_equiangular_with_poles_conservative_MWE.zarr"

# params
EPOCHS = 50
BATCH_SIZE = 8 #// world_size
IN_CHANNELS = 20
NUM_WORKERS = 32
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 0.00001
SAVE_EVERY = 5
## for embedding
INPUT_SIZE = (240, 121)
OUTPUT_PADDING = [0, 1]
PATCH_SIZE = (16, 8)
NUM_PATCHES = int((INPUT_SIZE[0] / PATCH_SIZE[0])) * int((INPUT_SIZE[1] / PATCH_SIZE[1]))
EMBED_DIM = 48
PATCH_DROPOUT = 0.1
VAE_LATENT_DIM = 128
SP_ENC_LATENT_DIMS = [1024, 256]
SP_DEC_LATENT_DIMS = [256, 1024, NUM_PATCHES*IN_CHANNELS]
SP_MLP_DIMS = [64, 64]
SP_N_HEADS = [4, 2]
SP_N_TRNFR_LAYERS = [4, 4]
SP_DROPOUTS = [0.1, 0.1]


def set_ddp(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    """set up DDP"""
    # os.environ["MASTER_ADDR"] = master_addr
    # os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return None


def get_dataloader(path_to_zarr, to_tensor, partition, batch_size, num_workers,
                   shuffle=False, verbose=True):
    """
    Loads the WeatherBenceh Dataset class and prepares the Dataloader
    -----------------------------------------------------------------
    """
    wb = WeatherBenchDataset(path_to_zarr, to_tensor,  partition)
    sampler = DistributedSampler(wb)
    loader = DataLoader(wb, batch_size=batch_size, num_workers=num_workers,
                        sampler=sampler, shuffle=shuffle)
    return loader


def set_model(local_rank):
    """set up model and set to device"""
    # initalize model
    spvae = SPVAE(
                  embed_dim=EMBED_DIM, patch_size=PATCH_SIZE,
                  num_patches=NUM_PATCHES, patch_dropout=PATCH_DROPOUT,
                  in_channels=IN_CHANNELS, keep_channels=False,
                  vae_latent_dim=VAE_LATENT_DIM,
                  sp_enc_latent_dims=SP_ENC_LATENT_DIMS,
                  sp_dec_latent_dims=SP_DEC_LATENT_DIMS,
                  sp_mlp_dims=SP_MLP_DIMS,
                  sp_n_heads=SP_N_HEADS,
                  sp_n_trnfr_layers=SP_N_TRNFR_LAYERS,
                  sp_dropouts=SP_DROPOUTS,
                  batch_size=BATCH_SIZE,
                  input_size=INPUT_SIZE,
                  output_padding=[0, 1]
            )
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    spvae = spvae.to('cuda:' + str(local_rank))
    spvae = DDP(spvae, device_ids=[local_rank], find_unused_parameters=True)

    return spvae


def set_optimizer(spvae):
    """set up optimizer and LR scheduler"""
    opt = torch.optim.Adam(spvae.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LEARNING_RATE_MIN)
    scaler = torch.amp.GradScaler("cuda")
    return opt, scaler


def train(train_loader, val_loader, model, optim, local_rank, save_every, scaler, verbose,
          nname, global_rank):
    """train the model"""
    train_loss = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} for process {local_rank} on node {nname}")
        print(f"global rank = {global_rank}; local rank = {local_rank}")
        print("train ...")
        model.train()
        for X in train_loader:
            X = X.to('cuda:' + str(local_rank))
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                X_, kl = model(X)
                X_ = X_.to('cuda:' + str(local_rank))
                kl = kl.to('cuda:' + str(local_rank))
                loss = ((X - X_)**2).sum() + kl
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
        train_loss[epoch] = loss
        print("validate ...")
        val_loss = np.zeros(EPOCHS)
        model.eval()
        with torch.no_grad():
            for X in val_loader:
                X = X.to('cuda:' + str(local_rank))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    X_, kl = model(X)
                    X_ = X_.to('cuda:' + str(local_rank))
                    kl = kl.to('cuda:' + str(local_rank))
                    loss = ((X - X_)**2).sum() + kl #spvae.vae.encoder.kl
        val_loss[epoch] = loss
        #save checkpoint
        if local_rank == 0 and global_rank == 0:
            print(f"saving model checkpoint after {EPOCH} epochs")
            print(f"since this process has rank {local_rank} on node {nname}")
            if epoch % save_every == 0:
                path = f"{HOMEDIR}data/scratch/" 
                tstamp = dt.datetime.today().strftime("%Y%m%d_%H%M%S")
                FNAME = f"chkpt_epoch{epoch}_rank{local_rank}_{tstamp}.tar"
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict":model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "loss": loss
                            },
                        path + FNAME
                        )
            if verbose: print("saved checkpoint to %s" %FNAME)
    return train_loss, val_loss


def main(rank, world_size, total_epochs, save_every, nname):
    """runs the main code for training and validation"""
    set_ddp(rank, world_size)
    global_rank = int(os.environ["SLURM_PROCID"])

    PATH_TO_ZARR = f"{HOMEDIR}{DATADIR}{ARRNAME}"
    trn_dl = get_dataloader(path_to_zarr=PATH_TO_ZARR, to_tensor=True,
                            partition="train",
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            shuffle=False, verbose=verbose)
    val_dl = get_dataloader(path_to_zarr=PATH_TO_ZARR, to_tensor=True,
                            partition="val",
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            shuffle=False, verbose=verbose)
    spvae = set_model(rank)
    opt, scaler = set_optimizer(spvae)
    tl, vl = train(trn_dl, val_dl, spvae, opt, rank, save_every, scaler, verbose,
                   nname, global_rank)
    # val_loss = validate(val_dl, spvae, rank, verbose, nname)
    if global_rank == 0 and rank == 0:
        print("saving loss arrays to NPZ file in scratch ...")
        print(f"since this process has rank {rank} on node {nname}")
        tstamp = dt.datetime.today().strftime("%Y%m%d_%H%M%S")
        FNAME = f"/home/bedartha/data/scratch/spvae_loss_{tstamp}.npz"
        np.savez(FNAME,
                 train_loss=tl / len(trn_dl),
                 val_loss=vl / len(val_dl))
        print("saved to: %s" % FNAME)
    dist.destroy_process_group()
    print(f"done with process {rank} of {world_size} on node {nname}")
    return None


if __name__ == "__main__":
    alloc_nodes = os.environ["SLURM_NODELIST"]
    nname = os.environ["SLURMD_NODENAME"]
    world_size = torch.cuda.device_count()
    print(f"spawning {world_size} processes on node {nname}")
    mp.spawn(main, args=(world_size, EPOCHS, SAVE_EVERY, nname), nprocs=world_size)
