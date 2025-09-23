#! /usr/bin/env python

import os
import time

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
verbose = True
# file info
HOMEDIR = "/home/bedartha/"
DATADIR = "public/datasets/for_model_development/weatherbench2/era5/"
ARRNAME = "1959-2023_01_10-6h-64x32_equiangular_conservative_ZLEVS_T2M.zarr"

# params
EPOCHS = 5
BATCH_SIZE = 8 #// world_size
IN_CHANNELS = 4
NUM_WORKERS = 32
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 0.00001
SAVE_EVERY = 5
## for embedding
INPUT_SIZE = (64, 32)
PATCH_SIZE = (4, 2)
NUM_PATCHES = int((INPUT_SIZE[0] / PATCH_SIZE[0]) * (INPUT_SIZE[1] / PATCH_SIZE[1]))
EMBED_DIM = 10
PATCH_DROPOUT = 0.1
VAE_LATENT_DIM = 128
SP_ENC_LATENT_DIMS = [512, 256]
SP_DEC_LATENT_DIMS = [256, 512, 1024]
SP_MLP_DIMS = [64, 64]
SP_N_HEADS = [5, 5]
SP_N_TRNFR_LAYERS = [7, 7]
SP_DROPOUTS = [0.1, 0.1]


def set_ddp(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    """set up DDP"""
    # dist.init_process_group("nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return None


def get_dataloader(path_to_zarr, to_tensor, partition, batch_size, num_workers,
                   shuffle=False, verbose=True):
    """
    Loads the WeatherBenceh Dataset class and prepares the Dataloader
    -----------------------------------------------------------------
    """
    if verbose: print("initializing WeatherBenchDataset class ...")
    wb = WeatherBenchDataset(path_to_zarr, to_tensor,  partition)
    if verbose: print("set up dataloader ...")
    sampler = DistributedSampler(wb)
    loader = DataLoader(wb, batch_size=batch_size, num_workers=num_workers,
                        sampler=sampler, shuffle=shuffle)
    return loader


def set_model(local_rank):
    """set up model and set to device"""
    # initalize model
    if verbose: print("set up model ...")
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
                  input_size=INPUT_SIZE
            )
    # more ddp stuff
    if verbose: print("model to device ...")
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    spvae = spvae.to('cuda:' + str(local_rank))
    # spvae = spvae.to(local_rank)
    if verbose: print("model to DDP ...")
    spvae = DDP(spvae, device_ids=[local_rank], find_unused_parameters=True)
    # spvae = DDP(spvae, device_ids=[local_rank])

    return spvae


def set_optimizer(spvae):
    """set up optimizer and LR scheduler"""
    if verbose: print("set up optimizer and LR scheduler ...")
    opt = torch.optim.Adam(spvae.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LEARNING_RATE_MIN)
    scaler = torch.amp.GradScaler("cuda")
    return opt, scaler


def train(train_loader, model, optim, local_rank, save_every, scaler, verbose):
    """train the model"""
    if verbose: print("train the VAE ...")
    train_loss = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()
        model.train()
        for X in tqdm(train_loader, disable=(not verbose)):
            X = X.to('cuda:' + str(local_rank))
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                X_, kl = model(X)
                X_ = X_.to('cuda:' + str(local_rank))
                kl = kl.to('cuda:' + str(local_rank))
                loss = ((X - X_)**2).sum() + kl #spvae.vae.encoder.kl
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
        train_loss[epoch] = loss
        epoch_end_time = time.perf_counter()
        if local_rank == 0:
            if epoch % save_every == 0:
                FNAME = f"{HOMEDIR}data/scratch/chkpt_epoch{epoch}_rank{local_rank}.tar"
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict":model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "loss": loss
                            },
                        FNAME
                        )
            if verbose: print("saved checkpoint to %s" %FNAME)
    return train_loss


def validate(val_loader, model, local_rank, verbose):
    """validate the model"""
    val_loss = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        if verbose: print("validate...")
        model.eval()
        with torch.no_grad():
            for X in tqdm(val_loader, disable=(not verbose)):
                X = X.to('cuda:' + str(local_rank))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    X_, kl = model(X)
                    X_ = X_.to('cuda:' + str(local_rank))
                    kl = kl.to('cuda:' + str(local_rank))
                    loss = ((X - X_)**2).sum() + kl #spvae.vae.encoder.kl
        val_loss[epoch] = loss
    return val_loss


def main(rank, world_size, total_epochs, save_every):
    """runs the main code for training and validation"""
    set_ddp(rank, world_size)

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
    train_loss = train(trn_dl, spvae, opt, rank, save_every, scaler, verbose)
    val_loss = validate(val_dl, spvae, rank, verbose)
    if rank == 0:
        if verbose: print("saving loss arrays to NPZ file in scratch ...")
        FNAME = "/home/bedartha/data/scratch/spvae_loss.npz"
        np.savez(FNAME,
                 train_loss=train_loss / len(trn_dl),
                 val_loss=val_loss / len(val_dl))
        if verbose: print("saved to: %s" % FNAME)
    dist.destroy_process_group()
    return None


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Spawning {world_size} processes")
    mp.spawn(main, args=(world_size, EPOCHS, SAVE_EVERY), nprocs=world_size)
