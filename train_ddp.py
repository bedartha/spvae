#! /usr/bin/env python

import os
import time

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from dataset import WeatherBenchDataset
from model import SPVAE


if __name__ == "__main__":
    # ddp params
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # general params
    verbose = False
    # file info
    HOMEDIR = "/home/bedartha/"
    DATADIR = "public/datasets/as_downloaded/weatherbench2/era5/"
    ARRNAME = "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

    # params
    EPOCHS = 10
    BATCH_SIZE = 4 #// world_size
    IN_CHANNELS = 4
    WORKERS = 17
    ## for embedding
    INPUT_SIZE = (64, 32)
    PATCH_SIZE = (4, 2)
    NUM_PATCHES = int(
            (
                INPUT_SIZE[0] / PATCH_SIZE[0]) * \
                    (INPUT_SIZE[1] / PATCH_SIZE[1])
            )
    EMBED_DIM = 10
    PATCH_DROPOUT = 0.1
    VAE_LATENT_DIM = 128
    SP_ENC_LATENT_DIMS = [512, 256]
    SP_DEC_LATENT_DIMS = [256, 512, 1024]
    SP_MLP_DIMS = [64, 64]
    SP_N_HEADS = [5, 5]
    SP_N_TRNFR_LAYERS = [7, 7]
    SP_DROPOUTS = [0.1, 0.1]

    if verbose: print("initializing WeatherBenchDataset class ...")
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_ZLEVS_T2M.zarr"
    wb_train = WeatherBenchDataset(path_to_zarr=OUTFILE, to_tensor=True,
                                   partition="train")
    wb_val = WeatherBenchDataset(path_to_zarr=OUTFILE, to_tensor=True,
                                 partition="val")
    if verbose: print("set up train and val dataloaders ...")
    train_sampler = DistributedSampler(wb_train)
    val_sampler = DistributedSampler(wb_val)
    train_loader = DataLoader(wb_train, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=WORKERS,
                              sampler=train_sampler)
    val_loader = DataLoader(wb_val, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=WORKERS,
                            sampler=val_sampler)
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
                  input_size=INPUT_SIZE
            )

    # device_ids = [i for i in range(torch.cuda.device_count())]
    # spvae = nn.DataParallel(spvae, device_ids=device_ids)
    # device = torch.device('cuda')
    # spvae = spvae.to(device)

    # more ddp stuff
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    spvae = spvae.to('cuda:' + str(local_rank))
    spvae = DDP(spvae, device_ids=[local_rank], find_unused_parameters=True)

    if verbose: print("train the VAE ...")
    opt = torch.optim.Adam(spvae.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=0.00001)
    scaler = torch.amp.GradScaler("cuda")
    train_loss = np.zeros(EPOCHS)
    val_loss = np.zeros(EPOCHS)
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()
        if verbose: print(f"epoch {epoch}")
        if verbose: print("train ...")
        spvae.train()
        for X in tqdm(train_loader, disable=(not verbose)):
            X = X.to('cuda:' + str(local_rank))
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                X_, kl = spvae(X)
                X_ = X_.to('cuda:' + str(local_rank))
                kl = kl.to('cuda:' + str(local_rank))
                loss = ((X - X_)**2).sum() + kl #spvae.vae.encoder.kl
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        train_loss[epoch] = loss
        epoch_end_time = time.perf_counter()
        if global_rank == 0:
            print(f"Epoch {epoch+1} Time", epoch_end_time - epoch_start_time)
        if epoch % 5 == 0:
            FNAME = "/home/bedartha/data/scratch/spvae_ddp_epoch%d.tar" % epoch
            torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":spvae.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": loss
                        },
                    FNAME
                    )
            print("saved checkpoint to %s" %FNAME)
        if verbose: print("validate...")
        spvae.eval()
        with torch.no_grad():
            for X in tqdm(val_loader, disable=(not verbose)):
                X = X.to('cuda:' + str(local_rank))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    X_, kl = spvae(X)
                    X_ = X_.to('cuda:' + str(local_rank))
                    kl = kl.to('cuda:' + str(local_rank))
                    loss = ((X - X_)**2).sum() + kl #spvae.vae.encoder.kl
        val_loss[epoch] = loss
    end = time.perf_counter()
    if global_rank == 0:
        print("Training Took", end - start)
    np.savez("/home/bedartha/data/scratch/spvae_loss.npz",
             train_loss=train_loss, val_loss=val_loss)

