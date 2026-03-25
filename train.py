import os
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
from params import Params




def set_ddp(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    """set up DDP"""
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return None


def get_dataloader(path_to_zarr, to_tensor, partition, batch_size, num_workers,
                   shuffle=False, verbose=True):
    """
    Loads the WeatherBench Dataset class and prepares the Dataloader
    -----------------------------------------------------------------
    """
    wb = WeatherBenchDataset(path_to_zarr, to_tensor,  partition)
    sampler = DistributedSampler(wb)
    loader = DataLoader(wb,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        sampler = sampler,
                        shuffle = shuffle
                        )
    return loader


def set_model(local_rank, params):
    """set up model and set to device"""
    # initalize model
    spvae = SPVAE(
                  embed_dim = params.embed_dim,
                  patch_size = params.patch_size,
                  num_patches = params.num_patches,
                  patch_dropout = params.patch_dropout,
                  in_channels = params.in_channels,
                  keep_channels = params.keep_channels,
                  vae_latent_dim = params.vae_latent_dim,
                  sp_enc_latent_dims = params.sp_enc_latent_dims,
                  sp_dec_latent_dims = params.sp_dec_latent_dims,
                  sp_mlp_dims = params.sp_mlp_dims,
                  sp_n_heads = params.sp_n_heads,
                  sp_n_trnfr_layers = params.sp_n_trnfr_layers,
                  sp_dropouts = params.sp_dropouts,
                  batch_size = params.batch_size,
                  input_size = params.input_size,
                  output_padding = params.output_padding
            )
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    spvae = spvae.to('cuda:' + str(local_rank))
    spvae = DDP(spvae, device_ids=[local_rank], find_unused_parameters=True)

    return spvae


def set_optimizer(spvae, params):
    """set up optimizer and LR scheduler"""
    opt = torch.optim.Adam(spvae.parameters(),
                           lr=params.learning_rate
                           )
    scheduler = CosineAnnealingLR(opt,
                                  T_max=params.epochs,
                                  eta_min=params.learning_rate_min
                                  )
    scaler = torch.amp.GradScaler("cuda")
    return opt, scaler


def train(train_loader, val_loader, model, optim, local_rank, params,
          scaler, verbose, nname, global_rank, out_dir):
    """train the model"""
    train_loss = np.zeros(params.epochs)
    val_loss = np.zeros(params.epochs)
    for epoch in range(params.epochs):
        
        if verbose: 
            print(f"epoch {epoch} | global rank {global_rank} | " + \
                    f"local rank {local_rank} | train ...")
      
        model.train()
        for X in train_loader:
            X = X.to('cuda:' + str(local_rank))
            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                X_, kl = model(X)
                X_ = X_.to('cuda:' + str(local_rank))
                kl = kl.to('cuda:' + str(local_rank))
                t_loss = ((X - X_)**2).sum() + kl
                scaler.scale(t_loss).backward()
                scaler.step(optim)
                scaler.update()
        train_loss[epoch] = t_loss
        
        if verbose: 
            print(f"epoch {epoch} | global rank {global_rank} | " + \
                    f"local rank {local_rank} | validate ...")

        model.eval()
        with torch.no_grad():
            for X in val_loader:
                X = X.to('cuda:' + str(local_rank))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    X_, kl = model(X)
                    X_ = X_.to('cuda:' + str(local_rank))
                    kl = kl.to('cuda:' + str(local_rank))
                    v_loss = ((X - X_)**2).sum() + kl
        val_loss[epoch] = v_loss
        
        #save checkpoint
        if local_rank == 0 and global_rank == 0:
            if epoch % params.save_every == 0:
                if verbose:
                    print(" ")
                    print(f"save checkpoint at {epoch} epochs ...")
                FNAME = f"{out_dir}/chkpt_epoch{epoch:03d}.tar"
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict":model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "train_loss": t_loss,
                            "val_loss": v_loss,
                            },
                        FNAME
                        )
        
                if verbose:
                    print(f"saved checkpoint to {FNAME}")
                    print(" ")

    return train_loss, val_loss


def main(rank, world_size, params, args, nname, out_dir):
    """runs the main code for training and validation"""
    set_ddp(rank, world_size)
    global_rank = int(os.environ["SLURM_PROCID"])

    trn_dl = get_dataloader(path_to_zarr=args.in_file,
                            to_tensor=True,
                            partition="train",
                            batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            shuffle=False,
                            verbose=args.verbose
                            )
    val_dl = get_dataloader(path_to_zarr=args.in_file,
                            to_tensor=True,
                            partition="val",
                            batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            shuffle=False, verbose=args.verbose
                            )
    spvae = set_model(rank, params)
    opt, scaler = set_optimizer(spvae, params)
    tl, vl = train(trn_dl, val_dl, spvae, opt, rank, params,
                   scaler, args.verbose, nname, global_rank, out_dir)

    if global_rank == 0 and rank == 0:
        if verbose: print("save training curves to NPZ file ...")
        FNAME = f"{out_dir}/training_curves.npz"
        np.savez(FNAME,
                 train_loss=tl / len(trn_dl),
                 val_loss=vl / len(val_dl),
                 epochs=np.range(1, params.epochs  + 1),
                 )
        if verbose: print(f"saved to: {FNAME}")
    dist.destroy_process_group()
    if verbose:
        print(f"done with process {rank} of {world_size}" +\
              f"on node {nname} with global rank {global_rank}")
    return None

