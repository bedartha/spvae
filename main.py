#! /usr/bin/env python

HOMEDIR = "/home/bedartha/"
DATADIR = "public/datasets/as_downloaded/weatherbench2/era5/"
ARRNAME = "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


import sys
import os
import xarray as xr
import numpy as np
from pprint import pprint

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import WeatherBenchDataset
from model import PatchEmbedding, StackedPerceiver, VariationalAutoencoder
from model import PatchDecoder


def get_dataset():
    """Returns the dataset as an xarray Dataset object."""
    PATH_TO_DATA = f"{HOMEDIR}{DATADIR}{ARRNAME}"
    ds = xr.open_zarr(PATH_TO_DATA, chunks=None)
    return ds


def data_info():
    """Prints information about the dataset to stdout"""
    ds = get_dataset()
    vnames = []
    for vk in ds.variables.keys():
        vnames.append(vk)
    pprint(vnames)
    return vnames


def get_z500():
    """Extracts Z500 from the ZARR dataset and saves to disk if reqd."""
    print("get z500 ...")
    ds = get_dataset()
    z = ds["geopotential"]
    z500 = z.sel(level=500)
    # ds.close()
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_Z500.zarr"
    print("saving Z500 to disk as zarr file ...")
    # print(OUTFILE)
    z500.to_zarr(OUTFILE, mode="w", zarr_format=2, consolidated=True)
    print("saved to: %s" % OUTFILE)
    return z500


def get_zlevs_t2m():
    """Extracts Z500 from the ZARR dataset and saves to disk if reqd."""
    print("get zlevs ...")
    extract_vars = ["2m_temperature", "geopotential"]
    levs = [250, 500, 850]
    ds = get_dataset()[extract_vars]
    ds = ds.sel(level=levs)
    print("convert xarray Dataset to xarray DataArray ...")
    t2m = ds["2m_temperature"]
    gpo = ds["geopotential"]
    print("create xarray dataset with extracted arrays ...")
    dsout = xr.Dataset(
            data_vars=dict(
                t2m=(["time", "lon", "lat"], t2m.data),
                z250=(["time", "lon", "lat"], gpo.sel(level=250).data),
                z500=(["time", "lon", "lat"], gpo.sel(level=500).data),
                z850=(["time", "lon", "lat"], gpo.sel(level=850).data),
                ),
            coords=dict(
                time=("time", ds.time.data),
                lon=("lon", ds.longitude.data),
                lat=("lat", ds.latitude.data),
                ),
            attrs=dict(
                desription="Data extracted from the original WB2 Zarr file"
                )
            )
    ds.close()
    print("saving ZLEVS and T2M to disk as ZARR archive ...")
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_ZLEVS_T2M.zarr"
    dsout.to_zarr(OUTFILE, mode="w", zarr_format=2, consolidated=True)
    print("saved to: %s" % OUTFILE)
    return dsout


if __name__ == "__main__":
    # data_info()
    # sys.exit()
    # extract three geopotential levels and t2m
    # get_zlevs_t2m()
    # sys.exit()

    # test out the weatherbench dataset class
    print("initializing WeatherBenchDataset class ...")
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_ZLEVS_T2M.zarr"
    wbds = WeatherBenchDataset(path_to_zarr=OUTFILE, to_tensor=True,
            partition="train")

    # test out num_workers speed up
    test_dataloader = False
    if test_dataloader:
        print("using torch DataLoader to sample mini-batches ...")
        dataloader = DataLoader(wbds, batch_size=5, shuffle=True, num_workers=13)
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch, sample_batched.shape)
            if i_batch == 99:
                break

    # test out the patch embedding
    batch_size = 1
    test_patch_embedding = True
    look_inside_conv2d_weights = False
    embed_dim = 10
    if test_patch_embedding:
        print(f"get sample of batch size {batch_size} ...")
        dataloader = DataLoader(wbds,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0)
        sample = next(iter(dataloader))
        W, H = sample.shape[2], sample.shape[3]
        w, h = 4, 2
        num_patches = int((W / w) * (H / h))
        print("testing out patch embedding ...")
        embeddings = PatchEmbedding(
                embed_dim=embed_dim,
                patch_size=(w, h),
                num_patches=num_patches,
                dropout=0.1, in_channels=4,
                keep_channels=False
                )
        if look_inside_conv2d_weights:
            children = embeddings.patcher.children()
            obj = []
            for child in children:
                obj.append(child)
            conv2d = obj[0]
            print(conv2d.weight.shape)
        print(sample.shape)
        x = embeddings(sample)

    # test out the perceiver block
    test_perceiver = False
    latent_dim = 512
    latent2_dim = 256
    mlp_dim = 64
    mlp2_dim = 64
    n_heads = 1
    n_layers = 5
    if test_perceiver:
        print("testing Perceiver IO block ...")
        stacked_prcvr = StackedPerceiver(
                                embed_dim=embed_dim,
                                latent_dims=[128, 64],
                                mlp_dims=[64, 32],
                                n_heads=[5, 5],
                                n_trnfr_layers=[7, 7],
                                dropouts=[0.1, 0.1],
                                batch_size=batch_size
                                )
        out = stacked_prcvr(x)

    # test out the vae
    vae_latent_dim = 128
    test_vae = True
    if test_vae:
        print("testing out the VAE ...")
        print(x.shape)
        vae = VariationalAutoencoder(
                vae_latent_dim=vae_latent_dim,
                sp_enc_latent_dims=[512, 256],
                sp_dec_latent_dims=[256, 512, 1024],
                sp_embed_dim=embed_dim,
                sp_mlp_dims=[64, 64],
                sp_n_heads=[5, 5],
                sp_n_trnfr_layers=[7, 7],
                sp_dropouts=[0.1, 0.1],
                batch_size=batch_size
                )
        x_ = vae(x)

    # test out the patch decoder
    test_patch_decoder = True
    if test_patch_decoder:
        print("test patch decoder ...")
        patch_dec = PatchDecoder(embed_dim=embed_dim,
                                 data_channels=4,
                                 num_patches=256,
                                 patch_size=(4,2),
                                 input_size=(W, H))
        out = patch_dec(x_)
