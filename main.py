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
from model import PatchEmbedding


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
    test_patch_embedding = True
    look_inside_conv2d_weights = False
    if test_patch_embedding:
        print("get sample of batch size 1 ...")
        dataloader = DataLoader(wbds, batch_size=1, shuffle=True, num_workers=0)
        sample = next(iter(dataloader))
        W, H = sample.shape[2], sample.shape[3]
        w, h = 4, 2
        num_patches = int((W / w) * (H / h))
        print("testing out patch embedding ...")
        embeddings = PatchEmbedding(
                embed_dim=10,
                patch_size=(w, h),
                num_patches=num_patches,
                dropout=0.1, in_channels=4
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
        print(x.shape)


    # out = eval(f"{sys.argv[1]}()")

