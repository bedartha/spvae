#! /usr/bin/env python

HOMEDIR = "/home/bedartha/"
DATADIR = "public/datasets/as_downloaded/weatherbench2/era5/"
ARRNAME = "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


import sys
import os
import xarray as xr
from pprint import pprint

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import WeatherBenchDataset


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


def get_zlevs():
    """Extracts Z500 from the ZARR dataset and saves to disk if reqd."""
    print("get zlevs ...")
    ds = get_dataset()
    z = ds["geopotential"]
    zlevs = z.sel(level=[250,500,850])
    # ds.close()
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_ZLEVS.zarr"
    print("saving ZLEVS to disk as zarr file ...")
    # print(OUTFILE)
    zlevs.to_zarr(OUTFILE, mode="w", zarr_format=2, consolidated=True)
    print("saved to: %s" % OUTFILE)
    return zlevs


if __name__ == "__main__":
    # test out the weatherbench dataset class
    OUTPATH = "~/public/datasets/for_model_development/weatherbench2/era5/"
    OUTFILE = f"{OUTPATH}{ARRNAME[:-5]}_ZLEVS.zarr"
    wbds = WeatherBenchDataset(path_to_zarr=OUTFILE, to_tensor=True)

    # test out num_workers speed up
    dataloader = DataLoader(wbds, batch_size=5, shuffle=True, num_workers=10)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.shape)
        if i_batch == 100:
            break
    # out = eval(f"{sys.argv[1]}()")

