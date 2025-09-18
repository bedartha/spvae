#! /usr/bin/env python

HOMEDIR = "/home/bedartha/"
DATADIR = "public/datasets/as_downloaded/weatherbench2/era5/"
ARRNAME = "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


import sys
import os
import xarray as xr
from pprint import pprint


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
    ds = get_dataset()
    z = ds["geopotential"]
    z500 = z.sel(level=500)#.copy()
    ds.close()
    return z500


if __name__ == "__main__":
    out = eval(f"{sys.argv[1]}()")
    z500_clim = out.mean(dim="time")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    subplot_kws=dict(projection=ccrs.PlateCarree(),
                 facecolor='grey')
    plt.figure(figsize=[12,8])
    z500_clim.plot(x='longitude', y='latitude',
                  #vmin=-2, vmax=32,
                  #cmap=cmocean.cm.thermal,
                  subplot_kws=subplot_kws,
                  transform=ccrs.Robinson())
    plt.savefig("./z500_clim.png")
    print("z500 climatology plot saved to disk")
