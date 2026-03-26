"""
Defines stuff needed to design the custom Dataset class for WeatherBench2

(c) 2025 Bedartha Goswami <bedartha.goswami@iiserpune.ac.in>
"""


import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


single_lev_var_list = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure",
        "total_column_water_vapour",
        ]
pressure_lev_var_list = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
        ]


class WeatherBenchDataset(Dataset):
    """
    Custom class for the WeatherBench dataset.
    """

    def __init__(self, path_to_zarr="./", to_tensor=True, partition="train"):
        """
        Arguments:
            path_to_zarr  (string): Path to dataset archive
            to_tensor (callable, optional): Converts data to torch.Tensor
        """
        self.path = path_to_zarr
        self.to_tensor = to_tensor
        self.partition = partition
        ds = xr.open_zarr(
                path_to_zarr,
                chunks={},
                )
        
        # reshape the dataset so that the pressure level dimension is
        # collapsed
        new_vars = {}
        for var in pressure_lev_var_list:
            for lev in ds['level'].data:
                new_var = f"{var}_{lev}hpa"
                new_vars[new_var] = ds[var].sel(level=lev).drop_vars('level')
        for var in single_lev_var_list:
            new_vars[var] = ds[var]
        ds = xr.Dataset(new_vars)

        if partition == "train":
            # ds = ds.sel(time=slice("1979-01-01", "2019-12-31"))
            ds = ds.sel(time=slice("2019-01-01", "2019-12-31"))
            # ds = ds.sel(time=slice("2019-01-01", "2019-01-31"))
        elif partition == "val":
            # ds = ds.sel(time=slice("2020-01-01", "2020-12-31"))
            # ds = ds.sel(time=slice("2020-01-01", "2020-12-31"))
            ds = ds.sel(time=slice("2020-01-01", "2020-01-31"))
        elif partition == "test":
            ds = ds.sel(time=slice("2022-01-01", "2023-12-31"))

        # ds.chunk(chunks={"time": 50, "lat": 32, "lon": 64})
        ds.chunk(chunks={"time": 50, "latitude": 16, "longitude": 32})
        # ds.chunk(chunks={"time": 50, "latitude": 32, "longitude": 64})
        self.ds = ds


    def __str__(self):
        """custom print info for the class"""
        return "WeatherBenchDataset object containing\n" + self.ds.__str__()

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.ds["time"])

    def __getitem__(self, idx):
        """
        Returns the data stored at specified index

        Items can be index in multiple ways:
             sample = wbds[i]
             sample = wbds[i:i+10]
             sample = wbds[[3,100,50]]
        """
        time = self.ds.time
        sample = self.ds.sel(time=time[idx]).to_array()
        # here the sample is a dask array because we use chunking
        if self.to_tensor:
            sample = torch.tensor(sample.data.compute())
        return sample


