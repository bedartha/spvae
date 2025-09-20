"""
Defines stuff needed to design the custom Dataset class for WeatherBench2

(c) 2025 Bedartha Goswami <bedartha.goswami@iiserpune.ac.in>
"""


import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    """
    Custom class for the WeatherBench dataset.
    """

    def __init__(self, path_to_zarr="./", to_tensor=True):
        """
        Arguments:
            var_names (list of strings): List of variables to extract
            levels (list of ints, optional): List of pressure levels, if reqd
            root_dir  (string): Path to dataset archive
            transform (callable, optional): Optional transforms to be used
        """
        self.path = path_to_zarr
        self.to_tensor = to_tensor
        ds = xr.open_zarr(
                path_to_zarr,
                chunks={},
                )

        ds.chunk(chunks={"time": 50, "lat": 32, "lon": 64})
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


