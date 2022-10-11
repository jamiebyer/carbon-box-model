from os import environ

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from pathlib import Path

import xarray as xr
import cftime
import datetime
import json

"""
Generates data/model-data.csv from CMIP5 and CMIP6 model data.

Data collected from:
- https://esgf-node.llnl.gov/search/cmip5/
- https://esgf-index1.ceda.ac.uk/search/cmip6/
"""

with open('data/models/models.json') as json_file:
    models = json.load(json_file)

def year_fraction(date):
    # Convert datetime to year decimal date.
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


df = pd.DataFrame()

for model, mod_dict in models.items():
    # Get year spacing to consolidate data split over many files
    year_min = mod_dict["year_range"][0]
    year_max = mod_dict["year_range"][1]
    year_spacing = mod_dict["year_spacing"]

    full_times = np.array([]).astype(datetime.datetime)
    full_fco2fos = np.array([])

    # Looping over each file for a model
    for year in np.arange(year_min, year_max+1, year_spacing):
        file_path = "data/models/"+model+"/fco2fos_Amon_"+model+"_"+mod_dict["ensemble"]+"_"+str(year)+"01-"+str(year+year_spacing-1)+"12.nc"
        data = xr.open_dataset(file_path)
        
        # Mean over latitudes and longitudes
        spatial_mean = data.mean(dim=["lat", "lon"])
        times = spatial_mean.indexes["time"]._data

        # Convert pandas datetimes to pydatetime
        if (type(times) is tuple) or (type(times) is np.ndarray):
            full_times = np.append(full_times, times)
        else:
            full_times = np.append(full_times, times.to_pydatetime())

        # Multiply emissions by Earth's surface area. Convert kg/s to Gt/yr
        surface_area = np.pi*(6371000)**2
        unit_conversion = (3.154E7)/(1E12)
        full_fco2fos = np.append(full_fco2fos, spatial_mean["fco2fos"].data * surface_area * unit_conversion)
    
    # Loop over time arrays and convert netcdf to datetime. Convert all times to decimal years.
    for i in range(len(full_times)):
        if (type(times) is tuple) or (type(times) is np.ndarray):
            full_times[i] = datetime.datetime(full_times[i].year, full_times[i].month, full_times[i].day, full_times[i].hour, full_times[i].minute, full_times[i].second)
        full_times[i] = year_fraction(full_times[i])
    
    # Add model data to pandas dataframe
    df = pd.concat([df, pd.DataFrame(
        {model+"_times": full_times, model+"_emissions": full_fco2fos}
        )], axis=1)

# Add IPCC A2 model
A2_times = np.array([0, 1850, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 10000])
A2_emissions = np.array([0, 0,  6.875, 8.125, 9.375, 12.5, 14.375, 16.25, 17.5, 19.75, 21.25, 23.125, 26.25, 28.75, 0, 0, 0])

df = pd.concat([df, pd.DataFrame(
        {"IPCC-A2_times": A2_times, "IPCC-A2_emissions": A2_emissions}
        )], axis=1)

# Save dataframe as csv
df.to_csv("./data/model-data.csv", index=False)

