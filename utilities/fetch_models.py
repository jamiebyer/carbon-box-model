from os import environ

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from pathlib import Path

import xarray as xr
import cftime
import datetime

# Generates data/model-data.csv


"""
Models: CESM2 (hot), UKESM1-0-LL (hot), CanESM5 (hot), GISS-E2-1-H (cold), MRI-ESM2-0 (cold), BCC-ESM1 (cold)

Variables: 
fco2antt, Carbon Mass Flux into Atmosphere Due to All Anthropogenic Emissions of CO2 [kg m-2 s-1], Amon
fco2fos, Carbon Mass Flux into Atmosphere Due to Fossil Fuel Emissions of CO2 [kg m-2 s-1], Amon

Experiments:
descriptions: https://www.ipcc-data.org/sim/gcm_monthly/AR5/CMIP5-Experiments.html

esmrcp85: Future projection (2006-2100) forced by RCP8.5. As in experiment 4.2_RCP8.5 but emissions-forced 
    (with atmospheric CO2 determined by the model itself).


ssp585: update of RCP8.5 based on SSP5
ssp126: update of RCP2.6 based on SSP1

- average for year
- convert date time to year ints
"""

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

fig = go.Figure()
models = ["GFDL-ESM2G_esmrcp85"]


models = {
    "GFDL-ESM2G_esmrcp85": {
        "ensemble": "r1i1p1",
        "year_range": [2006, 2096],
        "year_spacing": 5,
    },
    "CNRM-ESM2-1_esm-ssp534-over": {
        "ensemble": "r1i1p1f3_gr",
        "year_range": [2015, 2100],
        "year_spacing": 86,
    },
    "CNRM-ESM2-1_esm-ssp585": {
        "ensemble": "r1i1p1f3_gr",
        "year_range": [2015, 2100],
        "year_spacing": 86,
    },
    "MPI-ESM1-2-LR_esm-ssp585": {
        "ensemble": "r1i1p1f1_gn",
        "year_range": [2015, 2114], #Renamed from 99
        "year_spacing": 20,
    },
    "MPI-ESM1-2-LR_ssp245-cov-fossil": {
        "ensemble": "r5i1p1f99_gn",
        "year_range": [2020, 2059], #Renamed from 50
        "year_spacing": 20,
    },
    "UKESM1-0-LL_esm-ssp585": {
        "ensemble": "r1i1p1f2_gn",
        "year_range": [1999, 2100], #Renamed from 2015
        "year_spacing": 51,
    },
}

df = pd.DataFrame()

for model, mod_dict in models.items():
    mod_id, exp_id = model.split('_')

    year_min = mod_dict["year_range"][0]
    year_max = mod_dict["year_range"][1]
    year_spacing = mod_dict["year_spacing"]

    full_times = np.array([]).astype(datetime.datetime)
    full_fco2fos = np.array([])
    for year in np.arange(year_min, year_max+1, year_spacing):
        file_path = "data/models/"+model+"/fco2fos_Amon_"+model+"_"+mod_dict["ensemble"]+"_"+str(year)+"01-"+str(year+year_spacing-1)+"12.nc"
        data = xr.open_dataset(file_path)
        spatial_mean = data.mean(dim=["lat", "lon"])
        times = spatial_mean.indexes["time"]._data
        if (type(times) is tuple) or (type(times) is np.ndarray):
            full_times = np.append(full_times, times)
        else:
            full_times = np.append(full_times, times.to_pydatetime())
        full_fco2fos = np.append(full_fco2fos, spatial_mean["fco2fos"].data)
    
    # Convert times to decimal years
    for i in range(len(full_times)):
        if (type(times) is tuple) or (type(times) is np.ndarray):
            full_times[i] = datetime.datetime(full_times[i].year, full_times[i].month, full_times[i].day, full_times[i].hour, full_times[i].minute, full_times[i].second)
        full_times[i] = year_fraction(full_times[i])
    
    surface_area = np.pi*(6371000)**2
    unit_conversion = (3.154E7)/(1E12)
    
    df = pd.concat([df, pd.DataFrame(
        {model+"_times": full_times, model+"_fco2fos": full_fco2fos * surface_area * unit_conversion}
        )], axis=1)
    
    #fig.add_trace(go.Scatter(x=full_times, y=full_fco2fos*surface_area, name=mod_id+"_"+mod_dict["exp_id"]+"_"+mod_dict["ensemble"]))
    #fig.update_layout(title="...", xaxis_title="Time", 
    #    yaxis_title="Carbon Mass Flux [kg s-1]", showlegend=True)

#fig.show()

df.to_csv("./data/model-data.csv", index=False)

