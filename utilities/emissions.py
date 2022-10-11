import numpy as np
import pandas as pd

# model data stored in a csv
df = pd.read_csv("./data/model-data.csv")


def emissions(yr, model):
    """
    Function defining fossil fuel emissions from various models and equations.

    IPCC-A2 model from: http://www.grida.no/climate/ipcc/emission
    CMIP5 model data from: https://esgf-node.llnl.gov/search/cmip5/
    CMIP6 model data from: https://esgf-index1.ceda.ac.uk/search/cmip6/

    model parameter is one of: "short_sine", "long_sine", "short_exp", "long_exp",
        "IPCC-A2", "GFDL-ESM2G_esmrcp85", "CNRM-ESM2-1_esm-ssp585", "MPI-ESM1-2-LR_esm-ssp585", "UKESM1-0-LL_esm-ssp585"
    """
    yr = np.asarray(yr)
    if model == "short_sine": # short wavelength sine wave forcing
        T = 0.1  
        e = (20 / T) * (np.sin((2*np.pi*yr) / T))
    elif model == "long_sine":
        T = 100  
        e = (20 / T) * (np.sin((2*np.pi*yr) / T)) # long wavelength sine wave forcing
    elif model == "short_exp":
        T = 0.1  
        e = np.exp(-yr/2100) * (np.sin((2*np.pi*yr) / T))  # short wavelength exponential decay
    elif model == "long_exp":
        T = 100  # long wavelength
        e = np.exp(-yr/2100) * (np.sin((2*np.pi*yr) / T))  # long wavelength exponential decay
    else:
        # get model data from pandas dataframe
        t_yr = df[model + "_times"]
        e_GtC_yr = df[model + "_emissions"]
        # interpolate yr from model data
        e = np.interp(yr, t_yr, e_GtC_yr)

    return e
