import numpy as np
import pandas as pd

df = pd.read_csv("./data/model-data.csv")

def emissions(yr, model):
    """
    Function defining A2 emission scenario over the interval 1990-2100
    extended to pre-industrial (assuming linear increase from 0 in 1850 to 1900) and assuming full cessation of CO_2 input at 2101

    Example of use:
        import numpy as np
        import matplotlib.pyplot as plt
        yr = np.arange(0,2500, 10)
        e = emissions(yr)
        plt.plot(yr, e)
        plt.show()
    For additional information see http://www.grida.no/climate/ipcc/emission

    model is one of: "short_sine", "long_sine", "short_exp", "long_exp",
        "IPCC-A2", "GFDL-ESM2G_esmrcp85", "CNRM-ESM2-1_esm-ssp585", "MPI-ESM1-2-LR_esm-ssp585", "UKESM1-0-LL_esm-ssp585"
    """
    yr = np.asarray(yr)
    if model == "short_sine":
        T = 0.1  # short wavelength
        e = (20 / T) * (np.sin((2*np.pi*yr) / T))
    elif model == "long_sine":
        T = 100  # long wavelength
        e = (20 / T) * (np.sin((2*np.pi*yr) / T))
    elif model == "short_exp":
        T = 0.1  # short wavelength
        e = np.exp(-yr) * (np.sin((2*np.pi*yr) / T))  # exponential decay
    elif model == "long_exp":
        T = 100  # long wavelength
        e = np.exp(-yr) * (np.sin((2*np.pi*yr) / T))  # exponential decay
    else:
        t_yr = df[model + "_times"]
        e_GtC_yr = df[model + "_emissions"]
        e = np.interp(yr, t_yr, e_GtC_yr)

    return e
