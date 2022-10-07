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

    model is one of: "A2", "GFDL-ESM2G_esmrcp85", "CNRM-ESM2-1_esm-ssp534-over", "CNRM-ESM2-1_esm-ssp585", 
        "MPI-ESM1-2-LR_esm-ssp585", "MPI-ESM1-2-LR_ssp245-cov-fossil", "UKESM1-0-LL_esm-ssp585"
    """
    t_yr = df[model + "_times"]
    e_GtC_yr = df[model + "_emissions"]
    
    e = np.interp(yr, t_yr, e_GtC_yr)
    
    return e
