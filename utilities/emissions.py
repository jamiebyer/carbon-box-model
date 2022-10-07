import numpy as np
import pandas as pd


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

    if model == "A2":
        t_yr = np.array(
            [0, 1850, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 10000]
        )
        '''
        CO2 forcing stops at 2100: note that need some zeros 
        close in time to approximate stepwise shutoff with linear interp
        '''
        e_GtC_yr = np.array(
            [0, 0,  6.875, 8.125, 9.375, 12.5, 14.375, 16.25, 17.5, 19.75, 21.25, 23.125, 26.25, 28.75, 0, 0, 0]
        )
    else:
        df = pd.read_csv("./data/model-data.csv")
        t_yr = df[model + "_times"]
        e_GtC_yr = df[model + "_fco2fos"]
    
    e = np.interp(yr, t_yr, e_GtC_yr)
    
    return e
