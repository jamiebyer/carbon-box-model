import numpy as np
from utilities.emissions import emissions


def dm_dt(time, mass_array, k, emissions_model=None):
    """
    t: float (scalar), time.
    M: size 4 or 9 array, masses of carbon in boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).

    dMdt: size 4 or 9 array, change in mass for boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).
    """

    # Update values for fluxes from mass.
    flux_array = np.multiply(k, mass_array)

    flux_in = np.sum(flux_array, axis=1)  # row-wise for in-flux
    flux_out = np.sum(flux_array, axis=0)  # column-wise for out-flux

    dmdt = flux_in - flux_out

    if emissions_model:
        emission_array = emissions([time], model=emissions_model)[0]
    else:
        emission_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    dmdt[0] += emission_array

    return dmdt
