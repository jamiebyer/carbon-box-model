import numpy as np
from utilities.emissions import emissions


def dm_dt(time, mass_array, k, add_flux_c, add_emissions):
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

    # From step 0.1, add an arbitrary flux C (in Gt/yr)
    if add_flux_c:
        # T = 0.1 / (2 * np.pi)  # short wavelength
        T = 100 / (2 * np.pi)  # long wavelength
        carbon_forcing = (20 / T) * (np.sin(time / T))
        # carbon_forcing = np.exp(-time) * (np.sin(time / T))  # exponential decay
    else:
        carbon_forcing = 0

    if add_emissions:
        emission_array = emissions([time])[0]
    else:
        emission_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    dmdt[0] += carbon_forcing + emission_array

    return dmdt
