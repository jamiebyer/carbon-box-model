import numpy as np
from utilities.emissions import emissions


def dMdt(t, M, k, add_flux_C, add_emissions):
    """
    t: float (scalar), time.
    M: size 4 or 9 array, masses of carbon in boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).

    dMdt: size 4 or 9 array, change in mass for boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).
    """

    # Update values for fluxes from mass.
    F = np.multiply(k, M)

    F_in = np.sum(F, axis=1)  # row-wise for in-flux
    F_out = np.sum(F, axis=0)  # column-wise for out-flux

    dMdt = F_in - F_out

    # From step 0.1, add an arbitrary flux C (in Gt/yr)
    if add_flux_C:
        T = 0.1 / (2 * np.pi)
        # T = 100 / (2 * np.pi)
        C = (20 / T) * (np.sin(t / T))
    else:
        C = 0

    if add_emissions:
        e = emissions([t])[0]
    else:
        e = 0

    # add forcing fluxes to first slot: the atmosphere box
    dMdt[0] += C + e

    return dMdt
