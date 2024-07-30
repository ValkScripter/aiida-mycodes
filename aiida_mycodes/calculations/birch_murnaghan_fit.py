"""Calculation plugin for fitting the Birch-Murnaghan equation of state to a set of volumes and energies."""

import numpy as np
from numpy import power
from scipy.optimize import curve_fit

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction


def BM_law(V, E0, V0, B0, Bp):
    """
    Birch-Murnaghan equation of state
    :param V: volume
    :param E0: energy at the minimum
    :param V0: equilibrium volume
    :param B0: bulk modulus
    :param Bp: pressure derivative of the bulk modulus
    :return: energy at volume V
    """

    eta = power(V0 / V, 2.0 / 3.0)
    return E0 + 9.0 / 16.0 * B0 * V0 * power(eta - 1.0, 2.0) * (
        6.0 + Bp * (eta - 1.0) - 4.0 * eta
    )


def validate_inputs(volumes, energies, fit_options):
    """Validate the inputs of the entire input namespace of `birch_murnaghan_fit`."""
    if not volumes or not energies:
        return "Missing required inputs 'volumes' or 'energies'"

    if not isinstance(volumes, orm.List) or not isinstance(
        energies, orm.List
    ):
        return (
            "Inputs 'volumes' and 'energies' must be of type orm.List"
        )

    if len(volumes) != len(energies):
        return (
            "Inputs 'volumes' and 'energies' must have the same length"
        )

    if fit_options and not isinstance(fit_options, orm.Dict):
        return "Input 'fit_options' must be of type Dict"

    return None


@calcfunction
def birch_murnaghan_fit(
    volumes: orm.List,
    energies: orm.List,
    fit_options: orm.Dict = orm.Dict({}),
) -> orm.Dict:
    """
    Fit the Birch-Murnaghan equation of state to a set of volumes and energies.

    :param volumes: a list of volumes
    :param energies: a list of energies
    :param fit_options: fitting options ("p0": initial guess, "sigma": energy std)
    :return: a dictionary with the fitting results:
        "popt": the optimal parameters of the fit [E0, V0, B0, Bp]
        "pcov": the estimated covariance of popt
        "perr": the standard errors of the parameters
    """

    validate_inputs(volumes, energies, fit_options)

    volumes = np.array(volumes.get_list())
    energies = np.array(energies.get_list())
    fit_options = fit_options.get_dict()

    # Default values for fit_options if not given
    if "p0" not in fit_options:
        imin = np.argmin(energies)
        fit_options["p0"] = [energies[imin], volumes[imin], 1, 1]

    if "sigma" not in fit_options:
        fit_options["sigma"] = 0.1

    # Fit the Birch-Murnaghan equation of state to the data
    # pylint: disable=unbalanced-tuple-unpacking
    popt, pcov = curve_fit(
        BM_law,
        volumes,
        energies,
        p0=fit_options["p0"],
        sigma=fit_options["sigma"],
    )
    # pylint: enable=unbalanced-tuple-unpacking
    perr = np.sqrt(np.diag(pcov))

    return dict(
        popt=AttributeDict(
            {"E0": popt[0], "V0": popt[1], "B0": popt[2], "Bp": popt[3]}
        ),
        pcov=orm.ArrayData(pcov),
        perr=AttributeDict(
            {"E0": perr[0], "V0": perr[1], "B0": perr[2], "Bp": perr[3]}
        ),
    )
