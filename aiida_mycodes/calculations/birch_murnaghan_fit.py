"""Calculation plugin for fitting the Birch-Murnaghan equation of state to a set of volumes and energies."""

import numpy as np
from numpy import power
import pymc as pm
from scipy.optimize import curve_fit

from aiida import orm
from aiida.engine import calcfunction

__all__ = ["birch_murnaghan_fit"]


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
        (Bp - 4) * (eta - 1.0) + 2
    )


# pylint: disable=unused-argument
def BM_law_deriv(V, E0, V0, B0, Bp):
    """
    Derivative of the Birch-Murnaghan equation of state
    :param V: volume
    :param E0: energy at the minimum
    :param V0: equilibrium volume
    :param B0: bulk modulus
    :param Bp: pressure derivative of the bulk modulus
    :return: derivative of the energy at volume V
    """

    eta = power(V0 / V, 2.0 / 3.0)
    return (
        -(3.0 / 8.0)
        * B0
        * power(eta, 2.5)
        * (eta - 1)
        * (3 * (Bp - 4) * (eta - 1) + 4)
    )


def validate_inputs(volumes, energies, prior_means):
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

    if prior_means and not isinstance(prior_means, orm.Dict):
        return "Input 'prior_means' must be of type Dict"

    return None


# pylint: disable=too-many-locals, unreachable, unused-variable
# Deprecated: Birch-Murnaghan fit with scipy curve_fit
@calcfunction
def birch_murnaghan_fit(
    volumes: orm.List,
    energies: orm.List,
    fit_options: orm.Dict = lambda: orm.Dict({}),
):
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
    raise DeprecationWarning(
        "This function is deprecated, use `birch_murnaghan_fit_bayesian` instead."
    )
    validate_inputs(volumes, energies, fit_options)

    volumes = np.array(volumes.get_list())
    energies = np.array(energies.get_list())
    fit_options = fit_options.get_dict()

    # Default values for fit_options if not given
    if "p0" not in fit_options:
        imin = np.argmin(energies)
        fit_options["p0"] = [energies[imin], volumes[imin], 1, 1]

    if "sigma" not in fit_options:
        fit_options["sigma"] = 0.01

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
        popt=orm.Dict(dict(zip(["E0", "V0", "B0", "Bp"], popt))),
        pcov=orm.ArrayData(pcov),
        perr=orm.Dict(dict(zip(["E0", "V0", "B0", "Bp"], perr))),
    )


# pylint: enable=unreachable


# Birch-Murnaghan fit with pymc bayesian analysis
@calcfunction
def birch_murnaghan_fit_bayesian(
    volumes: orm.List,
    energies: orm.List,
    prior_means: orm.Dict = lambda: orm.Dict({}),
):
    """
    Fit the Birch-Murnaghan equation of state to a set of volumes and energies using pymc.

    :param volumes: a list of volumes
    :param energies: a list of energies
    :param prior_means: the prior means for the predictors [E0, V0, B0, Bp, sigma_E]: optional
    :return: a dictionary with the fitting results:
        "popt": the dict of the mean optimal parameters of the fit [E0, V0, B0, Bp, sigma_E]
        "perr": the dict of standard deviations of the fit parameters [E0, V0, B0, Bp, sigma_E]
    """
    validate_inputs(volumes, energies, prior_means)

    volumes = np.array(volumes.get_list())
    energies = np.array(energies.get_list())
    prior_means = prior_means.get_dict()

    # Default values for fit_options if not given
    if not "E0" in prior_means or not "V0" in prior_means:
        imin = np.argmin(energies)
        prior_means["E0"] = energies[imin]
        prior_means["V0"] = volumes[imin]
    if not "B0" in prior_means:
        prior_means["B0"] = 1.0
    if not "Bp" in prior_means:
        prior_means["Bp"] = 2.0
    if not "sigma" in prior_means:
        prior_means["sigma"] = 0.001

    # Define the pymc model
    model = pm.Model()

    with model:
        # Priors
        E0 = pm.Normal("E0", mu=prior_means["E0"], sigma=0.1)
        V0 = pm.Normal("V0", mu=prior_means["V0"], sigma=0.5)
        B0 = pm.Normal("B0", mu=prior_means["B0"], sigma=10.0)
        Bp = pm.Normal("Bp", mu=prior_means["Bp"], sigma=20.0)
        eta = power(V0 / volumes, 2.0 / 3.0)
        E = E0 + 9.0 / 16.0 * B0 * V0 * power(eta - 1.0, 2.0) * (
            6.0 + Bp * (eta - 1.0) - 4.0 * eta
        )
        sigma = pm.HalfNormal("sigma", sigma=prior_means["sigma"])

        # Likelihood
        E_obs = pm.Normal("E_obs", mu=E, sigma=sigma, observed=energies)

        # Run the sampler
        idata = pm.sample(1000)

    # Extract the results
    popt = idata.posterior.mean().to_pandas().to_dict()
    perr = idata.posterior.std().to_pandas().to_dict()

    return dict(
        popt=orm.Dict(popt),
        perr=orm.Dict(perr),
    )
