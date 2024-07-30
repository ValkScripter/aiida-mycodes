"""Workchain to fit the Birch-Murnaghan equation of state to a given set of volumes and energies."""

import numpy as np
from numpy import power
from scipy.optimize import curve_fit

from aiida import orm
from aiida.engine.processes import WorkChain


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `BirchMurnaghanFitWorkChain`."""
    volumes = inputs.get("volumes")
    energies = inputs.get("energies")
    fit_options = inputs.get("fit_options")

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


class BirchMurnaghanFitWorkChain(WorkChain):
    """
    Workchain to fit the Birch-Murnaghan equation of state to a given set of volumes and energies.
    1.  Takes a list of volumes and corresponding total energies, and optional fitting options
        If no fitting options are provided, p0=[E0, V0, B0, Bp] = [energies[imin], volumes[imin], 1, 1]
    2.  Fits the Birch-Murnaghan equation of state to the data
    3.  Returns the optimized energy (E0), volume (V0), bulk modulus (B0) and its pressure derivative (Bp)
        with the corresponding error (perr) and covariance matrix (pcov)
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)
        spec.input(
            "volumes", valid_type=orm.List, help="List of volumes"
        )
        spec.input(
            "energies", valid_type=orm.List, help="List of energies"
        )
        spec.input(
            "fit_options",
            valid_type=orm.Dict,
            help="Options for fitting: ('p0=[E0,V0,B0,Bp]': parameters guess, 'sigma': energy std)",
            default=lambda: orm.Dict({}),
        )

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.fit,
            cls.results,
        )

        spec.output(
            "optimized_parameters",
            valid_type=orm.Dict,
            help="Optimized parameters of the Birch-Murnaghan equation of state",
        )
        spec.output(
            "error",
            valid_type=orm.Dict,
            help="Error of the optimized parameters",
        )
        spec.output(
            "covariance_matrix",
            valid_type=orm.ArrayData,
            help="Covariance matrix of the optimized parameters",
        )

    def setup(self):
        """Initialize context variables."""
        self.ctx.volumes = self.inputs.volumes.get_list()
        self.ctx.energies = self.inputs.energies.get_list()
        self.ctx.fit_options = self.inputs.fit_options.get_dict()

        # Set default fitting options
        if "p0" not in self.ctx.fit_options:
            imin = np.argmin(self.ctx.energies)
            self.ctx.fit_options["p0"] = [
                self.ctx.energies[imin],
                self.ctx.volumes[imin],
                1,
                1,
            ]

        # Set default energy standard deviation
        if "sigma" not in self.ctx.fit_options:
            self.ctx.fit_options["sigma"] = 0.1

        self.ctx.p0 = self.ctx.fit_options["p0"]
        self.ctx.sigma = self.ctx.fit_options["sigma"]

    def fit(self):
        """Fit the Birch-Murnaghan equation of state to the data."""
        # pylint: disable=unbalanced-tuple-unpacking
        popt, pcov = curve_fit(
            BM_law,
            self.ctx.volumes,
            self.ctx.energies,
            p0=self.ctx.p0,
            sigma=self.ctx.sigma,
            absolute_sigma=False,
        )
        # pylint: enable=unbalanced-tuple-unpacking

        self.ctx.popt = popt
        self.ctx.pcov = pcov

    def results(self):
        """Return the optimized parameters, error and covariance matrix."""
        self.out(
            "optimized_parameters",
            orm.Dict(
                dict(zip(["E0", "V0", "B0", "Bp"], self.ctx.popt))
            ),
        )
        self.out(
            "error",
            orm.Dict(
                dict(
                    zip(
                        ["E0", "V0", "B0", "Bp"],
                        np.sqrt(np.diag(self.ctx.pcov)),
                    )
                )
            ),
        )
        self.out("covariance_matrix", orm.ArrayData(self.ctx.pcov))
