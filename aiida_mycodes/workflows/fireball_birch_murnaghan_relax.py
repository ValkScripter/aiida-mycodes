"""WorkChain to relax the structure of a crystal using the Fireball code and the Birch-Murnaghan equation of state."""

from typing import Union

from aiida_fireball.calculations.fireball import FireballCalculation
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
import numpy as np
from plumpy import WorkChainSpec

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import (
    Process,
    ProcessSpec,
    ToContext,
    WorkChain,
    calcfunction,
)

from ..calculations.birch_murnaghan_fit import (
    birch_murnaghan_fit_bayesian,
)
from ..calculations.structure.tools import scale_structure

# pylint: disable=no-member

__all__ = ("FireballBirchMurnaghanRelaxWorkChain",)


@calcfunction
def extract_volumes_energies(*results_dicts):
    """Extract the volumes and energies from the FireballCalculations' output_parameters."""
    volumes = []
    energies = []
    for result_dict in results_dicts:
        volumes.append(result_dict.get_dict()["volume"])
        energies.append(result_dict.get_dict()["energy"])

    return {
        "volumes": orm.List(volumes),
        "energies": orm.List(energies),
    }


class FireballBirchMurnaghanRelaxWorkChain(WorkChain, Process):
    """
    WorkChain to perform a relaxation of a structure using the Fireball code
    and fit the Birch-Murnaghan equation of state.

    Inputs:
        - structure (orm.StructureData): The input structure.
        - scale_std (orm.Float): The standard deviation to generate the scale factors
        with a normal law (1.0, scale_std). Default is 0.02.
        - nb_scale (orm.Int): The number of scale factors to generate. Default is 20.
        - scale_a (orm.Int): whether to scale the lattice parameter a (default: True)
        - scale_b (orm.Int): whether to scale the lattice parameter b (default: True)
        - scale_c (orm.Int): whether to scale the lattice parameter c (default: True)
        - prior_means (orm.Dict): The prior means for the pymc bayesian analysis to fit
        the Birch-Murnaghan equation of state [E0, V0, B0, Bp, sigma]. Optional.
        - fireball_code (orm.Code): The code for the Fireball calculation.
        - fireball_options (orm.Dict): The options for the Fireball calculation.
        - fireball_parameters (orm.Dict): The parameters for the Fireball calculation.
        - clean_workdir (orm.Bool): Clean the working directories of all child calculations
        if ``clean_workdir=True`` in the inputs. Default is True.

    Outputs:
        - relaxed_structure (orm.StructureData): The relaxed structure.
        - volumes (orm.List): The volumes used in the fit.
        - energies (orm.List): The energies used in the fit.
        - popt (orm.Dict): The mean optimal parameters of the fit [E0, V0, B0, Bp, sigma].
        - perr (orm.Dict): The standard deviations of the parameters.

    Exit Codes:
        - 401 ERROR_FIREBALL_FAILED: One of the Fireball calculations failed.

    Methods:
        - generate_scale_factors: Generate the list of scale factors.
        - run_fireball_parallel: Run the Fireball calculation for each scaled structure.
        - inspect_all: Verify that all Fireball calculations finished successfully.
        - extract_volumes_energies: Extract the volumes and energies from the Fireball output_parameters.
        - fit_birch_murnaghan: Fit the Birch-Murnaghan equation of state to the volumes and energies.
        - return_results: Return the relaxed structure, the volumes, the energies and the fit results.
        - on_termination: Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs.
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.expose_inputs(
            scale_structure,
            exclude=("scale_factor",),
            namespace_options={
                "help": "Inputs for scaling the structure",
                "required": True,
            },
        )
        spec.input(
            "nb_scale",
            valid_type=orm.Int,
            default=lambda: orm.Int(20),
            help="The number of scale factors to generate.",
        )
        spec.input(
            "scale_std",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.02),
            help="The standard deviation to generate the scale factors with a normal law (1.0, scale_std).",
        )
        spec.input(
            "prior_means",
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict({}),
            help="The prior means for the pymc bayesian analysis to fit \
                the Birch-Murnaghan equation of state [E0, V0, B0, Bp, sigma].",
        )
        spec.expose_inputs(
            FireballCalculation,
            exclude=("structure",),
            namespace="fireball",
            namespace_options={
                "help": "Inputs for the FireballCalculation",
                "required": True,
            },
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs.",
        )
        spec.outline(
            cls.generate_scale_factors,
            cls.scale_structure,
            cls.run_fireball_parallel,
            cls.inspect_all,
            cls.extract_volumes_energies,
            cls.fit_birch_murnaghan,
            cls.get_optimal_balance,
            cls.final_fireball,
            cls.inspect_final_fireball,
            cls.return_results,
        )
        spec.output(
            "relaxed_structure",
            valid_type=orm.StructureData,
            help="The relaxed structure.",
        )
        spec.output(
            "volumes",
            valid_type=orm.List,
            help="The volumes used in the fit.",
        )
        spec.output(
            "energies",
            valid_type=orm.List,
            help="The energies used in the fit.",
        )
        spec.output(
            "popt",
            valid_type=orm.Dict,
            help="The mean optimal parameters of the fit [E0, V0, B0, Bp, sigma].",
        )
        spec.output(
            "perr",
            valid_type=orm.Dict,
            help="The standard deviations of the parameters.",
        )

        # Exit codes
        spec.exit_code(
            401,
            "ERROR_FIREBALL_FAILED",
            message="One of the Fireball calculations failed",
        )
        spec.exit_code(
            402,
            "ERROR_FINAL_FIREBALL_FAILED",
            message="The final Fireball calculations failed",
        )

    def generate_scale_factors(self) -> None:
        """Generate the list of scale factors."""
        scale_std: float = self.inputs.scale_std.value
        nb_scale: int = self.inputs.nb_scale.value
        scale_factors: np.ndarray = np.random.normal(
            1.0, scale_std, nb_scale
        )
        self.ctx.scale_factors = scale_factors
        self.report(f"Generated {nb_scale} scale factors")

    def scale_structure(self):
        """Scale the input structure by each factor."""
        inputs = self.exposed_inputs(scale_structure)
        scale_factors = self.ctx.scale_factors
        scaled_structures = []
        for scale_factor in scale_factors:
            inputs["scale_factor"] = scale_factor
            inputs = prepare_process_inputs(scale_structure, inputs)
            scaled_structure = scale_structure(**inputs)
            scaled_structures.append(scaled_structure)
        self.ctx.scaled_structures = scaled_structures
        self.report(
            "Scaled structures: "
            + ", ".join(map(lambda s: f"<{s.pk}>", scaled_structures))
        )

    def run_fireball_parallel(self) -> ToContext:
        """Run the Fireball calculation for each scaled structure."""
        inputs = AttributeDict(
            self.exposed_inputs(FireballCalculation, "fireball")
        )

        scaled_structures = self.ctx.scaled_structures

        futures: dict = {}
        # pylint: disable=no-member
        for i, scaled_structure in enumerate(scaled_structures):
            inputs["structure"] = scaled_structure
            fireball_inputs = prepare_process_inputs(
                FireballCalculation, inputs
            )
            fireball_calc = self.submit(
                FireballCalculation, **fireball_inputs
            )
            self.report(
                f"launched FireballCalculation<{fireball_calc.pk}> for structure {i}"
            )
            futures[f"fireball_{i}"] = fireball_calc
        # pylint: enable=no-member

        return ToContext(**futures)

    def inspect_all(self):
        """Verify that all Fireball calculations finished successfully."""
        for i in range(self.inputs.nb_scale.value):
            fireball_calc = self.ctx[f"fireball_{i}"]
            if not fireball_calc.is_finished_ok:
                self.report(
                    f"FireballCalculation<{fireball_calc.pk}> failed with exit status {fireball_calc.exit_status}"
                )
                return self.exit_codes.ERROR_FIREBALL_FAILED
        self.report("All Fireball calculations finished successfully")

    def extract_volumes_energies(self):
        """Extract the volumes and energies from the Fireball output_parameters."""
        fireball_calcs = [
            self.ctx[f"fireball_{i}"]
            for i in range(self.inputs.nb_scale.value)
        ]
        results_dicts = [
            fireball.outputs.output_parameters
            for fireball in fireball_calcs
        ]
        volumes_energies = extract_volumes_energies(*results_dicts)
        self.ctx.volumes = volumes_energies["volumes"]
        self.ctx.energies = volumes_energies["energies"]
        self.report(
            f"Extracted {len(self.ctx.volumes)} volumes and energies"
        )

    def fit_birch_murnaghan(self):
        """Fit the Birch-Murnaghan equation of state to the volumes and energies."""
        volumes: orm.List = self.ctx.volumes
        energies: orm.List = self.ctx.energies
        fit_results = birch_murnaghan_fit_bayesian(
            volumes=volumes,
            energies=energies,
            prior_means=self.inputs.prior_means,
        )
        self.ctx.fit_results = fit_results
        self.report(
            f"Fitted Birch-Murnaghan equation of state: {fit_results['popt'].get_dict()}"
        )

    def get_optimal_balance(self):
        """Calculates the optimal balance for minimal energy depending on the number of strained directions."""
        V0 = self.ctx.fit_results["popt"].get_dict()["V0"]
        V = self.inputs.structure.get_ase().get_volume()
        n = sum(
            [
                self.inputs.scale_a.value,
                self.inputs.scale_b.value,
                self.inputs.scale_c.value,
            ]
        )
        balance = (V0 / V) ** (1.0 / n)
        self.ctx.optimal_balance = balance
        self.report(
            f"Calculated an optimal strain factor of {(balance-1.0)*100:.3f} %."
        )

    def final_fireball(self):
        """Launches the final calculation (relaxation) of the optimally strained structure"""
        scaled_structure = scale_structure(
            structure=self.inputs.structure,
            scale_factor=orm.Float(self.ctx.optimal_balance),
            scale_a=self.inputs.scale_a,
            scale_b=self.inputs.scale_b,
            scale_c=self.inputs.scale_c,
        )

        inputs = AttributeDict(
            self.exposed_inputs(FireballCalculation, "fireball")
        )
        inputs["structure"] = scaled_structure
        relax_inputs = prepare_process_inputs(
            FireballCalculation, inputs
        )
        future = self.submit(FireballCalculation, **relax_inputs)
        self.report(
            f"Submitted final FireballCalculation<{future.pk}> \
with cell params: {scaled_structure.get_ase().cell.cellpar()}."
        )
        self.to_context(final_relaxation=future)

    def inspect_final_fireball(self):
        """Verify that the final relaxation finished successfully."""
        workchain = self.ctx.final_relaxation
        if not workchain.is_finished_ok:
            self.report(
                f"Final relaxation failed with exit status {workchain.exit_status}."
            )
            return self.exit_codes.ERROR_FINAL_FIREBALL_FAILED
        self.report("Final relaxation successful.")
        return None

    def return_results(self):
        """Return the relaxed structure, the volumes, the energies and the fit results."""
        relaxed_structure = (
            self.ctx.final_relaxation.outputs.output_structure
        )
        self.out("relaxed_structure", relaxed_structure)
        self.out("volumes", self.ctx.volumes)
        self.out("energies", self.ctx.energies)
        self.out("popt", self.ctx.fit_results["popt"])
        self.out("perr", self.ctx.fit_results["perr"])

    def on_terminated(self):
        """Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )


def clean_calcjob_remote(node):
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        cleaned = True
    except (OSError, KeyError):
        pass
    return cleaned


def clean_workchain_calcs(workchain):
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            if clean_calcjob_remote(called_descendant):
                cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs
