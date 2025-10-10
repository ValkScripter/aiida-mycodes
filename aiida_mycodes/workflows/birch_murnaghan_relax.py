"""Workchain to relax the cell of a structure using the Birch-Murnaghan equation of state.
The workflow will launch relaxations on randomly strained structures and fit E(V) BM law to get the optimal volume.
The optimal balance is then calculated depending on the number of dimensions along which the structure is strained.
"""

from typing import Union

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
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

__all__ = ["BirchMurnaghanRelaxWorkChain"]


@calcfunction
def extract_volumes_energies(*results_dicts):
    """Extract the volumes and energies from the SCF output_parameters."""
    volumes = []
    energies = []
    for result_dict in results_dicts:
        volumes.append(result_dict.get_dict()["volume"])
        energies.append(result_dict.get_dict()["energy"])

    return {
        "volumes": orm.List(volumes),
        "energies": orm.List(energies),
    }


class BirchMurnaghanRelaxWorkChain(WorkChain, Process):
    """
    Workchain to relax the cell of a structure using the Birch-Murnaghan equation of state.

    Inputs:

        structure: the input structure
        scale_std: the standard deviation to generate the scale factors with a normal law (1.0, scale_std)
        nb_scale: the number of scale factors to generate
        scale_a: whether to scale the lattice parameter a (default: True)
        scale_b: whether to scale the lattice parameter b (default: True)
        scale_c: whether to scale the lattice parameter c (default: True)
        prior_means: the prior means for the pymc bayesian analysis
                    to fit the Birch-Murnaghan equation of state [E0, V0, B0, Bp, sigma]: optional
        pw_code: the code for the SCF PwBaseWorkChain
        scf_options: the options for the SCF PwBaseWorkChain
        scf_overrides: the overrides for the SCF PwBaseWorkChain
        kpoints_distance: the distance between kpoints in the SCF PwBaseWorkChain
                    to create the KpointsData used for all SCF calculations
        protocol: the protocol to use for the SCF PwBaseWorkChain

    Outputs:

        relaxed_structure: the relaxed structure
        volumes: the volumes used in the fit
        energies: the energies used in the fit
        popt: the mean optimal parameters of the fit [E0, V0, B0, Bp, sigma]
        perr: the standard deviations of the parameters

    1. Generate a list of scale factors by drawing from a normal law N(1.0, scale_std)
    2. Scale the input structure by each factor
    3. Compute the energy of each scaled structure using SCF PwBaseWorkChain with the same number of kpoints
    4. Extract the volumes and energies from the SCF output_parameters
    5. Fit the Birch-Murnaghan equation of state to the volumes and energies
    6. Get the optimal scale factor and generate the accordingly scaled_structure
    7. Perform a relaxation of the structure and return it with the volumes, the energies and the fit results
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
            PwRelaxWorkChain,
            exclude=("structure",),
            namespace="relax",
            namespace_options={
                "help": "Inputs for PwRelaxWorkChain. \
If kpoints_distance is given it will be transformed into fixed kpoints grid.",
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
            cls.run_relax_parallel,
            cls.inspect_all,
            cls.extract_volumes_energies,
            cls.fit_birch_murnaghan,
            cls.get_optimal_balance,
            cls.final_relaxation,
            cls.inspect_final_relaxation,
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
            "ERROR_RELAX_FAILED",
            message="One of the relaxations failed",
        )
        spec.exit_code(
            402,
            "ERROR_FINAL_RELAX_FAILED",
            message="The final relaxation failed",
        )

    def generate_scale_factors(self):
        """Generate the list of scale factors."""
        scale_std = self.inputs.scale_std.value
        nb_scale = self.inputs.nb_scale.value
        scale_factors = np.random.normal(1.0, scale_std, nb_scale)
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

    def run_relax_parallel(self):
        """Run the PwRelaxWorkChain for each scaled structure."""
        inputs = AttributeDict(
            self.exposed_inputs(PwRelaxWorkChain, "relax")
        )

        if inputs.base.kpoints_distance is not None:
            self.ctx.kpts = create_kpoints_from_distance(
                structure=self.inputs.structure,
                distance=inputs.base.kpoints_distance,
                force_parity=False,
            )
        else:
            self.ctx.kpts = inputs.base.kpoints
        inputs["base"]["kpoints"] = self.ctx.kpts
        inputs["base"].pop("kpoints_distance")

        scaled_structures = self.ctx.scaled_structures

        futures = {}
        # pylint: disable=no-member
        for i, scaled_structure in enumerate(scaled_structures):
            inputs["structure"] = scaled_structure
            relax_inputs = prepare_process_inputs(
                PwRelaxWorkChain, inputs
            )
            relax_workchain = self.submit(
                PwRelaxWorkChain, **relax_inputs
            )
            self.report(
                f"launched PwRelaxWorkChain<{relax_workchain.pk}> for structure {i}"
            )
            futures[f"relax_{i}"] = relax_workchain
        # pylint: enable=no-member

        return ToContext(**futures)

    # pylint: disable=inconsistent-return-statements, no-member
    def inspect_all(self):
        """
        Verify that all relaxations finished successfully.
        """
        for i in range(self.inputs.nb_scale.value):
            relax_workchain = self.ctx[f"relax_{i}"]
            if not relax_workchain.is_finished_ok:
                self.report(
                    f"PwRelaxWorkChain<{relax_workchain.pk}> failed with exit status {relax_workchain.exit_status}"
                )
                return self.exit_codes.ERROR_RELAX_FAILED
        self.report("All relaxations finished successfully")

    def extract_volumes_energies(self):
        """Extract the volumes and energies from the SCF output_parameters."""
        relax_workchains = [
            self.ctx[f"relax_{i}"]
            for i in range(self.inputs.nb_scale.value)
        ]
        results_dicts = [
            relax_wc.outputs.output_parameters
            for relax_wc in relax_workchains
        ]
        volumes_energies = extract_volumes_energies(*results_dicts)
        self.ctx.volumes = volumes_energies["volumes"]
        self.ctx.energies = volumes_energies["energies"]
        self.report(
            f"Extracted {len(self.ctx.volumes)} volumes and energies"
        )

    def fit_birch_murnaghan(self):
        """Fit the Birch-Murnaghan equation of state to the volumes and energies."""
        volumes = self.ctx.volumes
        energies = self.ctx.energies
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

    def final_relaxation(self):
        """Launches the relaxation of the optimally strained structure"""
        scaled_structure = scale_structure(
            structure=self.inputs.structure,
            scale_factor=orm.Float(self.ctx.optimal_balance),
            scale_a=self.inputs.scale_a,
            scale_b=self.inputs.scale_b,
            scale_c=self.inputs.scale_c,
        )

        inputs = AttributeDict(
            self.exposed_inputs(PwRelaxWorkChain, "relax")
        )
        inputs["structure"] = scaled_structure
        inputs.base.kpoints = self.ctx.kpts
        relax_inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        future = self.submit(PwRelaxWorkChain, **relax_inputs)
        self.report(
            f"Submitted final PwRelaxWorkChain<{future.pk}> \
with cell params: {scaled_structure.get_ase().cell.cellpar()}."
        )
        self.to_context(final_relaxation=future)

    def inspect_final_relaxation(self):
        """Verify that the final relaxation finished successfully."""
        workchain = self.ctx.final_relaxation
        if not workchain.is_finished_ok:
            self.report(
                f"Final relaxation failed with exit status {workchain.exit_status}."
            )
            return self.exit_codes.ERROR_FINAL_RELAX_FAILED
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
