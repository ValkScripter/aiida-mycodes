"""Workchain to relax the cell of a structure using the Birch-Murnaghan equation of state."""

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain

from aiida_mycodes.calculations.birch_murnaghan_fit import (
    birch_murnaghan_fit,
)
from aiida_mycodes.calculations.structure.tools import scale_structure


class BirchMurnaghanRelaxWorkChain(WorkChain):
    """
    Workchain to relax the cell of a structure using the Birch-Murnaghan equation of state.

    Inputs:

        structure: the input structure
        scale_std: the standard deviation to generate the scale factors with a normal law (1.0, scale_std)
        nb_scale: the number of scale factors to generate
        pw_code: the code for the SCF PwBaseWorkChain
        scf_options: the options for the SCF PwBaseWorkChain
        scf_overrides: the overrides for the SCF PwBaseWorkChain
        protocol: the protocol to use for the SCF PwBaseWorkChain

    Outputs:

        relaxed_structure: the relaxed structure
        volumes: the volumes used in the fit
        energies: the energies used in the fit
        fit_results: the results of the Birch-Murnaghan fit

    1. Generate a list of scale factors by drawing from a normal law N(1.0, scale_std)
    2. Scale the input structure by each factor
    3. Compute the energy of each scaled structure using SCF PwBaseWorkChain
    4. Extract the volumes and energies from the SCF output_parameters
    5. Fit the Birch-Murnaghan equation of state to the volumes and energies
    6. Return the relaxed structure, the volumes, the energies and the fit results
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input(
            "scale_std",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.02),
        )
        spec.input(
            "nb_scale", valid_type=orm.Int, default=lambda: orm.Int(20)
        )
        spec.input("pw_code", valid_type=orm.Code)
        spec.input("scf_options", valid_type=orm.Dict)
        spec.input("scf_overrides", valid_type=orm.Dict)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("moderate"),
        )
        spec.outline(
            cls.generate_scale_factors,
            cls.scale_structure,
            cls.run_scf_parallel,
            cls.extract_volumes_energies,
            cls.fit_birch_murnaghan,
            cls.return_results,
        )
        spec.output("relaxed_structure", valid_type=orm.StructureData)
        spec.output("volumes", valid_type=orm.List)
        spec.output("energies", valid_type=orm.List)
        spec.output("fit_results", valid_type=AttributeDict)

    def generate_scale_factors(self):
        """Generate the list of scale factors."""
        scale_std = self.inputs.scale_std.value
        nb_scale = self.inputs.nb_scale.value
        scale_factors = np.random.normal(1.0, scale_std, nb_scale)
        self.ctx.scale_factors = scale_factors

    def scale_structure(self):
        """Scale the input structure by each factor."""
        structure = self.inputs.structure
        scale_factors = self.ctx.scale_factors
        scaled_structures = []
        for scale_factor in scale_factors:
            scaled_structure = scale_structure(
                structure=structure,
                scale_factor=orm.Float(scale_factor),
            )
            scaled_structures.append(scaled_structure)
        self.ctx.scaled_structures = scaled_structures

    def run_scf_parallel(self):
        """Run the SCF PwBaseWorkChain for each scaled structure."""
        scaled_structures = self.ctx.scaled_structures
        pw_code = self.inputs.pw_code
        scf_options = self.inputs.scf_options.get_dict()
        scf_overrides = self.inputs.scf_overrides.get_dict()
        protocol = self.inputs.protocol.value

        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=pw_code,
            structure=scaled_structures[0],
            protocol=protocol,
            options=scf_options,
            overrides=scf_overrides,
        )

        scf_workchains = []
        for scaled_structure in scaled_structures:
            scf_builder.structure = scaled_structure
            scf_workchain = self.submit(scf_builder)
            scf_workchains.append(scf_workchain)

        self.to_context(scf_workchains=scf_workchains)

    def extract_volumes_energies(self):
        """Extract the volumes and energies from the SCF output_parameters."""
        scf_workchains = self.ctx.scf_workchains
        volumes = []
        energies = []
        for scf_workchain in scf_workchains:
            output_parameters = scf_workchain.outputs.output_parameters
            volumes.append(output_parameters.get_dict()["volume"])
            energies.append(output_parameters.get_dict()["energy"])
        self.ctx.volumes = volumes
        self.ctx.energies = energies

    def fit_birch_murnaghan(self):
        """Fit the Birch-Murnaghan equation of state to the volumes and energies."""
        volumes = self.ctx.volumes
        energies = self.ctx.energies
        fit_results = birch_murnaghan_fit(
            volumes=orm.List(volumes), energies=orm.List(energies)
        )
        self.ctx.fit_results = fit_results

    def return_results(self):
        """Return the relaxed structure, the volumes, the energies and the fit results."""
        V0 = self.ctx.fit_results["popt"]["V0"]
        V = self.inputs.structure.get_ase().get_volume()
        balance = (V0 / V) ** (1.0 / 3.0)
        relaxed_structure = scale_structure(
            structure=self.inputs.structure,
            scale_factor=orm.Float(balance),
        )
        self.out("relaxed_structure", relaxed_structure)
        self.out("volumes", orm.List(self.ctx.volumes))
        self.out("energies", orm.List(self.ctx.energies))
        self.out("fit_results", AttributeDict(self.ctx.fit_results))
