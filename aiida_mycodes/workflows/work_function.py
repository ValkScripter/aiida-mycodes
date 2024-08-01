"""Workchain to compute the workfunction of a surface using Quantum ESPRESSO."""

# pylint: disable=no-member, inconsistent-return-statements

import numpy as np

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction

from aiida_mycodes.workflows.electrostatic_potential import (
    ElectrostaticPotentialWorkChain,
)

__all__ = ["WorkFunctionWorkChain"]


@calcfunction
def get_fermi_energy(output_parameters):
    """Extract the Fermi energy from the SCF output_parameters."""
    return orm.Float(output_parameters.get_dict()["fermi_energy"])


@calcfunction
def get_vacuum_level(output_data, vacuum_width, slab):
    """Compute the vacuum level of the slab."""
    ase_slab = slab.get_ase()
    vacuum_width = vacuum_width.value
    potentials = output_data.get_array()
    zs = potentials[:, 0]
    ms = potentials[:, 2]

    Z = ase_slab.cell[2, 2]

    mask = np.logical_or(
        zs < vacuum_width / 6, zs > Z - vacuum_width / 6
    )  # to take the average only in the central 1/3 of the vacuum region
    vacuum_level = np.mean(ms[mask])

    return orm.Float(vacuum_level)


@calcfunction
def get_interior_average_slab(output_data, vacuum_width, slab):
    """Compute the interior macroscopic average potential of the slab."""
    ase_slab = slab.get_ase()
    vacuum_width = vacuum_width.value
    potentials = output_data.get_array()
    zs = potentials[:, 0]
    ms = potentials[:, 2]

    Z = ase_slab.cell[2, 2]

    mask = np.logical_and(
        zs > Z / 2 - (Z - vacuum_width) / 6,
        zs < Z / 2 + (Z - vacuum_width) / 6,
    )
    interior_average = np.mean(ms[mask])

    return orm.Float(interior_average)


@calcfunction
def get_interior_average_bulk(output_data):
    """Compute the interior macroscopic average potential of the bulk."""
    potentials = output_data.get_array()
    ms = potentials[:, 2]

    interior_average = np.mean(ms)

    return orm.Float(interior_average)


@calcfunction
def get_workfunction_1(fermi_energy_slab, vacuum_level_slab):
    """Compute the workfunction using the first method."""
    workfunction_1 = vacuum_level_slab.value - fermi_energy_slab.value

    return orm.Float(workfunction_1)


@calcfunction
def get_workfunction_2(
    fermi_energy_bulk,
    vacuum_level_slab,
    interior_average_slab,
    interior_average_bulk,
):
    """Compute the workfunction using the second method."""
    workfunction_2 = (
        vacuum_level_slab.value
        - interior_average_slab.value
        + interior_average_bulk.value
        - fermi_energy_bulk.value
    )

    return orm.Float(workfunction_2)


class WorkFunctionWorkChain(WorkChain):
    """
    Workchain to compute the workfunction of a surface using Quantum ESPRESSO.
    It will use 2 methods to compute the workfunction:
    1.  The difference between the Fermi level of the slab and the vacuum level of the slab
    2.  The difference between the Fermi level of the bulk and the vacuum level of the slab,
        with a interior macroscopic average potential referenced to the bulk

    Inputs:

        slab: the slab structure centered in the cell in the Z direction
        slab_bulk: the slab-consistent bulk structure
        pw_code: the code for the SCF PwBaseWorkChain
        pp_code: the code for the PP PpCalculation
        average_code: the code for the AverageCalculation
        npts: the number of points of interpolation for the electrostatic potential
        window_size: the window size for the macroscopic average (Angstrom),
            needs to be a multiple of interlayer separation
        vacuum_width: the width of the vacuum region (Angstrom)
        scf_options_slab: the options for the slab SCF PwBaseWorkChain
        scf_options_bulk: the options for the bulk SCF PwBaseWorkChain
        scf_overrides_slab: the overrides for the slab SCF PwBaseWorkChain
        scf_overrides_bulk: the overrides for the bulk SCF PwBaseWorkChain
        pp_options: the options for the PP PpCalculation
        protocol: the protocol to use for the SCF PwBaseWorkChains
        clean_workdir: whether to clean the remote folder of all workchains and calculations after completion

    Outputs:

        workfunction_1: the workfunction computed using the first method
        workfunction_2: the workfunction computed using the second method
        potential_slab: the averaged potentials of the slab
        potential_bulk: the averaged potentials of the bulk
        fermi_energy_slab: the Fermi energy of the slab
        fermi_energy_bulk: the Fermi energy of the bulk
        vacuum_level_slab: the vacuum level of the slab
        interior_average_slab: the interior macroscopic average potential of the slab
        interior_average_bulk: the interior macroscopic average potential of the bulk

    1. Run in parallel the ElectrostaticPotentialWorkChain for the slab and the bulk
    2. Extract the averaged potentials, Fermi energies and vacuum levels
    3. Compute the workfunction using the 2 methods
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "slab",
            valid_type=orm.StructureData,
            help="The slab structure centered in the cell in the Z direction",
        )
        spec.input(
            "slab_bulk",
            valid_type=orm.StructureData,
            help="The slab-consistent bulk structure",
        )
        spec.input(
            "pw_code",
            valid_type=orm.Code,
            help="The code for the SCF PwBaseWorkChain",
        )
        spec.input(
            "pp_code",
            valid_type=orm.Code,
            help="The code for the PP PpCalculation",
        )
        spec.input(
            "average_code",
            valid_type=orm.Code,
            help="The code for the AverageCalculation",
        )
        spec.input(
            "npts",
            valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help="Number of points of interpolation for the electrostatic potential",
        )
        spec.input(
            "window_size",
            valid_type=orm.Float,
            default=lambda: orm.Float(6.0),
            help="Window size for the macroscopic average (Angstrom), needs to be a multiple of interlayer separation",
        )
        spec.input(
            "vacuum_width",
            valid_type=orm.Float,
            default=lambda: orm.Float(15.0),
            help="Width of the vacuum region (Angstrom)",
        )
        spec.input(
            "scf_options_slab",
            valid_type=orm.Dict,
            help="The options for the slab SCF PwBaseWorkChain",
        )
        spec.input(
            "scf_options_bulk",
            valid_type=orm.Dict,
            help="The options for the bulk SCF PwBaseWorkChain",
        )
        spec.input(
            "scf_overrides_slab",
            valid_type=orm.Dict,
            help="The overrides for the slab SCF PwBaseWorkChain",
        )
        spec.input(
            "scf_overrides_bulk",
            valid_type=orm.Dict,
            help="The overrides for the bulk SCF PwBaseWorkChain",
        )
        spec.input(
            "pp_options",
            valid_type=orm.Dict,
            help="The options for the PP PpCalculation",
        )
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("moderate"),
            help="The protocol to use for the SCF PwBaseWorkChains",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Whether to clean the remote folder of all workchains and calculations after completion",
        )

        spec.outline(
            cls.setup,
            cls.submit_all_electrostatic_potential,
            cls.inspect_all_electrostatic_potential,
            cls.extract_quantities,
            cls.compute_workfunction,
            cls.return_results,
            cls.on_termination,
        )

        spec.output(
            "workfunction_1",
            valid_type=orm.Float,
            help="The workfunction computed using the first method",
        )
        spec.output(
            "workfunction_2",
            valid_type=orm.Float,
            help="The workfunction computed using the second method",
        )
        spec.output(
            "potentials_slab",
            valid_type=orm.ArrayData,
            help="The averaged potentials of the slab",
        )
        spec.output(
            "potentials_bulk",
            valid_type=orm.ArrayData,
            help="The averaged potentials of the bulk",
        )
        spec.output(
            "fermi_energy_slab",
            valid_type=orm.Float,
            help="The Fermi energy of the slab",
        )
        spec.output(
            "fermi_energy_bulk",
            valid_type=orm.Float,
            help="The Fermi energy of the bulk",
        )
        spec.output(
            "vacuum_level_slab",
            valid_type=orm.Float,
            help="The vacuum level of the slab",
        )
        spec.output(
            "interior_average_slab",
            valid_type=orm.Float,
            help="The interior macroscopic average potential of the slab",
        )
        spec.output(
            "interior_average_bulk",
            valid_type=orm.Float,
            help="The interior macroscopic average potential of the bulk",
        )

        # Exit codes
        spec.exit_code(
            401,
            "ERROR_SLAB_FAILED",
            message="The slab ElectrostaticPotentialWorkChain failed.",
        )
        spec.exit_code(
            402,
            "ERROR_BULK_FAILED",
            message="The bulk ElectrostaticPotentialWorkChain failed.",
        )

    def setup(self):
        """Define the context variables and perform some initial validation."""
        self.ctx.workfunction_1 = None
        self.ctx.workfunction_2 = None
        self.ctx.potentials_slab = None
        self.ctx.potentials_bulk = None
        self.ctx.fermi_energy_slab = None
        self.ctx.fermi_energy_bulk = None
        self.ctx.vacuum_level_slab = None
        self.ctx.interior_average_slab = None
        self.ctx.interior_average_bulk = None

    def submit_all_electrostatic_potential(self):
        """Run in parallel the ElectrostaticPotentialWorkChain for the slab and the bulk."""
        slab_inputs = {
            "structure": self.inputs.slab,
            "pw_code": self.inputs.pw_code,
            "pp_code": self.inputs.pp_code,
            "average_code": self.inputs.average_code,
            "npts": self.inputs.npts,
            "window_size": self.inputs.window_size,
            "scf_options": self.inputs.scf_options_slab,
            "scf_overrides": self.inputs.scf_overrides_slab,
            "pp_options": self.inputs.pp_options,
            "protocol": self.inputs.protocol,
            "clean_workdir": self.inputs.clean_workdir,
        }

        bulk_inputs = {
            "structure": self.inputs.slab_bulk,
            "pw_code": self.inputs.pw_code,
            "pp_code": self.inputs.pp_code,
            "average_code": self.inputs.average_code,
            "npts": self.inputs.npts,
            "window_size": self.inputs.window_size,
            "scf_options": self.inputs.scf_options_bulk,
            "scf_overrides": self.inputs.scf_overrides_bulk,
            "pp_options": self.inputs.pp_options,
            "protocol": self.inputs.protocol,
            "clean_workdir": self.inputs.clean_workdir,
        }

        slab_workchain = self.submit(
            ElectrostaticPotentialWorkChain, **slab_inputs
        )
        self.report(
            f"launching ElectrostaticPotentialWorkChain<{slab_workchain.pk}> for the slab"
        )

        bulk_workchain = self.submit(
            ElectrostaticPotentialWorkChain, **bulk_inputs
        )
        self.report(
            f"launching ElectrostaticPotentialWorkChain<{bulk_workchain.pk}> for the bulk"
        )

        futures = {
            "slab_workchain": slab_workchain,
            "bulk_workchain": bulk_workchain,
        }

        return ToContext(**futures)

    def inspect_all_electrostatic_potential(self):
        """Verify the ElectrostaticPotentialWorkChains finished successfully."""
        slab_workchain = self.ctx.slab_workchain
        if not slab_workchain.is_finished_ok:
            self.report(
                f"slab ElectrostaticPotentialWorkChain failed with exit status {slab_workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SLAB_FAILED

        bulk_workchain = self.ctx.bulk_workchain
        if not bulk_workchain.is_finished_ok:
            self.report(
                f"bulk ElectrostaticPotentialWorkChain failed with exit status {bulk_workchain.exit_status}"
            )
            return self.exit_codes.ERROR_BULK_FAILED

    def extract_quantities(self):
        """Extract the averaged potentials, Fermi energies and vacuum levels."""
        slab_workchain = self.ctx.slab_workchain
        bulk_workchain = self.ctx.bulk_workchain

        self.ctx.potentials_slab = slab_workchain.outputs.output_data
        self.ctx.potentials_bulk = bulk_workchain.outputs.output_data

        self.ctx.fermi_energy_slab = get_fermi_energy(
            slab_workchain.outputs.output_parameters
        )
        self.ctx.fermi_energy_bulk = get_fermi_energy(
            bulk_workchain.outputs.output_parameters
        )

        self.ctx.vacuum_level_slab = get_vacuum_level(
            slab_workchain.outputs.output_data,
            self.inputs.vacuum_width,
            self.inputs.slab,
        )

        self.ctx.interior_average_slab = get_interior_average_slab(
            slab_workchain.outputs.output_data,
            self.inputs.vacuum_width,
            self.inputs.slab,
        )

        self.ctx.interior_average_bulk = get_interior_average_bulk(
            bulk_workchain.outputs.output_data
        )

        self.report("Extracted all quantities")

    def compute_workfunction(self):
        """Compute the workfunction using the 2 methods."""
        self.ctx.workfunction_1 = get_workfunction_1(
            self.ctx.fermi_energy_slab, self.ctx.vacuum_level_slab
        )

        self.ctx.workfunction_2 = get_workfunction_2(
            self.ctx.fermi_energy_bulk,
            self.ctx.vacuum_level_slab,
            self.ctx.interior_average_slab,
            self.ctx.interior_average_bulk,
        )

        self.report(
            f"Workfunction 1: {self.ctx.workfunction_1.value} eV"
        )
        self.report(
            f"Workfunction 2: {self.ctx.workfunction_2.value} eV"
        )

    def return_results(self):
        """Return the results."""
        self.out("workfunction_1", self.ctx.workfunction_1)
        self.out("workfunction_2", self.ctx.workfunction_2)
        self.out("potentials_slab", self.ctx.potentials_slab)
        self.out("potentials_bulk", self.ctx.potentials_bulk)
        self.out("fermi_energy_slab", self.ctx.fermi_energy_slab)
        self.out("fermi_energy_bulk", self.ctx.fermi_energy_bulk)
        self.out("vacuum_level_slab", self.ctx.vacuum_level_slab)
        self.out(
            "interior_average_slab", self.ctx.interior_average_slab
        )
        self.out(
            "interior_average_bulk", self.ctx.interior_average_bulk
        )

    def on_termination(self):
        """Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs."""

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
