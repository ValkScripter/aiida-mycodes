"""Workchain to compute the electrostatic potential of a structure (along Z direction) using Quantum ESPRESSO."""

# pylint: disable=no-member, inconsistent-return-statements

from aiida_quantumespresso.calculations.average import (
    AverageCalculation,
)
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida import orm
from aiida.engine import ToContext, WorkChain

__all__ = ["ElectrostaticPotentialWorkChain"]


class ElectrostaticPotentialWorkChain(WorkChain):
    """
    Workchain to compute the planar averaged electrostatic potential
    and the macroscopic average potential of a structure (along Z direction) using Quantum ESPRESSO.

    Inputs:

        structure: the input structure
        pw_code: the code for the SCF PwBaseWorkChain
        pp_code: the code for the PP PpCalculation
        average_code: the code for the AverageCalculation
        npts: the number of points of interpolation
        window_size: the window size for the macroscopic average (Angstrom)
        scf_options: the options for the SCF PwBaseWorkChain
        scf_overrides: the overrides for the SCF PwBaseWorkChain
        pp_options: the options for the PP PpCalculation
        protocol: the protocol to use for the SCF PwBaseWorkChain

    Outputs:

        output_data: array with 3 columns:
                        z (Ang),
                        planar averaged potential (eV),
                        macroscopic averaged potential (eV)
        output_parameters: output_parameters of the SCF PwBaseWorkChain

    1.  Run a SCF calculation to get the charge density
    2.  Run a PP calculation to get the electrostatic potential in space
    3.  Run a AverageCalculation to compute the planar averaged potential
        and the macroscopic average potential along Z direction
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("pw_code", valid_type=orm.Code)
        spec.input("pp_code", valid_type=orm.Code)
        spec.input("average_code", valid_type=orm.Code)
        spec.input(
            "npts",
            valid_type=orm.Int,
            default=orm.Int(1000),
            help="Number of points of interpolation",
        )
        spec.input(
            "window_size",
            valid_type=orm.Float,
            help="Window size for the macroscopic average (Angstrom)",
        )
        spec.input("scf_options", valid_type=orm.Dict)
        spec.input("scf_overrides", valid_type=orm.Dict)
        spec.input("pp_options", valid_type=orm.Dict)
        spec.input(
            "protocol", valid_type=orm.Str, default=orm.Str("moderate")
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
        )

        spec.output("output_data", valid_type=orm.ArrayData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.outline(
            cls.setup,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_pp,
            cls.inspect_pp,
            cls.run_average,
            cls.inspect_average,
            cls.return_results,
            cls.on_termination,
        )

        # Error codes
        spec.exit_code(
            401,
            "ERROR_SCF_FAILED",
            message="SCF PwBaseWorkChain sub process failed",
        )
        spec.exit_code(
            402,
            "ERROR_PP_FAILED",
            message="PP PpCalculation sub process failed",
        )
        spec.exit_code(
            403,
            "ERROR_AVERAGE_FAILED",
            message="AverageCalculation sub process failed",
        )

    def setup(self):
        """Initialize context variables."""
        self.ctx.structure = self.inputs.structure
        self.ctx.pw_code = self.inputs.pw_code
        self.ctx.pp_code = self.inputs.pp_code
        self.ctx.average_code = self.inputs.average_code
        self.ctx.npts = self.inputs.npts.value
        self.ctx.window_size = self.inputs.window_size.value
        self.ctx.scf_options = self.inputs.scf_options.get_dict()
        self.ctx.scf_overrides = self.inputs.scf_overrides.get_dict()
        self.ctx.pp_options = self.inputs.pp_options.get_dict()
        self.ctx.protocol = self.inputs.protocol.value

        self.ctx.scf_options["clean_workdir"] = False
        self.ctx.pp_options["clean_workdir"] = False

    def run_scf(self):
        """Run the SCF calculation."""
        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=self.ctx.pw_code,
            structure=self.ctx.structure,
            protocol=self.ctx.protocol,
            overrides=self.ctx.scf_overrides,
            options=self.ctx.scf_options,
        )

        node = self.submit(scf_builder)
        self.report(
            f"launching PwBaseWorkChain<{node.pk}> for SCF calculation"
        )

        return ToContext(scf_workchain=node)

    def inspect_scf(self):
        """Verify the SCF calculation."""
        workchain = self.ctx.scf_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"SCF PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SCF_FAILED

        self.ctx.scf_remote_folder = workchain.outputs.remote_folder

    def run_pp(self):
        """Run the PP calculation."""
        pp_builder = PpCalculation.get_builder()
        pp_builder.code = self.ctx.pp_code
        pp_builder.parent_folder = self.ctx.scf_remote_folder
        pp_builder.metadata.options = self.ctx.pp_options

        parameters = {
            "INPUTPP": {
                "plot_num": 11,  # electrostatic potential
            },
            "PLOT": {
                "iflag": 3,  # 3D plot
            },
        }

        pp_builder.parameters = orm.Dict(parameters)

        node = self.submit(pp_builder)
        self.report(
            f"launching PpCalculation<{node.pk}> to get the electrostatic potential"
        )

        return ToContext(pp_calculation=node)

    def inspect_pp(self):
        """Verify the PP calculation."""
        pp_calculation = self.ctx.pp_calculation
        if not pp_calculation.is_finished_ok:
            self.report(
                f"PpCalculation failed with exit status {pp_calculation.exit_status}"
            )
            return self.exit_codes.ERROR_PP_FAILED

        self.ctx.pp_remote_folder = pp_calculation.outputs.remote_folder

    def run_average(self):
        """Run the AverageCalculation."""
        average_builder = AverageCalculation.get_builder()
        average_builder.code = self.ctx.average_code
        average_builder.parent_folder = self.ctx.pp_remote_folder
        average_builder.npts = self.inputs.npts
        average_builder.window_size = self.inputs.window_size
        average_builder.average_axis = orm.Int(3)  # Z direction
        options = self.ctx.pp_options.copy()
        options["resources"] = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        average_builder.metadata.options = options

        node = self.submit(average_builder)
        self.report(
            f"launching AverageCalculation<{node.pk}> to compute the averaged potentials"
        )

        return ToContext(average_calculation=node)

    def inspect_average(self):
        """Verify the AverageCalculation."""
        average_calculation = self.ctx.average_calculation
        if not average_calculation.is_finished_ok:
            self.report(
                f"AverageCalculation failed with exit status {average_calculation.exit_status}"
            )
            return self.exit_codes.ERROR_AVERAGE_FAILED

    def return_results(self):
        """Return the results."""
        self.out(
            "output_data",
            self.ctx.average_calculation.outputs.output_data,
        )
        self.out(
            "output_parameters",
            self.ctx.scf_workchain.outputs.output_parameters,
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
