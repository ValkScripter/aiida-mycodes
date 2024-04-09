"""WorkChain to study evolution of the bandstructure and projected bandstructures from an origin to a target structure"""

import pathlib

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
)
from aiida_wannier90_workflows.workflows.projwfcbands import (
    ProjwfcBandsWorkChain,
)
from aiida_wannier90_workflows.workflows.projwfcbands import (
    validate_inputs as validate_inputs_projwfcbands,
)
from importlib_resources import files
from numpy import linspace

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from . import protocols
from ..calculations.structure.tools import interpolate_structures

__all__ = ["validate_inputs", "ProjwfcBandsInterpolationWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `ProjwfcBandsInterpolationWorkChain`."""
    if inputs.nb_structures.value < 2:
        return "Number of interpolated structures must be >= 2."
    ase_origin = inputs.origin_structure.get_ase()
    ase_target = inputs.target_structure.get_ase()

    if len(ase_origin) != len(ase_target):
        return "`origin` and `target` structures must have the same number of atoms."

    if not (
        ase_origin.get_atomic_numbers()
        == ase_target.get_atomic_numbers()
    ).all():
        return "`origin` and `target` structures must have the same ordering of elements to interpolate."


class ProjwfcBandsInterpolationWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain to automatically compute ProjwfcBands for a set of interpolated structures
    between an origin and a target structure
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.input(
            "origin_structure",
            valid_type=orm.StructureData,
            help="The origin structure.",
        )
        spec.input(
            "target_structure",
            valid_type=orm.StructureData,
            help="The target structure.",
        )
        spec.input(
            "nb_structures",
            valid_type=orm.Int,
            help="The number of interpolated structures (>=2).",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(True),
            help=(
                "If True, work directories of all called calculation will be cleaned "
                "at the end of execution."
            ),
        )

        spec.expose_inputs(
            ProjwfcBandsWorkChain,
            namespace="projwfcbands",
            exclude=("structure", "clean_workdir", "relax"),
            namespace_options={
                "help": "Inputs for the `ProjwfcBandsWorkChain` for bandstructure and projections calculation."
            },
        )

        spec.inputs["projwfcbands"].validator = (
            validate_inputs_projwfcbands
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.interpolate_structures,
            cls.run_all_projwfcbands,
            cls.inspect_all_projwfcbands,
            cls.results,
            cls.on_terminated,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_PROJWFCBANDS",
            message="One or more PROJWFCBANDS sub processes failed",
        )

        spec.output_namespace(
            "output_parameters_scf",
            valid_type=orm.Dict,
            dynamic=True,
            help="The output parameters of each SCF ``PwBaseWorkChain`` performed.",
        )
        spec.output_namespace(
            "output_parameters_bands",
            valid_type=orm.Dict,
            dynamic=True,
            help="The output parameters of each BANDS ``PwBaseWorkChain`` performed.",
        )
        spec.output_namespace(
            "output_bandstructures",
            valid_type=orm.BandsData,
            dynamic=True,
            help="The output bandstructures of each SCF ``PwBaseWorkChain`` performed.",
        )
        spec.output_namespace(
            "output_projwfc",
            dynamic=True,
            help="The output results of each ``ProjwfcBaseWorkChain`` performed.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""

        return files(protocols) / "projwfcbands_interpolation.yaml"

    def setup(self):
        """Init required context variables."""
        self.ctx.interpolated_structures_list = []
        self.ctx.labels = []

    def interpolate_structures(self):
        """Call interpolate_structures calculation locally"""
        for index, balance in enumerate(
            linspace(0.0, 1.0, self.inputs.nb_structures.value)
        ):
            interpolated_structure = interpolate_structures(
                self.inputs.origin_structure,
                self.inputs.target_structure,
                orm.Float(balance),
            )
            self.ctx.interpolated_structures_list.append(
                interpolated_structure
            )
            self.ctx.labels.append(f"structure_{index}")
        pk_list = list(
            map(lambda s: s.pk, self.ctx.interpolated_structures_list)
        )
        self.report(f"Structures to process: {pk_list}")

    def run_all_projwfcbands(self):
        """Call all ProjwfcBandsWorkChain's for each interpolated structure"""
        futures = {}
        for index, structure in enumerate(
            self.ctx.interpolated_structures_list
        ):
            inputs = AttributeDict(
                self.exposed_inputs(
                    ProjwfcBandsWorkChain, namespace="projwfcbands"
                )
            )
            inputs.structure = structure
            inputs.metadata.call_link_label = f"PB{index}"
            inputs = prepare_process_inputs(
                ProjwfcBandsWorkChain, inputs
            )
            future = self.submit(ProjwfcBandsWorkChain, **inputs)
            label = self.ctx.labels[index]
            futures[label] = future
            self.report(
                f"launched ProjwfcBandsWorkChain for {label}<{future.pk}>"
            )
        return ToContext(**futures)

    # pylint: disable=inconsistent-return-statements
    def inspect_all_projwfcbands(self):
        """Check that all ProjwfcBandsWorkChain's finished successfully"""
        work_chains = [self.ctx[label] for label in self.ctx.labels]
        failed_work_chains = []
        for work_chain, label in zip(work_chains, self.ctx.labels):
            if not work_chain.is_finished_ok:
                failed_work_chains.append(work_chain)
                self.report(
                    f"ProjwfcBandsWorkChain for ({label}) failed with exit status {work_chain.exit_status}"
                )
        if len(failed_work_chains) > 0:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFCBANDS

    # pylint: enable=inconsistent-return-statements

    def results(self):
        """Compile all output data and send to outputs."""
        output_parameters_scf = {
            label: self.ctx[label].outputs.scf_parameters
            for label in self.ctx.labels
        }
        self.out("output_parameters_scf", output_parameters_scf)
        output_parameters_bands = {
            label: self.ctx[label].outputs.bands_parameters
            for label in self.ctx.labels
        }
        self.out("output_parameters_bands", output_parameters_bands)
        output_bandstructures = {
            label: self.ctx[label].outputs.band_structure
            for label in self.ctx.labels
        }
        self.out("output_bandstructures", output_bandstructures)
        output_projwfc = {
            label: self.ctx[label].outputs.projwfc
            for label in self.ctx.labels
        }
        self.out("output_projwfc", output_projwfc)

    def on_terminated(self):
        """Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs."""

        super().on_terminated()

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
