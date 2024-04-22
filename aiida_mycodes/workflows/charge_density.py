"""WorkChain to study charge densities of bands around Fermi level"""

# pylint: disable=no-member

import pathlib
import typing as ty

from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
    recursive_merge,
)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from importlib_resources import files
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes import (
    ProcessBuilder,
    ToContext,
    WorkChain,
    if_,
)
from aiida.orm.nodes.data.base import to_aiida_type

from . import protocols

# import warnings


__all__ = ["validate_inputs", "ChargeDensityWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `ChargeDensityWorkChain`."""
    if not ("scf" in inputs) ^ ("remote_scf_folder" in inputs):
        return (
            "Specify either the `scf` or the `remote_scf_folder` input."
        )

    if "scf" in inputs and not "structure" in inputs:
        return "Structure input is required if SCF calculation is provided."


def validate_pp(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the cd_2D and cd_3D namespaces of `ChargeDensityWorkChain`."""


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


class ChargeDensityWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain to calculate charge densities in 2D cuts and in 3D of a set of bands around Fermi energy at a specific k-point
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="The input structure.",
            required=False,
        )
        spec.input(
            "kpoint_index",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="The index (1 being the first) of the kpoint (from the SCF calculation)\
                  to use to calculate charge densities.",
        )
        spec.input(
            "first_band_index",
            valid_type=orm.Int,
            default=lambda: orm.Int(-1),
            help="The index of the first band to compute charge densities.\
                  The index is relative to the HOMO band at given kpoint:\n\
                -1 -> band below HOMO\n\
                0 -> HOMO band\n\
                1 -> LUMO band, etc.",
        )  # it will read the occupations at the given kpoint from the SCF results, unoccupancy threshold being 0.05.
        spec.input(
            "last_band_index",
            valid_type=orm.Int,
            default=lambda: orm.Int(-1),
            help="The index of the last band to compute charge densities.\
                  The index is relative to the HOMO band at given kpoint:\n\
                -1 -> band below HOMO\n\
                0 -> HOMO band\n\
                1 -> LUMO band, etc.",
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
        spec.input(
            "dry_run",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help="Terminate workchain steps before submitting calculations (test purposes only).",
        )

        spec.input(
            "remote_scf_folder",
            valid_type=orm.RemoteData,
            serializer=to_aiida_type,
            required=False,
            help="Specify this to retrieve results if no SCF calculation is to be done.",
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=(
                "clean_workdir",
                "pw.structure",
                "pw.parent_folder",
            ),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for scf calculation.",
                "required": False,
                "populate_defaults": False,
            },
        )

        spec.expose_inputs(
            PpCalculation,
            namespace="cd_2D",
            exclude=(
                "clean_workdir",
                "parent_folder",
            ),
            namespace_options={
                "help": "Inputs for the `PpCalculation` for the 2D charge density calculation (gnuplot).",
                "populate_defaults": False,
            },
        )

        spec.expose_inputs(
            PpCalculation,
            namespace="cd_3D",
            exclude=(
                "clean_workdir",
                "parent_folder",
            ),
            namespace_options={
                "help": "Inputs for the `PpCalculation` for the 3D charge density calculation (cube).",
                "populate_defaults": False,
            },
        )

        spec.inputs.validator = validate_inputs
        spec.inputs["cd_2D"].validator = validate_pp
        spec.inputs["cd_3D"].validator = validate_pp

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_all_cd,
            cls.inspect_all_cd,
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="SCF WorkChain sub process failed ",
        )

        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_CD_2D",
            message="2D charge density calculation sub process failed",
        )

        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_CD_3D",
            message="3D charge density calculation sub process failed",
        )

        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_CD_BOTH",
            message="Both charge density calculation sub processes failed",
        )

        spec.expose_outputs(PpCalculation, namespace="cd_2D")
        spec.expose_outputs(PpCalculation, namespace="cd_3D")

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""

        return files(protocols) / "charge_density.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ,unused-argument
        cls,
        pw_code: ty.Union[str, int, orm.Code],
        pp_code: ty.Union[str, int, orm.Code],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        options: dict = None,
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :param pw_code: code for pw.x.
        :type pw_code: ty.Union[str, int, orm.Code]
        :param pp_code: code for projwfc.x.
        :type pp_code: ty.Union[str, int, orm.Code]
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        # pylint: disable=too-many-locals,too-many-function-args
        type_check(pw_code, (str, int, orm.Code))
        type_check(pp_code, (str, int, orm.Code))
        type_check(structure, orm.StructureData)
        type_check(protocol, str, allow_none=True)
        type_check(overrides, dict, allow_none=True)

        inputs = cls.get_protocol_inputs(
            protocol=protocol, overrides=overrides
        )

        args = (pw_code, structure, protocol)
        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("scf", None),
            options=options,
            **kwargs,
        )
        scf_builder["pw"].pop("structure", None)
        scf_builder.pop("clean_workdir", None)

        metadata_cd_2D = inputs.get("cd_2D", {}).get(
            "metadata", {"options": {}}
        )
        metadata_cd_3D = inputs.get("cd_3D", {}).get(
            "metadata", {"options": {}}
        )

        if options:
            metadata_cd_2D["options"] = recursive_merge(
                metadata_cd_2D["options"], options
            )
            metadata_cd_3D["options"] = recursive_merge(
                metadata_cd_3D["options"], options
            )

        builder = cls.get_builder()
        builder.clean_workdir = orm.Bool(
            inputs.get("clean_workdir", True)
        )
        builder.dry_run = orm.Bool(inputs.get("dry_run", False))
        builder.kpoint_index = orm.Int(inputs.get("kpoint_index", 1))
        builder.first_band_index = orm.Int(
            inputs.get("first_band_index", 0)
        )
        builder.last_band_index = orm.Int(
            inputs.get("last_band_index", 1)
        )
        builder.structure = structure
        builder.scf = scf_builder
        builder.cd_2D.code = pp_code
        parameters = inputs.get("cd_2D", {}).get("parameters")
        parameters["INPUTPP"]["kpoint"] = builder.kpoint_index
        builder.cd_2D.parameters = orm.Dict(parameters)
        builder.cd_2D.metadata = metadata_cd_2D
        builder.cd_3D.code = pp_code
        parameters = inputs.get("cd_3D", {}).get("parameters")
        parameters["INPUTPP"]["kpoint"] = builder.kpoint_index
        builder.cd_3D.parameters = orm.Dict(parameters)
        builder.cd_3D.metadata = metadata_cd_3D

        return builder

    def setup(self):
        """Init required context variables."""
        self.ctx.dry_run = (
            "dry_run" in self.inputs and self.inputs.dry_run.value
        )
        if "remote_scf_folder" in self.inputs:
            self.ctx.scf_parent_folder = self.inputs.remote_scf_folder

    def should_run_scf(self):
        """Return whether the work chain should run an SCF calculation."""
        return "scf" in self.inputs

    def run_scf(self):
        """Run an SCF calculation, to generate the wavefunction."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, "scf")
        )
        inputs.pw.structure = self.inputs.structure

        inputs.metadata.call_link_label = "scf"
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(PwBaseWorkChain, **inputs)

        self.report(f"launching SCF PwBaseWorkChain<{future.pk}>")

        return ToContext(workchain_scf=future)

    # pylint: disable=inconsistent-return-statements
    def inspect_scf(self):
        """Verify that the SCF calculation finished successfully."""
        workchain = self.ctx.workchain_scf
        if not workchain.is_finished_ok:
            self.report(
                f"SCF PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_parent_folder = workchain.outputs.remote_folder
        self.ctx.homo_index = np.argmin(
            workchain.outputs.output_band.get_array("occupations")[
                self.inputs.kpoint_index.value - 1
            ]
            > 0.05
        )  # in 1-numeration

    def run_all_cd(self):
        """Call all PpCalculation's for cd_2D and cd_3D"""
        cd_2D_inputs = AttributeDict(
            self.exposed_inputs(PpCalculation, namespace="cd_2D")
        )
        cd_3D_inputs = AttributeDict(
            self.exposed_inputs(PpCalculation, namespace="cd_3D")
        )
        if self.should_run_scf():
            cd_2D_inputs.parent_folder = self.ctx.scf_parent_folder
            cd_3D_inputs.parent_folder = self.ctx.scf_parent_folder

        cd_2D_parameters = self.inputs.cd_2D.parameters.get_dict()
        cd_3D_parameters = self.inputs.cd_3D.parameters.get_dict()

        occupations = cd_2D_inputs.parent_folder.creator.outputs.output_band.get_array(
            "occupations"
        )

        self.ctx.homo_index = np.argmin(
            occupations[self.inputs.kpoint_index.value - 1] > 0.05
        )  # in 1-numeration

        cd_2D_parameters["INPUTPP"]["kband(1)"] = (
            self.ctx.homo_index + self.inputs.first_band_index
        )
        cd_2D_parameters["INPUTPP"]["kband(2)"] = (
            self.ctx.homo_index + self.inputs.last_band_index
        )
        cd_3D_parameters["INPUTPP"]["kband(1)"] = (
            self.ctx.homo_index + self.inputs.first_band_index
        )
        cd_3D_parameters["INPUTPP"]["kband(2)"] = (
            self.ctx.homo_index + self.inputs.last_band_index
        )

        cd_2D_inputs.parameters = orm.Dict(cd_2D_parameters)
        cd_3D_inputs.parameters = orm.Dict(cd_3D_parameters)
        cd_2D_inputs["metadata"]["call_link_label"] = "cd_2D"
        cd_3D_inputs["metadata"]["call_link_label"] = "cd_3D"

        if self.ctx.dry_run:
            return cd_2D_inputs, cd_3D_inputs

        future_2D = self.submit(PpCalculation, **cd_2D_inputs)
        self.report(f"launching cd_2D PpCalculation<{future_2D.pk}>")
        self.to_context(**{"cd_2D": future_2D})

        future_3D = self.submit(PpCalculation, **cd_3D_inputs)
        self.report(f"launching cd_3D PpCalculation<{future_3D.pk}>")
        self.to_context(**{"cd_3D": future_3D})

    def inspect_all_cd(self):
        """Check that all PpCalculation's finished successfully"""
        error_codes = []

        calculation = self.ctx.cd_2D
        if not calculation.is_finished_ok:
            self.report(
                f"PpCalculation for cd_2D failed with exit status {calculation.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_CD_2D
            )

        calculation = self.ctx.cd_3D
        if not calculation.is_finished_ok:
            self.report(
                f"PpCalculation for cd_3D failed with exit status {calculation.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_CD_3D
            )

        if len(error_codes) > 1:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CD_BOTH
        if len(error_codes) == 1:
            return error_codes[0]

    # pylint: enable=inconsistent-return-statements

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report("workchain successfully completed")

        self.out_many(
            self.exposed_outputs(
                self.ctx.cd_2D, PpCalculation, namespace="cd_2D"
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.cd_3D, PpCalculation, namespace="cd_3D"
            )
        )

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
