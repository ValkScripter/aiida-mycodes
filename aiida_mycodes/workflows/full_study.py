"""WorkChain to launch a full study of a material using Quantum ESPRESSO."""

# pylint: disable=no-member

import pathlib
import typing as ty

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
)
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from importlib_resources import files

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes import ProcessBuilder, ToContext, WorkChain

from aiida_mycodes.workflows.charge_density import (
    ChargeDensityWorkChain,
)
from aiida_mycodes.workflows.projwfcbands import MyProjwfcBandsWorkChain
from aiida_mycodes.workflows.xdos import XdosWorkChain

from . import protocols

__all__ = ["validate_inputs", "FullStudyWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs."""


def validate_charge_density_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of ChargeDensityWorkChain."""


def validate_projwfcbands_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of MyProjwfcBandsWorkChain."""
    if "nbands_factor" in inputs and "nbnd" in inputs["bands"]["pw"][
        "parameters"
    ].base.attributes.get("SYSTEM", {}):
        return (
            PwBandsWorkChain.exit_codes.ERROR_INVALID_INPUT_NUMBER_OF_BANDS.message
        )

    # Cannot specify both `bands_kpoints` and `bands_kpoints_distance`
    if all(
        key in inputs
        for key in ["bands_kpoints", "bands_kpoints_distance"]
    ):
        return (
            PwBandsWorkChain.exit_codes.ERROR_INVALID_INPUT_KPOINTS.message
        )


def validate_xdos_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of `XdosWorkChain`."""

    assert len(inputs["xmins"]) == len(
        inputs["xmaxs"]
    ), "Length of xmins and xmaxs should be equal"
    assert len(inputs["ymins"]) == len(
        inputs["ymaxs"]
    ), "Length of ymins and ymaxs should be equal"
    assert len(inputs["zmins"]) == len(
        inputs["zmaxs"]
    ), "Length of zmins and zmaxs should be equal"
    assert len(inputs["xmins"]) == len(
        inputs["ymins"]
    ), "Length of xmins and ymins should be equal"
    assert len(inputs["xmins"]) == len(
        inputs["zmins"]
    ), "Length of xmins and zmins should be equal"
    assert (
        len(inputs["xmins"]) > 0
    ), "At least one box should be defined"
    assert all(
        isinstance(x, (int, float)) for x in inputs["xmins"]
    ), "xmins should be a list of integers or floats"
    assert all(
        isinstance(x, (int, float)) for x in inputs["xmaxs"]
    ), "xmaxs should be a list of integers or floats"
    assert all(
        isinstance(x, (int, float)) for x in inputs["ymins"]
    ), "ymins should be a list of integers or floats"
    assert all(
        isinstance(x, (int, float)) for x in inputs["ymaxs"]
    ), "ymaxs should be a list of integers or floats"
    assert all(
        isinstance(x, (int, float)) for x in inputs["zmins"]
    ), "zmins should be a list of integers or floats"
    assert all(
        isinstance(x, (int, float)) for x in inputs["zmaxs"]
    ), "zmaxs should be a list of integers or floats"


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


class FullStudyWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain to launch a full study of a material using Quantum ESPRESSO.

    1. SCF calculation with PwBaseWorkChain
    2. Launches in parallel:
        - Charge density calculations with ChargeDensityWorkChain
        - Projected band structure calculations with MyProjwfcBandsWorkChain
        - DOS, PDOS and LDOS calculations with XdosWorkChain
    """

    # pylint: disable=inconsistent-return-statements

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="The input structure.",
        )

        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, work directories of all calculations will be cleaned at the end of the workchain.",
        )

        spec.input(
            "dry_run",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, will prepare the inputs for the calculations but not launch them.",
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=[
                "pw.structure",
                "pw.parent_folder",
                "clean_workdir",
            ],
        )

        spec.expose_inputs(
            ChargeDensityWorkChain,
            namespace="charge_density",
            exclude=["structure", "scf", "remote_scf_folder"],
            namespace_options={
                "help": "Inputs for the ChargeDensityWorkChain.",
                "required": False,
            },
        )

        spec.expose_inputs(
            MyProjwfcBandsWorkChain,
            namespace="projwfcbands",
            exclude=["structure", "scf", "remote_scf_folder"],
            namespace_options={
                "help": "Inputs for the MyProjwfcBandsWorkChain.",
                "required": False,
            },
        )

        spec.expose_inputs(
            XdosWorkChain,
            namespace="xdos",
            exclude=["structure", "scf"],
            namespace_options={
                "help": "Inputs for the XdosWorkChain.",
                "required": False,
            },
        )

        spec.inputs.validator = validate_inputs
        spec.inputs["charge_density"].validator = (
            validate_charge_density_inputs
        )
        spec.inputs["projwfcbands"].validator = (
            validate_projwfcbands_inputs
        )
        spec.inputs["xdos"].validator = validate_xdos_inputs

        spec.outline(
            cls.setup,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_all_parallel,
            cls.inspect_all_parallel,
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="The SCF sub process failed.",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_CHARGE_DENSITY",
            message="The ChargeDensity sub process failed.",
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_PROJWFCBANDS",
            message="The ProjwfcBands sub process failed.",
        )
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_XDOS",
            message="The Xdos sub process failed.",
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_ALL_BUT_CHARGE_DENSITY",
            message="All parallel sub processes failed except for the ChargeDensity.",
        )
        spec.exit_code(
            406,
            "ERROR_SUB_PROCESS_FAILED_ALL_BUT_PROJWFCBANDS",
            message="All parallel sub processes failed except for the ProjwfcBands.",
        )
        spec.exit_code(
            407,
            "ERROR_SUB_PROCESS_FAILED_ALL_BUT_XDOS",
            message="All parallel sub processes failed except for the Xdos.",
        )
        spec.exit_code(
            408,
            "ERROR_SUB_PROCESS_FAILED_ALL",
            message="All parallel sub processes failed.",
        )

        spec.expose_outputs(PwBaseWorkChain, namespace="scf")
        spec.expose_outputs(
            ChargeDensityWorkChain, namespace="charge_density"
        )
        spec.expose_outputs(
            MyProjwfcBandsWorkChain, namespace="projwfcbands"
        )
        spec.expose_outputs(XdosWorkChain, namespace="xdos")

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""

        return files(protocols) / "full_study.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code: ty.Union[str, int, orm.Code],
        dos_code: ty.Union[str, int, orm.Code],
        projwfc_code: ty.Union[str, int, orm.Code],
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
        :param dos_code: code for dos.x.
        :type dos_code: ty.Union[str, int, orm.Code]
        :param projwfc_code: code for projwfc.x.
        :type projwfc_code: ty.Union[str, int, orm.Code]
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

        # Type checks
        type_check(pw_code, (str, int, orm.Code))
        type_check(dos_code, (str, int, orm.Code))
        type_check(projwfc_code, (str, int, orm.Code))
        type_check(pp_code, (str, int, orm.Code))
        type_check(structure, orm.StructureData)
        type_check(protocol, str, allow_none=True)
        type_check(overrides, dict, allow_none=True)
        type_check(options, dict, allow_none=True)

        # Get protocol inputs
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Set SCF inputs
        args = (pw_code, structure, protocol)
        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("scf", None),
            options=options,
            **kwargs,
        )
        scf_builder["pw"].pop(
            "structure", None
        )  # Remove structure input
        scf_builder.pop(
            "clean_workdir", None
        )  # Remove clean_workdir input

        # Set charge density inputs
        args = (pw_code, pp_code, structure)
        charge_density_builder = (
            ChargeDensityWorkChain.get_builder_from_protocol(
                *args,
                protocol=protocol,
                overrides=inputs.get("charge_density", None),
                options=options,
                **kwargs,
            )
        )
        charge_density_builder.pop("scf", None)  # Remove SCF inputs
        charge_density_builder.pop(
            "structure", None
        )  # Remove structure input
        charge_density_builder.pop(
            "remote_scf_folder", None
        )  # Remove remote_scf_folder input

        # Set projected band structure inputs
        args = (pw_code, projwfc_code, structure)
        projwfcbands_builder = (
            MyProjwfcBandsWorkChain.get_builder_from_protocol(
                *args,
                protocol=protocol,
                overrides=inputs.get("projwfcbands", None),
                options=options,
                **kwargs,
            )
        )
        projwfcbands_builder.pop("scf", None)  # Remove SCF inputs
        projwfcbands_builder.pop(
            "structure", None
        )  # Remove structure input
        projwfcbands_builder.pop(
            "remote_scf_folder", None
        )  # Remove remote_scf_folder input

        # Set DOS, PDOS and LDOS inputs (XdosWorkChain)
        args = (pw_code, dos_code, projwfc_code, structure)
        xdos_builder = XdosWorkChain.get_builder_from_protocol(
            *args,
            protocol=protocol,
            overrides=inputs.get("xdos", None),
            options=options,
            **kwargs,
        )
        xdos_builder.pop("scf", None)  # Remove SCF inputs
        xdos_builder.pop("structure", None)  # Remove structure input

        # Construct builder
        builder = cls.get_builder()
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.dry_run = orm.Bool(inputs["dry_run"])
        builder.structure = structure
        builder.scf = scf_builder
        builder.charge_density = charge_density_builder
        builder.projwfcbands = projwfcbands_builder
        builder.xdos = xdos_builder

        return builder

    def setup(self):
        """Init required context variables."""
        self.ctx.dry_run = (
            "dry_run" in self.inputs and self.inputs.dry_run.value
        )

    def run_scf(self):
        """Run an SCF calculation, to generate the wavefunction."""
        # Get the inputs for the SCF calculation
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, "scf")
        )
        inputs.pw.structure = self.inputs.structure  # Set the structure

        inputs.metadata.call_link_label = "scf"  # Label the call link
        inputs = prepare_process_inputs(
            PwBaseWorkChain, inputs
        )  # Prepare the inputs

        if self.ctx.dry_run:
            return inputs

        future = self.submit(
            PwBaseWorkChain, **inputs
        )  # Submit the SCF calculation

        self.report(f"launching SCF PwBaseWorkChain<{future.pk}>")

        return ToContext(workchain_scf=future)

    def inspect_scf(self):
        """Inspect the SCF calculation."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f"SCF PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_parent_folder = workchain.outputs.remote_folder

    def run_all_parallel(self):
        """Run all parallel calculations."""
        # Get the inputs for the charge density calculation
        charge_density_inputs = AttributeDict(
            self.exposed_inputs(
                ChargeDensityWorkChain, "charge_density"
            )
        )
        charge_density_inputs.structure = self.inputs.structure
        charge_density_inputs.remote_scf_folder = (
            self.ctx.scf_parent_folder
        )

        charge_density_inputs.metadata.call_link_label = (
            "charge_density"
        )
        charge_density_inputs = prepare_process_inputs(
            ChargeDensityWorkChain, charge_density_inputs
        )

        # Get the inputs for the projected band structure calculation
        projwfcbands_inputs = AttributeDict(
            self.exposed_inputs(MyProjwfcBandsWorkChain, "projwfcbands")
        )
        projwfcbands_inputs.structure = self.inputs.structure
        projwfcbands_inputs.remote_scf_folder = (
            self.ctx.scf_parent_folder
        )

        projwfcbands_inputs.metadata.call_link_label = "projwfcbands"
        projwfcbands_inputs = prepare_process_inputs(
            MyProjwfcBandsWorkChain, projwfcbands_inputs
        )

        # Get the inputs for the DOS, PDOS and LDOS calculation
        xdos_inputs = AttributeDict(
            self.exposed_inputs(XdosWorkChain, "xdos")
        )
        xdos_inputs.structure = self.inputs.structure
        xdos_inputs.nscf.pw.parent_folder = self.ctx.scf_parent_folder
        xdos_inputs.nscf.pw.pseudos = (
            self.ctx.workchain_scf.inputs.pw.pseudos
        )

        xdos_inputs.metadata.call_link_label = "xdos"
        xdos_inputs = prepare_process_inputs(XdosWorkChain, xdos_inputs)

        if self.ctx.dry_run:
            return (
                charge_density_inputs,
                projwfcbands_inputs,
                xdos_inputs,
            )

        # Submit the calculations
        future_charge_density = self.submit(
            ChargeDensityWorkChain, **charge_density_inputs
        )
        self.report(
            f"launching ChargeDensityWorkChain<{future_charge_density.pk}>"
        )
        self.to_context(
            **{"workchain_charge_density": future_charge_density}
        )

        future_projwfcbands = self.submit(
            MyProjwfcBandsWorkChain, **projwfcbands_inputs
        )
        self.report(
            f"launching MyProjwfcBandsWorkChain<{future_projwfcbands.pk}>"
        )
        self.to_context(
            **{"workchain_projwfcbands": future_projwfcbands}
        )

        future_xdos = self.submit(XdosWorkChain, **xdos_inputs)
        self.report(f"launching XdosWorkChain<{future_xdos.pk}>")
        self.to_context(**{"workchain_xdos": future_xdos})

    def inspect_all_parallel(self):
        """Inspect all parallel calculations."""
        workchain_charge_density = self.ctx.workchain_charge_density
        workchain_projwfcbands = self.ctx.workchain_projwfcbands
        workchain_xdos = self.ctx.workchain_xdos

        error_codes = []

        if not workchain_charge_density.is_finished_ok:
            self.report(
                f"ChargeDensityWorkChain failed with exit status {workchain_charge_density.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_CHARGE_DENSITY
            )

        if not workchain_projwfcbands.is_finished_ok:
            self.report(
                f"MyProjwfcBandsWorkChain failed with exit status {workchain_projwfcbands.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFCBANDS
            )

        if not workchain_xdos.is_finished_ok:
            self.report(
                f"XdosWorkChain failed with exit status {workchain_xdos.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_XDOS
            )

        if len(error_codes) == 3:
            return (
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL
            )  # All parallel calculations failed
        if len(error_codes) == 2:
            if (
                not self.exit_codes.ERROR_SUB_PROCESS_FAILED_CHARGE_DENSITY
                in error_codes
            ):
                return (
                    self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL_BUT_CHARGE_DENSITY
                )  # All parallel calculations failed except for the ChargeDensity
            if (
                not self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFCBANDS
                in error_codes
            ):
                return (
                    self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL_BUT_PROJWFCBANDS
                )  # All parallel calculations failed except for the ProjwfcBands
            if (
                not self.exit_codes.ERROR_SUB_PROCESS_FAILED_XDOS
                in error_codes
            ):
                return (
                    self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL_BUT_XDOS
                )  # All parallel calculations failed except for the Xdos
        if len(error_codes) == 1:
            return error_codes[0]

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report("workchain successfully completed")

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_scf, PwBaseWorkChain, namespace="scf"
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_charge_density,
                ChargeDensityWorkChain,
                namespace="charge_density",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_projwfcbands,
                MyProjwfcBandsWorkChain,
                namespace="projwfcbands",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_xdos, XdosWorkChain, namespace="xdos"
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
