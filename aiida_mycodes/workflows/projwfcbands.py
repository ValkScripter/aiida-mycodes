"""WorkChain to automatically calculate QE projected band structure with optional SCF."""

# pylint: disable=no-member, inconsistent-return-statements

import typing as ty

from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
)
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.base.projwfc import (
    ProjwfcBaseWorkChain,
)
from importlib_resources import files

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

__all__ = ["validate_inputs", "MyProjwfcBandsWorkChain"]


def validate_inputs(
    inputs, ctx=None
):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace."""

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

    if not ("scf" in inputs) ^ ("remote_scf_folder" in inputs):
        return (
            "Specify either the `scf` or the `remote_scf_folder` input."
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


class MyProjwfcBandsWorkChain(ProtocolMixin, WorkChain):
    """WorkChain to compute QE projected band structure for a given structure with optional SCF."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # yapf: disable
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for the SCF calculation.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="bands",
            exclude=(
                "clean_workdir",
                "pw.structure",
                "pw.kpoints",
                "pw.kpoints_distance",
                "pw.parent_folder",
            ),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for the BANDS calculation."
            },
        )
        spec.input(
            "structure", valid_type=orm.StructureData, help="The inputs structure."
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, work directories of all called calculation will be cleaned at the end of execution.",
        )
        spec.input(
            "nbands_factor",
            valid_type=orm.Float,
            required=False,
            help="The number of bands for the BANDS calculation is that used for the SCF multiplied by this factor.",
        )
        spec.input(
            "bands_kpoints",
            valid_type=orm.KpointsData,
            required=False,
            help="Explicit kpoints to use for the BANDS calculation. Specify either this or `bands_kpoints_distance`.",
        )
        spec.input(
            "bands_kpoints_distance",
            valid_type=orm.Float,
            required=False,
            help="Minimum kpoints distance for the BANDS calculation. Specify either this or `bands_kpoints`.",
        )

        spec.input(
            "remote_scf_folder",
            valid_type=orm.RemoteData,
            serializer=to_aiida_type,
            required=False,
            help="Specify this to retrieve results if no SCF calculation is to be done.",
        )

        spec.expose_inputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            exclude=("clean_workdir", "projwfc.parent_folder"),
            namespace_options={
                "help": "Inputs for the `ProjwfcBaseWorkChain` for the projwfc.x calculation."
            },
        )

        spec.input(
            "Emin",
            valid_type=orm.Float,
            required=True,
            help="Minimum energy relative to Fermi energy for the Projwfc calculation.",
        )

        spec.input(
            "Emax",
            valid_type=orm.Float,
            required=True,
            help="Maximum energy relative to Fermi energy for the Projwfc calculation.",
        )

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ).else_(
                cls.retrieve_scf,
            ),
            cls.run_bands,
            cls.inspect_bands,
            cls.run_projwfc,
            cls.inspect_projwfc,
            cls.results,
        )
        spec.exit_code(
            201,
            'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `bands.pw.parameters.SYSTEM.nbnd`.'
        )
        spec.exit_code(
            202,
            'ERROR_INVALID_INPUT_KPOINTS',
            message='Cannot specify both `bands_kpoints` and `bands_kpoints_distance`.'
        )
        spec.exit_code(
            401,
            'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBaseWorkChain sub process failed'
        )
        spec.exit_code(
            402,
            'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='The bands PwBaseWorkChain sub process failed'
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_PROJWFC",
            message="The ProjwfcBaseWorkChain sub process failed",
        )

        spec.output('primitive_structure', valid_type=orm.StructureData,
            required=False,
            help='The normalized and primitivized structure for which the bands are computed.')
        spec.output('seekpath_parameters', valid_type=orm.Dict,
            required=False,
            help='The parameters used in the SeeKpath call to normalize the input or relaxed structure.')
        spec.output('scf_parameters', valid_type=orm.Dict,
            help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output('band_parameters', valid_type=orm.Dict,
            help='The output parameters of the BANDS `PwBaseWorkChain`.')
        spec.output('band_structure', valid_type=orm.BandsData,
            help='The computed band structure.')
        spec.expose_outputs(ProjwfcBaseWorkChain, namespace="projwfc")
        # yapf: enable

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        return files(protocols) / "projwfcbands.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code: ty.Union[str, int, orm.Code],
        projwfc_code: ty.Union[str, int, orm.Code],
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
        :param projwfc_code: code for projwfc.x.
        :type projwfc_code: ty.Union[str, int, orm.Code]
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param protocol: the protocol to use.
        :type protocol: str
        :param overrides: a dictionary to override the default inputs.
        :type overrides: dict
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :type options: dict
        :param kwargs: additional keyword arguments.
        :return: the builder.
        :rtype: ProcessBuilder
        """
        # pylint: disable=too-many-function-args
        type_check(pw_code, (str, int, orm.Code))
        type_check(projwfc_code, (str, int, orm.Code))
        type_check(structure, orm.StructureData)
        type_check(protocol, str, allow_none=True)
        type_check(overrides, dict, allow_none=True)
        type_check(options, dict, allow_none=True)

        # Prepare workchain builder
        builder = cls.get_builder()

        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)

        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("scf", None),
            options=options,
            **kwargs,
        )
        bands = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("bands", None),
            options=options,
            **kwargs,
        )
        projwfc = ProjwfcBaseWorkChain.get_builder_from_protocol(
            projwfc_code,
            protocol=protocol,
            overrides=inputs.get("projwfc", None),
            # **kwargs,
        )

        scf["pw"].pop("structure", None)
        scf.pop("clean_workdir", None)
        bands["pw"].pop("structure", None)
        bands.pop("clean_workdir", None)
        bands.pop("kpoints_distance", None)
        bands.pop("kpoints_force_parity", None)
        projwfc.pop("clean_workdir", None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.scf = scf
        builder.bands = bands
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.nbands_factor = orm.Float(inputs["nbands_factor"])
        builder.bands_kpoints_distance = orm.Float(
            inputs["bands_kpoints_distance"]
        )
        builder.projwfc = projwfc
        builder.Emin = orm.Float(inputs["Emin"])
        builder.Emax = orm.Float(inputs["Emax"])

        return builder

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.current_number_of_bands = None
        self.ctx.bands_kpoints = self.inputs.get("bands_kpoints", None)

    def should_run_seekpath(self):
        """Seekpath should only be run if the `bands_kpoints` input is not specified."""
        return "bands_kpoints" not in self.inputs

    def should_run_scf(self):
        """SCF should only be run if the `remote_scf_folder` input is not specified and 'scf' is in the inputs."""
        return (
            "remote_scf_folder" not in self.inputs
            and "scf" in self.inputs
        )

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the normalized structure and path along high-symmetry k-points .

        This is only called if the `bands_kpoints` input was not specified.
        """
        inputs = {
            "reference_distance": self.inputs.get(
                "bands_kpoints_distance", None
            ),
            "metadata": {"call_link_label": "seekpath"},
        }
        result = seekpath_structure_analysis(
            self.ctx.current_structure, **inputs
        )
        self.ctx.current_structure = result["primitive_structure"]
        self.ctx.bands_kpoints = result["explicit_kpoints"]

        self.out("primitive_structure", result["primitive_structure"])
        self.out("seekpath_parameters", result["parameters"])

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace="scf")
        )
        inputs.metadata.call_link_label = "scf"
        inputs.pw.structure = self.ctx.current_structure

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            f"launching PwBaseWorkChain<{running.pk}> in scf mode"
        )

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f"scf PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = (
            workchain.outputs.output_parameters.base.attributes.get(
                "number_of_bands"
            )
        )
        self.ctx.fermi_energy = (
            workchain.outputs.output_parameters.dict.fermi_energy
        )

    def retrieve_scf(self):
        """Retrieve some output data from the SCF calculation."""
        workchain = self.inputs.remote_scf_folder.creator
        if not workchain.is_finished_ok:
            self.report(
                f"scf PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        self.ctx.workchain_scf = workchain
        self.ctx.current_folder = self.inputs.remote_scf_folder
        self.ctx.current_number_of_bands = (
            workchain.outputs.output_parameters.base.attributes.get(
                "number_of_bands"
            )
        )
        self.ctx.fermi_energy = (
            workchain.outputs.output_parameters.dict.fermi_energy
        )

    def run_bands(self):
        """Run the PwBaseWorkChain in bands mode along the path of high-symmetry determined by seekpath."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace="bands")
        )
        inputs.metadata.call_link_label = "bands"
        inputs.kpoints = self.ctx.bands_kpoints
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault("CONTROL", {})
        inputs.pw.parameters.setdefault("SYSTEM", {})
        inputs.pw.parameters.setdefault("ELECTRONS", {})

        # If `nbands_factor` is defined in the inputs we set the `nbnd` parameter
        if "nbands_factor" in self.inputs:
            factor = self.inputs.nbands_factor.value
            parameters = (
                self.ctx.workchain_scf.outputs.output_parameters.get_dict()
            )
            nbands = int(parameters["number_of_bands"])
            nelectron = int(parameters["number_of_electrons"])
            nbnd = max(
                int(0.5 * nelectron * factor),
                int(0.5 * nelectron) + 4,
                nbands,
            )
            inputs.pw.parameters["SYSTEM"]["nbnd"] = nbnd

        # Otherwise set the current number of bands, unless explicitly set in the inputs
        else:
            inputs.pw.parameters["SYSTEM"].setdefault(
                "nbnd", self.ctx.current_number_of_bands
            )

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            f"launching PwBaseWorkChain<{running.pk}> in bands mode"
        )

        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                f"bands PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

    def run_projwfc(self):
        """Run projwfc.x."""
        inputs = AttributeDict(
            self.exposed_inputs(
                ProjwfcBaseWorkChain, namespace="projwfc"
            )
        )
        inputs.metadata.call_link_label = "projwfc"
        inputs.clean_workdir = orm.Bool(False)

        inputs.projwfc.parent_folder = (
            self.ctx.workchain_bands.outputs.remote_folder
        )

        inputs = prepare_process_inputs(ProjwfcBaseWorkChain, inputs)
        parameters = inputs.projwfc.parameters.get_dict()
        parameters["PROJWFC"]["Emin"] = (
            self.inputs.Emin + self.ctx.fermi_energy
        )
        parameters["PROJWFC"]["Emax"] = (
            self.inputs.Emax + self.ctx.fermi_energy
        )
        inputs.projwfc.parameters = orm.Dict(parameters)

        running = self.submit(ProjwfcBaseWorkChain, **inputs)

        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_projwfc=running)

    def inspect_projwfc(
        self,
    ):  # pylint: disable=inconsistent-return-statements
        """Verify that the ProjwfcBaseWorkChain for the projwfc.x run finished successfully."""
        workchain = self.ctx.workchain_projwfc

        if not workchain.is_finished_ok:
            self.report(
                f"ProjwfcBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report("workchain succesfully completed")
        self.out(
            "scf_parameters",
            self.ctx.workchain_scf.outputs.output_parameters,
        )
        self.out(
            "band_parameters",
            self.ctx.workchain_bands.outputs.output_parameters,
        )
        self.out(
            "band_structure",
            self.ctx.workchain_bands.outputs.output_band,
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_projwfc,
                ProjwfcBaseWorkChain,
                namespace="projwfc",
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
