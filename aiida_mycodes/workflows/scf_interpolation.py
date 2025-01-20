"""WorkChain to study evolution of the SCF properties from an origin to a target structure"""

import pathlib
import typing as ty

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
    recursive_merge,
)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from importlib_resources import files
from numpy import linspace

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes import ProcessBuilder, ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from . import protocols
from ..calculations.structure.tools import interpolate_structures

__all__ = ["validate_inputs", "ScfInterpolationWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `ScfInterpolationWorkChain`."""

    if not ("nb_structures" in inputs) ^ ("balance_list" in inputs):
        return (
            "Specify either `nb_structures` or `balance_list` not both"
        )

    if "nb_structures" in inputs and inputs["nb_structures"].value < 2:
        return "Number of interpolated structures must be >= 2."

    if "balance_list" in inputs and len(inputs["balance_list"]) < 2:
        return "Number of interpolated structures must be >= 2."

    ase_origin = inputs["origin_structure"].get_ase()
    ase_target = inputs["target_structure"].get_ase()

    if len(ase_origin) != len(ase_target):
        return "`origin` and `target` structures must have the same number of atoms."

    if not (
        ase_origin.get_atomic_numbers()
        == ase_target.get_atomic_numbers()
    ).all():
        return "`origin` and `target` structures must have the same ordering of elements to interpolate."


class ScfInterpolationWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain to automatically compute SCFs for a set of interpolated structures
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
            help="The number of interpolated structures (>=2). [specify this or `balance_list`]",
            required=False,
        )
        spec.input(
            "balance_list",
            valid_type=orm.List,
            help="The list of balance coefficients to generate interpolated structures (length>=2).\
                  [specify this or `nb_structures`]",
            required=False,
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
            PwBaseWorkChain,
            namespace="base",
            exclude=("pw.structure", "clean_workdir"),
            namespace_options={
                "help": "Inputs for the SCF 'PwBaseWorkChain' sub process.",
            },
        )

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.interpolate_structures,
            cls.run_all_scfs,
            cls.inspect_all_scfs,
            cls.results,
            cls.on_termination,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="One or more SCF sub processes failed",
        )

        spec.output_namespace(
            "output_parameters",
            valid_type=orm.Dict,
            dynamic=True,
            help="The output parameters of each SCF ``PwBaseWorkChain`` performed.",
        )

        spec.output_namespace(
            "interpolated_structures",
            dynamic=True,
            valid_type=orm.StructureData,
            help="The created interpolated structures.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""

        return files(protocols) / "scf_interpolation.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ,unused-argument
        cls,
        pw_code: ty.Union[str, int, orm.Code],
        origin_structure: orm.StructureData,
        target_structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        options: dict = None,
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :param pw_code: code for pw.x.
        :type pw_code: ty.Union[str, int, orm.Code]
        :param origin_structure: the origin ``StructureData``.
        :type origin_structure: orm.StructureData
        :param target_structure: the target ``StructureData``.
        :type target_structure: orm.StructureData
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        # pylint: disable=no-member,too-many-locals,too-many-function-args
        type_check(pw_code, (str, int, orm.Code))
        type_check(origin_structure, orm.StructureData)
        type_check(target_structure, orm.StructureData)
        type_check(protocol, str, allow_none=True)
        type_check(overrides, dict, allow_none=True)

        inputs = cls.get_protocol_inputs(
            protocol=protocol, overrides=overrides
        )
        inputs["base"]["kpoints_distance"] = to_aiida_type(
            inputs["base"]["kpoints_distance"]
        )

        args = (pw_code, origin_structure)
        base_builder = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            protocol=protocol,
            overrides=inputs.get("base", None),
            options=options,
        )
        base_builder["pw"].pop("structure", None)
        base_builder.pop("clean_workdir", None)

        metadata_base = inputs.get("base", {}).get(
            "metadata", {"options": {}}
        )

        if options:
            metadata_base["options"] = recursive_merge(
                metadata_base["options"], options
            )

        builder = cls.get_builder()
        builder.origin_structure = origin_structure
        builder.target_structure = target_structure
        nb_structures = inputs.get("nb_structures", None)
        builder.nb_structures = orm.Int(nb_structures)
        balance_list = inputs.get("balance_list", None)
        builder.balance_list = orm.List(balance_list)
        builder.base = base_builder
        builder.base.pw.metadata = metadata_base

        return builder

    def setup(self):
        """Init required context variables."""
        if "nb_structures" in self.inputs:
            self.ctx.balance_list = linspace(
                0.0, 1.0, self.inputs.nb_structures.value
            )
        else:
            self.ctx.balance_list = self.inputs.balance_list.get_list()
        self.ctx.interpolated_structures_list = []
        self.ctx.labels = []

    def interpolate_structures(self):
        """Call interpolate_structures calculation locally"""
        for index, balance in enumerate(self.ctx.balance_list):
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
        structures_dict = dict(
            zip(self.ctx.labels, self.ctx.interpolated_structures_list)
        )
        self.out("interpolated_structures", structures_dict)
        self.report(f"Structures to process: {pk_list}")

    def run_all_scfs(self):
        """Call all PwBaseWorkChain's for each interpolated structure"""
        futures = {}
        for index, structure in enumerate(
            self.ctx.interpolated_structures_list
        ):
            inputs = AttributeDict(
                self.exposed_inputs(PwBaseWorkChain, namespace="base")
            )
            inputs.pw.structure = structure
            inputs.metadata.call_link_label = f"SCF{index}"
            inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
            future = self.submit(PwBaseWorkChain, **inputs)
            label = self.ctx.labels[index]
            futures[label] = future
            self.report(
                f"launched PwBaseWorkChain for {label}<{future.pk}>"
            )
        return ToContext(**futures)

    # pylint: disable=inconsistent-return-statements
    def inspect_all_scfs(self):
        """Check that all PwBaseWorkChain's finished successfully"""
        work_chains = [self.ctx[label] for label in self.ctx.labels]
        failed_work_chains = []
        for work_chain, label in zip(work_chains, self.ctx.labels):
            if not work_chain.is_finished_ok:
                failed_work_chains.append(work_chain)
                self.report(
                    f"PwBaseWorkChain for ({label}) failed with exit status {work_chain.exit_status}"
                )
        if len(failed_work_chains) > 0:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

    # pylint: enable=inconsistent-return-statements

    def results(self):
        """Compile all output data and send to outputs."""
        output_parameters = {
            label: self.ctx[label].outputs.output_parameters
            for label in self.ctx.labels
        }
        self.out("output_parameters", output_parameters)

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
