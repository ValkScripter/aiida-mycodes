"""Workchain to run a high-level phonon workflow."""

# pylint: disable=no-member, inconsistent-return-statements

from typing import Union

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.matdyn.base import (
    MatdynBaseWorkChain,
)
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.q2r.base import Q2rBaseWorkChain
from plumpy import WorkChainSpec

from aiida import orm
from aiida.engine import Process, ProcessSpec, WorkChain
from aiida.engine.processes import if_

__all__ = ["PhononWorkChain"]


class PhononWorkChain(WorkChain, Process):
    """
    Class for running a high-level phonon workflow.

    This class orchestrates sub-processes to perform phonon calculations, specifically launching
    and handling outputs of `PhBaseWorkChain`, `Q2rBaseWorkChain`, and `MatdynBaseWorkChain`.
    The `PhononWorkChain` determines the execution sequence of sub-processes, manages dependencies
    between them, and defines a structured workflow for phonon simulations.
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.input(
            "remote_ph_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="Remote folder of a previous `PhBaseWorkChain`.",
        )

        spec.expose_inputs(
            PhBaseWorkChain,
            namespace="ph",
            namespace_options={
                "help": "Inputs for the `PhBaseWorkChain` sub process.",
                "required": False,
            },
        )

        spec.expose_inputs(
            Q2rBaseWorkChain,
            namespace="q2r",
            exclude=[
                "q2r.parent_folder",
            ],
            namespace_options={
                "help": "Inputs for the `Q2rBaseWorkChain` sub process.",
            },
        )

        spec.expose_inputs(
            MatdynBaseWorkChain,
            namespace="matdyn",
            exclude=[
                "matdyn.force_constants",
                "matdyn.parent_folder",
            ],
            namespace_options={
                "help": "Inputs for the `MatdynBaseWorkChain` sub process.",
            },
        )

        spec.expose_inputs(
            MatdynBaseWorkChain,
            namespace="matdyn_2",
            exclude=[
                "matdyn.force_constants",
                "matdyn.parent_folder",
            ],
            namespace_options={
                "help": "Inputs for the second `MatdynBaseWorkChain` sub process.",
                "required": False,
            },
        )

        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, work directories of all called calculations will be cleaned at the end of execution.",
        )

        spec.inputs.validator = cls.validate_inputs

        # noinspection PyTypeChecker
        spec.outline(
            cls.setup,
            if_(cls.should_run_ph)(
                cls.launch_ph,
                cls.inspect_ph,
            ),
            cls.launch_q2r,
            cls.inspect_q2r,
            cls.launch_matdyn,
            cls.inspect_matdyn,
        )

        spec.expose_outputs(
            PhBaseWorkChain,
            namespace="ph",
            namespace_options={
                "help": "Outputs of the `PhBaseWorkChain` sub process.",
                "required": False,
            },
        )

        spec.expose_outputs(
            Q2rBaseWorkChain,
            namespace="q2r",
            namespace_options={
                "help": "Outputs of the `Q2rBaseWorkChain` sub process.",
            },
        )

        spec.expose_outputs(
            MatdynBaseWorkChain,
            namespace="matdyn",
            namespace_options={
                "help": "Outputs of the `MatdynBaseWorkChain` sub process.",
            },
        )

        spec.expose_outputs(
            MatdynBaseWorkChain,
            namespace="matdyn_2",
            namespace_options={
                "help": "Outputs of the second `MatdynBaseWorkChain` sub process.",
                "required": False,
            },
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_PH",
            message="The `PhBaseWorkChain` sub processes failed.",
        )

        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_Q2R",
            message="The `Q2rBaseWorkChain` sub processes failed.",
        )

        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_MATDYN",
            message="The `MatdynBaseWorkChain` sub processes failed.",
        )

        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_MATDYN_2",
            message="The second `MatdynBaseWorkChain` sub processes failed.",
        )

        spec.exit_code(
            405,
            "ERROR_NO_REMOTE_PH_FOLDER_OR_PH_INPUT_SPECIFIED",
            message="Neither `remote_ph_folder` nor `ph` input was specified.",
        )

    @classmethod
    def validate_inputs(cls, value, port_namespace):
        """Validate the inputs of the top level namespace."""
        if not (
            "remote_ph_folder" in port_namespace
            or "ph" in port_namespace
        ):
            return "Specify either `remote_ph_folder` or `ph` input"
        if ("remote_ph_folder" in value and "ph" in value) or (
            "remote_ph_folder" not in value and "ph" not in value
        ):
            return "Specify either `remote_ph_folder` or `ph`, not both"
        if not ("remote_ph_folder" in value) and ("ph" in value):
            if value["ph"] is None:
                return "Specify `ph` input if `remote_ph_folder` is not specified"

    def setup(self):
        """
        Set up the context variables.
        """
        self.ctx.ph_workchain_remote_folder = self.inputs.get(
            "remote_ph_folder", None
        )
        if (
            self.ctx.ph_workchain_remote_folder is None
            and "ph" not in self.inputs
        ):
            return (
                self.exit_codes.ERROR_NO_REMOTE_PH_FOLDER_OR_PH_INPUT_SPECIFIED
            )
        self.ctx.q2r_workchain_force_constants = None

    def should_run_ph(self):
        """
        Return whether the `PhBaseWorkChain` should be run based on the presence of the `ph` input namespace.
        """
        return "ph" in self.inputs and self.inputs["ph"] is not None

    def should_run_matdyn_2(self):
        """
        Return whether the second `MatdynBaseWorkChain` should be run
        based on the presence of the `matdyn_2` input namespace.
        """
        return (
            "matdyn_2" in self.inputs
            and self.inputs["matdyn_2"] is not None
        )

    def launch_ph(self):
        """
        Launch the `PhBaseWorkChain`.
        """
        inputs = self.exposed_inputs(PhBaseWorkChain, "ph")
        ph_inputs = prepare_process_inputs(PhBaseWorkChain, inputs)
        future = self.submit(PhBaseWorkChain, **ph_inputs)
        self.report(f"launching PhBaseWorkChain<{future.pk}>")
        self.to_context(ph_workchain=future)

    def inspect_ph(self):
        """
        Check that the `PhBaseWorkChain` finished successfully.
        """
        workchain = self.ctx.ph_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"PhBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PH
        self.ctx.ph_workchain_remote_folder = (
            workchain.outputs.remote_folder
        )
        self.out_many(
            self.exposed_outputs(
                workchain, PhBaseWorkChain, namespace="ph"
            )
        )
        self.report("Phonon calculation completed")

    def launch_q2r(self):
        """
        Launch the `Q2rBaseWorkChain`.
        """
        inputs = self.exposed_inputs(Q2rBaseWorkChain, "q2r")
        # inputs["q2r"]["parent_folder"] = self.ctx.ph_workchain_remote_folder
        q2r_inputs = prepare_process_inputs(Q2rBaseWorkChain, inputs)
        q2r_inputs["q2r"][
            "parent_folder"
        ] = self.ctx.ph_workchain_remote_folder
        future = self.submit(Q2rBaseWorkChain, **q2r_inputs)
        self.report(f"launching Q2rBaseWorkChain<{future.pk}>")
        self.to_context(q2r_workchain=future)

    def inspect_q2r(self):
        """
        Check that the `Q2rBaseWorkChain` finished successfully.
        """
        workchain = self.ctx.q2r_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"Q2rBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Q2R
        self.ctx.q2r_workchain_force_constants = (
            workchain.outputs.force_constants
        )
        self.out_many(
            self.exposed_outputs(
                workchain, Q2rBaseWorkChain, namespace="q2r"
            )
        )
        self.report("Q2rBaseWorkChain completed")

    def launch_matdyn(self):
        """
        Launch the `MatdynBaseWorkChain` and eventually the second one.
        """
        inputs = self.exposed_inputs(MatdynBaseWorkChain, "matdyn")
        # inputs["matdyn"]["force_constants"] = self.ctx.q2r_workchain_force_constants
        matdyn_inputs = prepare_process_inputs(
            MatdynBaseWorkChain, inputs
        )
        matdyn_inputs["matdyn"][
            "force_constants"
        ] = self.ctx.q2r_workchain_force_constants
        future = self.submit(MatdynBaseWorkChain, **matdyn_inputs)
        self.report(f"launching MatdynBaseWorkChain<{future.pk}>")
        self.to_context(matdyn_workchain=future)

        if self.should_run_matdyn_2():
            self.launch_matdyn_2()

    def inspect_matdyn(self):
        """
        Check that the `MatdynBaseWorkChain` finished successfully and eventually the second one.
        """
        workchain = self.ctx.matdyn_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"MatdynBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN
        self.out_many(
            self.exposed_outputs(
                workchain, MatdynBaseWorkChain, namespace="matdyn"
            )
        )
        self.report("MatdynBaseWorkChain completed")

        if self.should_run_matdyn_2():
            self.inspect_matdyn_2()

    def launch_matdyn_2(self):
        """
        Launch the second `MatdynBaseWorkChain`.
        """
        inputs = self.exposed_inputs(MatdynBaseWorkChain, "matdyn_2")
        # inputs["matdyn"]["force_constants"] = self.ctx.q2r_workchain_force_constants
        matdyn_inputs = prepare_process_inputs(
            MatdynBaseWorkChain, inputs
        )
        matdyn_inputs["matdyn"][
            "force_constants"
        ] = self.ctx.q2r_workchain_force_constants
        future = self.submit(MatdynBaseWorkChain, **matdyn_inputs)
        self.report(
            f"launching second MatdynBaseWorkChain<{future.pk}>"
        )
        self.to_context(matdyn_2_workchain=future)

    def inspect_matdyn_2(self):
        """
        Check that the second `MatdynBaseWorkChain` finished successfully.
        """
        workchain = self.ctx.matdyn_2_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"second MatdynBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN_2
        self.out_many(
            self.exposed_outputs(
                workchain, MatdynBaseWorkChain, namespace="matdyn_2"
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
