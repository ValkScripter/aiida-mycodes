from typing import Union

from aiida import orm
from aiida.engine import WorkChain, Process, ProcessSpec
from aiida.engine.processes import if_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.matdyn.base import MatdynBaseWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.q2r.base import Q2rBaseWorkChain
from plumpy import WorkChainSpec


def validate_inputs(inputs):
    """Validate the inputs of the entire input namespace of `PhononWorkChain`."""
    if not ("remote_ph_folder" in inputs ^ ("ph" in inputs)):
        return "Specify either `remote_ph_folder` or `ph` not both"

class PhononWorkChain(WorkChain, Process):

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
            }
        )

        spec.expose_inputs(
            Q2rBaseWorkChain,
            namespace="q2r",
            namespace_options={
                "help": "Inputs for the `Q2rBaseWorkChain` sub process.",
            }
        )

        spec.expose_inputs(
            MatdynBaseWorkChain,
            namespace="matdyn",
            namespace_options={
                "help": "Inputs for the `MatdynBaseWorkChain` sub process.",
            }
        )

        spec.inputs.validator = validate_inputs

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
            }
        )

        spec.expose_outputs(
            Q2rBaseWorkChain,
            namespace="q2r",
            namespace_options={
                "help": "Outputs of the `Q2rBaseWorkChain` sub process.",
            }
        )

        spec.expose_outputs(
            MatdynBaseWorkChain,
            namespace="matdyn",
            namespace_options={
                "help": "Outputs of the `MatdynBaseWorkChain` sub process.",
            }
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

    def setup(self):
        """
        Set up the context variables.
        """
        self.ctx.ph_workchain_remote_folder = self.inputs.get("remote_ph_folder", None)
        self.ctx.q2r_workchain_force_constants = None

    def should_run_ph(self):
        """
        Return whether the `PhBaseWorkChain` should be run based on the presence of the `ph` input namespace.
        """
        return "ph" in self.inputs

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
        self.ctx.ph_workchain_remote_folder = workchain.outputs.remote_folder
        self.out_many(self.exposed_outputs(workchain, PhBaseWorkChain, namespace="ph"))
        self.report("Phonon calculation completed")

    def launch_q2r(self):
        """
        Launch the `Q2rBaseWorkChain`.
        """
        inputs = self.exposed_inputs(Q2rBaseWorkChain, "q2r")
        inputs["q2r"]["parent_folder"] = self.ctx.ph_workchain_remote_folder
        q2r_inputs = prepare_process_inputs(Q2rBaseWorkChain, inputs)
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
        self.ctx.q2r_workchain_force_constants = workchain.outputs.force_constants
        self.out_many(self.exposed_outputs(workchain, Q2rBaseWorkChain, namespace="q2r"))
        self.report("Q2rBaseWorkChain completed")

    def launch_matdyn(self):
        """
        Launch the `MatdynBaseWorkChain`.
        """
        inputs = self.exposed_inputs(MatdynBaseWorkChain, "matdyn")
        inputs["matdyn"]["force_constants"] = self.ctx.q2r_workchain_force_constants
        matdyn_inputs = prepare_process_inputs(MatdynBaseWorkChain, inputs)
        future = self.submit(MatdynBaseWorkChain, **matdyn_inputs)
        self.report(f"launching MatdynBaseWorkChain<{future.pk}>")
        self.to_context(matdyn_workchain=future)

    def inspect_matdyn(self):
        """
        Check that the `MatdynBaseWorkChain` finished successfully.
        """
        workchain = self.ctx.matdyn_workchain
        if not workchain.is_finished_ok:
            self.report(
                f"MatdynBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN
        self.out_many(self.exposed_outputs(workchain, MatdynBaseWorkChain, namespace="matdyn"))
        self.report("MatdynBaseWorkChain completed")
