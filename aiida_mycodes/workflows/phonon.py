from typing import Union

from aiida.engine import WorkChain, Process, ProcessSpec
from aiida.engine.processes import if_
from aiida_quantumespresso.workflows.matdyn.base import MatdynBaseWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.q2r.base import Q2rBaseWorkChain
from plumpy import WorkChainSpec


class PhononWorkChain(WorkChain, Process):

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

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

        spec.outline(
            cls.setup,
            if_(cls.should_run_ph)(
                cls.launch_ph,
                cls.inspect_ph,
            ).else_(
                cls.retrieve_ph,
            ),
            cls.launch_q2r,
            cls.inspect_q2r,
            cls.launch_matdyn,
            cls.inspect_matdyn,
            cls.results,
        )