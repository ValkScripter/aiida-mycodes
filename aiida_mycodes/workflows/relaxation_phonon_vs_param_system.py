"""
This WorkChains performs RelaxationPhononWorkChains on a structure with a different SYSTEM parameter each time.
This can be used to study the influence of the smearing on the phonon dispersion for instance.
"""

from typing import Union

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import Process, WorkChain, append_
from aiida.engine.processes.process_spec import ProcessSpec
from aiida.engine.processes.workchains.workchain import WorkChainSpec

from .relaxation_phonon import RelaxationPhononWorkChain


class RelaxationPhononVSParamSystemWorkChain(WorkChain, Process):
    """
    This WorkChains performs RelaxationPhononWorkChains on a structure with a different SYSTEM parameter each time.
    This can be used to study the influence of the smearing on the phonon dispersion for instance.
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.input(
            "param_system_key",
            valid_type=orm.Str,
            required=True,
            help="The key in the 'SYSTEM' parameter card to study against (e.g. 'degauss' or 'ecutrho').",
        )

        spec.input(
            "param_system_values",
            valid_type=orm.List,
            required=True,
            help="The values of the 'SYSTEM' parameter card for the given key to study against.",
        )

        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="Whether to clean the work directory after the workchain finishes.",
        )

        spec.expose_inputs(
            RelaxationPhononWorkChain,
            namespace="relaxation_phonon",
            namespace_options={
                "help": "Inputs for the `RelaxationPhononWorkChain` used to relax and calculate the phonon properties.",
                "required": True,
            },
        )

        spec.outline(
            cls.setup,
            cls.launch_relaxation_phonon_workchains_parallel,
            cls.inspect_relaxation_phonon_workchains,
            cls.results,
        )

        spec.expose_outputs(
            RelaxationPhononWorkChain,
            namespace_options={
                "help": "Outputs from the `RelaxationPhononWorkChain` for each strained structure.",
                "dynamic": True,
            },
        )
        spec.exit_code(
            401,
            "ERROR_ONE_SUBWORKFLOW_FAILED",
            message="One of the subworkflows failed.",
        )

    def setup(self):
        """Setup the workchain ctx and inputs."""
        self.ctx.structure = self.inputs.relaxation_phonon.structure
        self.ctx.param_system_key = self.inputs.param_system_key.value
        self.ctx.param_system_values = (
            self.inputs.param_system_values.get_list()
        )
        self.ctx.n = len(self.ctx.param_system_values)

    def launch_relaxation_phonon_workchains_parallel(self):
        """Launches the RelaxationPhononWorkChains in parallel."""
        for value in self.ctx.param_system_values:
            inputs = AttributeDict(
                self.exposed_inputs(
                    RelaxationPhononWorkChain, "relaxation_phonon"
                )
            )
            # Relax
            parameters = inputs.relax.base.pw.parameters.get_dict()
            parameters["SYSTEM"][self.ctx.param_system_key] = value
            inputs.relax.base.pw.parameters = orm.Dict(parameters)
            # Fine relax
            parameters = inputs.finerelax.base.pw.parameters.get_dict()
            parameters["SYSTEM"][self.ctx.param_system_key] = value
            inputs.finerelax.base.pw.parameters = orm.Dict(parameters)
            # SCF
            parameters = inputs.scf.pw.parameters.get_dict()
            parameters["SYSTEM"][self.ctx.param_system_key] = value
            inputs.scf.pw.parameters = orm.Dict(parameters)

            inputs.metadata.label = f"{self.ctx.structure.label}_relaxation_phonon_for_\
{self.ctx.param_system_key}_{value:.3f}"
            inputs = prepare_process_inputs(
                RelaxationPhononWorkChain, inputs
            )
            self.report(
                f"Launching relaxation-phonon workchain for {self.ctx.param_system_key}={value:.2e}"
            )
            future = self.submit(RelaxationPhononWorkChain, **inputs)
            self.to_context(workflows=append_(future))

        self.report(
            f"Submitted {len(self.ctx.workflows)} relaxation-phonon workchains."
        )

    def inspect_relaxation_phonon_workchains(self):
        """Inspect the results."""
        failed_pks = []
        failed_values = []
        for i, workflow in enumerate(self.ctx.workflows):
            if not workflow.is_finished_ok:
                failed_pks.append(workflow.pk)
                failed_values.append(self.ctx.param_system_values[i])
        if len(failed_pks) > 0:
            self.report(
                f"Relaxation-phonon workchains failed for values \
{', '.join(map(lambda c: f'{c[0]:.3f}<{c[1]}>', zip(failed_values, failed_pks)))}."
            )
            return self.exit_codes.ERROR_ONE_SUBWORKFLOW_FAILED

    def results(self):
        """Returns the results as outputs."""
        for i, workflow in enumerate(self.ctx.workflows):
            outputs = self.exposed_outputs(
                workflow, RelaxationPhononWorkChain
            )
            outputs = {f"value_{i}": outputs}
            self.out_many(AttributeDict(outputs))

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


def clean_calcjob_remote(node: orm.CalcJobNode) -> bool:
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        cleaned = True
    except (OSError, KeyError):
        pass
    return cleaned


def clean_workchain_calcs(workchain: orm.WorkChainNode) -> list[int]:
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            if clean_calcjob_remote(called_descendant):
                cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs
