"""
This workchain perfoms a RelaxationPhononWorkChain for every strained structure scaled between a strain_min and
a strain_max. Scaling directions can be chosen (a,b,c).
"""

from typing import Union

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import Process, WorkChain, append_, calcfunction
from aiida.engine.processes.process_spec import ProcessSpec
from aiida.engine.processes.workchains.workchain import WorkChainSpec

from aiida_mycodes.calculations.structure.tools import scale_structure

from .relaxation_phonon import RelaxationPhononWorkChain


@calcfunction
def get_strains(
    strain_min: orm.Float,
    strain_max: orm.Float,
    num_points: orm.Int,
) -> orm.List:
    """
    Calculate a list of strains based on the minimum strain, maximum strain, and the number of points.

    This function generates a list of evenly spaced strain values between the specified minimum and
    maximum strain values (inclusive). The number of points is used to determine the interval
    between consecutive strain values.

    :param strain_min: The minimum strain value for the range.
    :param strain_max: The maximum strain value for the range.
    :param num_points: The number of points to include in the list of strains.
    :return: A list of evenly spaced strain values.
    """
    return orm.List(
        list(
            np.linspace(
                strain_min.value, strain_max.value, num_points.value
            )
        )
    )


@calcfunction
def embed_in_list(*args):
    """
    Embed the given arguments into an instance of `orm.List`.

    This function takes a variable number of arguments, encapsulates them as
    a list, and then wraps this list into an `orm.List`. The function is
    designed for integration with the `@calcfunction` decorator and provides
    a utility to generate `orm.List` objects in an efficient and clean manner.

    :param args: Variable length argument list containing the input values to
        be included in the `orm.List`.
    :type args: Any
    :return: An instance of `orm.List` containing all provided arguments as
        items in the list.
    :rtype: orm.List
    """
    return orm.List(
        list(args),
    )


class PhononStrainWorkChain(WorkChain, Process):
    """
    This workchain perfoms a RelaxationPhononWorkChain for every strained structure scaled between a strain_min and
    a strain_max. Scaling directions can be chosen (a,b,c).
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="The structure to strain, relax, and calculate the phonon properties.",
        )
        spec.input(
            "strain_min",
            valid_type=orm.Float,
            required=True,
            help="The minimum of the strain range to apply.",
        )
        spec.input(
            "strain_max",
            valid_type=orm.Float,
            required=True,
            help="The maximum of the strain range to apply.",
        )
        spec.input(
            "num_points",
            valid_type=orm.Int,
            required=True,
            help="The number of points to use in the strain range (including the endpoints).",
        )
        spec.input(
            "scale_a",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="Whether to scale the structure along the a-axis.",
        )
        spec.input(
            "scale_b",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="Whether to scale the structure along the b-axis.",
        )
        spec.input(
            "scale_c",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="Whether to scale the structure along the c-axis.",
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
            exclude=["structure"],
            namespace_options={
                "help": "Inputs for the `RelaxationPhononWorkChain` used to relax and calculate the phonon properties.",
                "required": True,
            },
        )

        spec.outline(
            cls.setup,
            cls.calculate_strains,
            cls.create_strained_structures,
            cls.launch_relaxation_phonon_workchains_parallel,
            cls.inspect_relaxation_phonon_workchains,
            cls.results,
        )

        spec.output(
            "strains",
            valid_type=orm.List,
            required=True,
            help="The list of strains.",
        )
        spec.output_namespace(
            "strained_structures",
            dynamic=True,
            help="The respectively strained structures.",
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
        self.ctx.structure = self.inputs.structure
        self.ctx.n = self.inputs.num_points.value
        self.ctx.strains = []
        self.ctx.strained_structures = []

    def calculate_strains(self):
        """Calculates the strain values for scaling the structures."""
        self.report(
            f"Strain range: {self.inputs.strain_min.value} to {self.inputs.strain_max.value},\
{self.inputs.num_points.value} points."
        )
        strains = get_strains(
            self.inputs.strain_min,
            self.inputs.strain_max,
            self.inputs.num_points,
        )
        self.ctx.strains = strains.get_list()
        self.ctx.strains_orm = strains

    def create_strained_structures(self):
        """Creates the strained structures with the previous strain values."""
        self.report(
            f'Creating strained structures along {"a" if self.inputs.scale_a.value else ""}\
{"b" if self.inputs.scale_b.value else ""}{"c" if self.inputs.scale_c.value else ""} axes.'
        )
        for strain in self.ctx.strains:
            temp_structure = scale_structure(
                structure=self.ctx.structure,
                scale_factor=strain,
                scale_a=self.inputs.scale_a.value,
                scale_b=self.inputs.scale_b.value,
                scale_c=self.inputs.scale_c.value,
            )
            temp_structure.label = (
                f"{self.ctx.structure.label}_strain_{strain:.3f}"
            )
            self.ctx.strained_structures.append(temp_structure)

    def launch_relaxation_phonon_workchains_parallel(self):
        """Launches in parallel all RelaxationPhononWorkChains"""
        for strain, strained_structure in zip(
            self.ctx.strains, self.ctx.strained_structures
        ):
            inputs = AttributeDict(
                self.exposed_inputs(
                    RelaxationPhononWorkChain, "relaxation_phonon"
                )
            )
            inputs["structure"] = strained_structure
            inputs.metadata.label = (
                f"{strained_structure.label}_relaxation_phonon"
            )
            inputs = prepare_process_inputs(
                RelaxationPhononWorkChain, inputs
            )
            self.report(
                f"Launching relaxation-phonon workchain for strain {strain}."
            )
            future = self.submit(RelaxationPhononWorkChain, **inputs)
            self.to_context(workflows=append_(future))

        self.report(
            f"Submitted {len(self.ctx.workflows)} relaxation-phonon workchains."
        )

    def inspect_relaxation_phonon_workchains(self):
        """Inspects all the workchains."""
        failed_pks = []
        failed_strains = []
        for i, workflow in enumerate(self.ctx.workflows):
            if not workflow.is_finished_ok:
                failed_pks.append(workflow.pk)
                failed_strains.append(self.ctx.strains[i])
        if len(failed_pks) > 0:
            self.report(
                f"Relaxation-phonon workchains for strains\
            {', '.join(map(lambda c: f'{c[0]:.3f}<{c[1]}>', zip(failed_strains, failed_pks)))} failed."
            )
            return self.exit_codes.ERROR_ONE_SUBWORKFLOW_FAILED

    def results(self):
        """Returns the results as outputs"""
        self.out("strains", self.ctx.strains_orm)

        self.out_many(
            {
                f"strained_structures.strained_structure_{i}": self.ctx.strained_structures[
                    i
                ]
                for i in range(self.ctx.n)
            }
        )

        for i, workflow in enumerate(self.ctx.workflows):
            outputs = self.exposed_outputs(
                workflow, RelaxationPhononWorkChain
            )
            outputs = {f"strain_{i}": outputs}
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
