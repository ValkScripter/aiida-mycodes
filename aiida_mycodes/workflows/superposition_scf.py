"""
This file adds to SuperpositionSCFWorkChains (1 for QE and 1 for Fireball)
They all perform a stacking of 2 structures vertically and then an SCF calculation on the stacked structures.
"""

from typing import Union

from aiida_fireball.calculations.fireball import FireballCalculation
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida import orm
from aiida.engine import (
    Process,
    ProcessSpec,
    ToContext,
    WorkChain,
    run_get_node,
)
from aiida.engine.processes.workchains.workchain import WorkChainSpec

from ..calculations.structure.tools import superposition


class SuperpositionSCFWorkChain(WorkChain, Process):
    """This workchain performs a superposition of 2 structures firest and then
    launches an SCF calculation using QE on the stacked structures.
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.expose_inputs(
            superposition,
            namespace="superposition",
            namespace_options={
                "help": "Inputs for the `superposition` calculation.",
                "required": True,
            },
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for the SCF calculation.",
                "required": True,
            },
        )

        spec.outline(
            cls.setup,
            cls.run_superposition,
            cls.inspect_superposition,
            cls.run_scf,
            cls.inspect_scf,
        )

        spec.expose_outputs(
            superposition,
            namespace="superposition",
            namespace_options={
                "help": "The outputs of the `superposition` calculation.",
                "required": True,
            },
        )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace="scf",
            namespace_options={
                "help": "The outputs of the `PwBaseWorkChain` for the SCF calculation.",
                "required": True,
            },
        )

        spec.exit_code(
            401,
            "ERROR_SUPERPOSITION_FAILED",
            message="The superposition calculation failed.",
        )

        spec.exit_code(
            402,
            "ERROR_SCF_FAILED",
            message="The SCF calculation failed.",
        )

    def setup(self):
        """Setup the workchain ctx and inputs."""
        self.ctx.superposition_inputs = self.exposed_inputs(
            superposition, "superposition"
        )
        self.ctx.scf_inputs = self.exposed_inputs(
            PwBaseWorkChain, "scf"
        )

    # pylint: disable=line-too-long
    def run_superposition(self):
        """Run the superposition calculation."""
        inputs = self.ctx.superposition_inputs

        inputs = prepare_process_inputs(superposition, inputs)
        _, node = run_get_node(superposition, **inputs)
        self.report(
            f"Launched superposition calculation with distance = {inputs['distance'].value if isinstance(inputs['distance'], orm.Float) else inputs['distance'].get_list()[0]}"
        )
        self.ctx.superposition_calculation = node

    def inspect_superposition(self):
        """Inspect the result of the superposition calculation."""
        calculation = self.ctx.superposition_calculation
        if not calculation.is_finished_ok:
            return self.exit_codes.ERROR_SUPERPOSITION_FAILED
        self.out_many(
            self.exposed_outputs(
                calculation, superposition, namespace="superposition"
            )
        )

    def run_scf(self):
        """Run the SCF calculation."""
        inputs = self.ctx.scf_inputs
        inputs["clean_workdir"] = orm.Bool(True)
        inputs["pw"][
            "structure"
        ] = self.ctx.superposition_calculation.outputs.stack_structure
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.superposition_calculation.outputs.stack_structure.label}_scf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        future = self.submit(PwBaseWorkChain, **inputs)
        self.report("Launched PwBaseWorkChain for SCF calculation.")
        return ToContext(scf_workchain=future)

    def inspect_scf(self):
        """Inspect the result of the SCF calculation."""
        workchain = self.ctx.scf_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_SCF_FAILED
        self.out_many(
            self.exposed_outputs(
                workchain, PwBaseWorkChain, namespace="scf"
            )
        )


class FireballSuperpositionSCFWorkChain(WorkChain, Process):
    """This workchain performs a superposition of 2 structures firest and then
    launches an SCF calculation using Fireball on the stacked structures.
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.expose_inputs(
            superposition,
            namespace="superposition",
            namespace_options={
                "help": "Inputs for the `superposition` calculation.",
                "required": True,
            },
        )

        spec.expose_inputs(
            FireballCalculation,
            namespace="scf",
            exclude=("structure",),
            namespace_options={
                "help": "Inputs for the `Fireball` calculation for the SCF calculation.",
                "required": True,
            },
        )

        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="If True, work directories of all calculations are deleted at the end of execution.",
        )

        spec.outline(
            cls.setup,
            cls.run_superposition,
            cls.inspect_superposition,
            cls.run_scf,
            cls.inspect_scf,
        )

        spec.expose_outputs(
            superposition,
            namespace="superposition",
            namespace_options={
                "help": "The outputs of the `superposition` calculation.",
                "required": True,
            },
        )
        spec.expose_outputs(
            FireballCalculation,
            namespace="scf",
            namespace_options={
                "help": "The outputs of the `Fireball` calculation for the SCF calculation.",
                "required": True,
            },
        )

        spec.exit_code(
            401,
            "ERROR_SUPERPOSITION_FAILED",
            message="The superposition calculation failed.",
        )

        spec.exit_code(
            402,
            "ERROR_SCF_FAILED",
            message="The SCF calculation failed.",
        )

    def setup(self):
        """Setup the workchain ctx and inputs."""
        self.ctx.superposition_inputs = self.exposed_inputs(
            superposition, "superposition"
        )
        self.ctx.scf_inputs = self.exposed_inputs(
            FireballCalculation, "scf"
        )

    # pylint: disable=line-too-long
    def run_superposition(self):
        """Run the superposition calculation."""
        inputs = self.ctx.superposition_inputs

        inputs = prepare_process_inputs(superposition, inputs)
        _, node = run_get_node(superposition, **inputs)
        self.report(
            f"Launched superposition calculation with distance = {inputs['distance'].value if isinstance(inputs['distance'], orm.Float) else inputs['distance'].get_list()[0]}"
        )
        self.ctx.superposition_calculation = node

    def inspect_superposition(self):
        """Inspect the result of the superposition calculation."""
        calculation = self.ctx.superposition_calculation
        if not calculation.is_finished_ok:
            return self.exit_codes.ERROR_SUPERPOSITION_FAILED
        self.out_many(
            self.exposed_outputs(
                calculation, superposition, namespace="superposition"
            )
        )

    def run_scf(self):
        """Run the SCF calculation."""
        inputs = self.ctx.scf_inputs
        inputs["structure"] = (
            self.ctx.superposition_calculation.outputs.stack_structure
        )
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.superposition_calculation.outputs.stack_structure.label}_scf"

        inputs = prepare_process_inputs(FireballCalculation, inputs)
        future = self.submit(FireballCalculation, **inputs)
        self.report("Launched FireballCalculation for SCF calculation.")
        return ToContext(scf_workchain=future)

    def inspect_scf(self):
        """Inspect the result of the SCF calculation."""
        workchain = self.ctx.scf_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_SCF_FAILED
        self.out_many(
            self.exposed_outputs(
                workchain, FireballCalculation, namespace="scf"
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
