"""
WorkChain to perform a sequence of calculations:
    1. a standard relaxation of the structure
    2. a finer relaxation of the structure for phonon calculations
    3. a fine SCF calculation of the finer relaxed structure
    4. a D3hess calculation after the SCF to compute the DFT-D3 dispersion correction hessian matrix
    5. a Phonon calculation workflow with ph.x, q2r.x and 2*matdyn.x on q-path and q-mesh
"""

# pylint: disable=line-too-long

from typing import Union

from aiida_quantumespresso.calculations.d3hess import D3hessCalculation
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida import orm
from aiida.engine import Process, ProcessSpec, ToContext, WorkChain, if_
from aiida.engine.processes.workchains.workchain import WorkChainSpec

from .phonon import PhononWorkChain


class RelaxationPhononWorkChain(WorkChain, Process):
    """
    WorkChain to perform a sequence of calculations:
    1. a standard relaxation of the structure
    2. a finer relaxation of the structure for phonon calculations
    3. a fine SCF calculation of the finer relaxed structure
    4. a D3hess calculation after the SCF to compute the DFT-D3 dispersion correction hessian matrix
    5. a Phonon calculation workflow with ph.x, q2r.x and 2*matdyn.x on q-path and q-mesh
    """

    @classmethod
    def define(cls, spec: Union[WorkChainSpec, ProcessSpec]):
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="The structure to relax and calculate the phonon properties.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="If True, work directories of all calculations are deleted at the end of execution.",
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="relax",
            exclude=("clean_workdir", "structure", "base_final_scf"),
            namespace_options={
                "help": "Inputs for the `PwRelaxWorkChain` for the standard relaxation of the structure.",
                "required": False,
            },
        )
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="finerelax",
            exclude=("clean_workdir", "structure", "base_final_scf"),
            namespace_options={
                "help": "Inputs for the `PwRelaxWorkChain` for the finer relaxation of the structure for phonon calculations.",
                "required": True,
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for the fine SCF calculation of the finer relaxed structure.",
                "required": True,
            },
        )
        spec.expose_inputs(
            D3hessCalculation,
            namespace="d3hess",
            exclude=("parent_folder"),
            namespace_options={
                "help": "Inputs for the `D3hessCalculation` after the SCF to compute the DFT-D3 dispersion correction hessian matrix.",
                "required": False,
            },
        )
        spec.expose_inputs(
            PhononWorkChain,
            namespace="phonon",
            exclude=(
                "clean_workdir",
                "remote_ph_folder",
                "ph.ph.parent_folder",
            ),
            namespace_options={
                "help": "Inputs for the `PhononWorkChain` for the phonon calculation workflow.",
                "required": True,
            },
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_relaxation_standard)(
                cls.run_relaxation_standard,
                cls.inspect_relaxation_standard,
            ),
            cls.run_relaxation_finerelax,
            cls.inspect_relaxation_finerelax,
            cls.run_scf,
            cls.inspect_scf,
            if_(cls.should_run_d3hess)(
                cls.run_d3hess,
                cls.inspect_d3hess,
            ),
            cls.run_phonon,
            cls.inspect_phonon,
        )

        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace="relax",
            namespace_options={
                "help": "The outputs of the `PwRelaxWorkChain` for the standard relaxation of the structure.",
                "required": False,
            },
        )
        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace="finerelax",
            namespace_options={
                "help": "The outputs of the `PwRelaxWorkChain` for the finer relaxation of the structure for phonon calculations.",
                "required": True,
            },
        )
        spec.expose_outputs(
            PwBaseWorkChain,
            namespace="scf",
            namespace_options={
                "help": "The outputs of the `PwBaseWorkChain` for the fine SCF calculation of the finer relaxed structure.",
                "required": True,
            },
        )
        spec.expose_outputs(
            D3hessCalculation,
            namespace="d3hess",
            namespace_options={
                "help": "The outputs of the `D3hessCalculation` after the SCF to compute the DFT-D3 dispersion correction hessian matrix.",
                "required": False,
            },
        )
        spec.expose_outputs(
            PhononWorkChain,
            namespace="phonon",
            namespace_options={
                "help": "The outputs of the `PhononWorkChain` for the phonon calculation workflow.",
                "required": True,
            },
        )

        spec.exit_code(
            401,
            "ERROR_RELAXATION_STANDARD_FAILED",
            message="The standard relaxation of the structure failed.",
        )
        spec.exit_code(
            402,
            "ERROR_RELAXATION_FINERELAX_FAILED",
            message="The finer relaxation of the structure for phonon calculations failed.",
        )
        spec.exit_code(
            403,
            "ERROR_SCF_FAILED",
            message="The fine SCF calculation of the finer relaxed structure failed.",
        )
        spec.exit_code(
            404,
            "ERROR_D3HESS_FAILED",
            message="The D3hess calculation after the SCF to compute the DFT-D3 dispersion correction hessian matrix failed.",
        )
        spec.exit_code(
            405,
            "ERROR_PHONON_FAILED",
            message="The phonon calculation workflow failed.",
        )

    def setup(self):
        """Setup the workchain ctx and inputs."""
        self.ctx.current_structure = (
            self.inputs.structure
        )  # current structure to work with before each step
        self.ctx.clean_workdir = self.inputs.clean_workdir
        for process, key in zip(
            [
                PwRelaxWorkChain,
                PwRelaxWorkChain,
                PwBaseWorkChain,
                D3hessCalculation,
                PhononWorkChain,
            ],
            ["relax", "finerelax", "scf", "d3hess", "phonon"],
        ):
            self.ctx[f"{key}_inputs"] = self.exposed_inputs(
                process, key
            )
        # The respective inputs have to be properly filled before launching with e.g. excluded input ports

    def should_run_relaxation_standard(self):
        """Check if the standard relaxation should be run."""
        return "relax" in self.inputs and self.inputs.relax is not None

    def run_relaxation_standard(self):
        """Run the standard relaxation."""
        inputs = self.ctx.relax_inputs
        inputs["clean_workdir"] = orm.Bool(True)
        inputs["structure"] = self.ctx.current_structure
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.current_structure.label}_relax"

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        future = self.submit(PwRelaxWorkChain, **inputs)
        self.report(
            "Launched PwRelaxWorkChain for standard relaxation."
        )
        return ToContext(relax_workchain=future)

    def inspect_relaxation_standard(self):
        """Inspect the result of the standard relaxation."""
        workchain = self.ctx.relax_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_RELAXATION_STANDARD_FAILED
        self.ctx.current_structure = workchain.outputs.output_structure
        self.ctx.current_structure.label = (
            f"{self.inputs.structure.label}_relaxed"
        )
        self.out_many(
            self.exposed_outputs(
                workchain, PwRelaxWorkChain, namespace="relax"
            )
        )

    def run_relaxation_finerelax(self):
        """Run the finer relaxation."""
        inputs = self.ctx.finerelax_inputs
        inputs["clean_workdir"] = orm.Bool(True)
        inputs["structure"] = self.ctx.current_structure
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.current_structure.label}_finerelax"

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        future = self.submit(PwRelaxWorkChain, **inputs)
        self.report("Launched PwRelaxWorkChain for finer relaxation.")
        return ToContext(finerelax_workchain=future)

    def inspect_relaxation_finerelax(self):
        """Inspect the result of the finer relaxation."""
        workchain = self.ctx.finerelax_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_RELAXATION_FINERELAX_FAILED
        self.ctx.current_structure = workchain.outputs.output_structure
        self.ctx.current_structure.label = (
            f"{self.inputs.structure.label}_finerelaxed"
        )
        self.out_many(
            self.exposed_outputs(
                workchain, PwRelaxWorkChain, namespace="finerelax"
            )
        )

    def run_scf(self):
        """Run the fine SCF calculation."""
        inputs = self.ctx.scf_inputs
        inputs["clean_workdir"] = orm.Bool(
            False
        )  # Needed for phonon calculations and d3hess
        inputs["pw"]["structure"] = self.ctx.current_structure
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.current_structure.label}_scf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        future = self.submit(PwBaseWorkChain, **inputs)
        self.report(
            "Launched PwBaseWorkChain for fine SCF calculation."
        )
        return ToContext(scf_workchain=future)

    def inspect_scf(self):
        """Inspect the result of the fine SCF calculation."""
        workchain = self.ctx.scf_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_SCF_FAILED
        self.out_many(
            self.exposed_outputs(
                workchain, PwBaseWorkChain, namespace="scf"
            )
        )

    def should_run_d3hess(self):
        """Check if the D3hess calculation should be run."""
        return (
            "d3hess" in self.inputs and self.inputs.d3hess is not None
        )

    def run_d3hess(self):
        """Run the D3hess calculation."""
        inputs = self.ctx.d3hess_inputs
        inputs["parent_folder"] = (
            self.ctx.scf_workchain.outputs.remote_folder
        )
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.current_structure.label}_d3hess"

        inputs = prepare_process_inputs(D3hessCalculation, inputs)
        future = self.submit(D3hessCalculation, **inputs)
        self.report("Launched D3hessCalculation.")
        return ToContext(d3hess_calculation=future)

    def inspect_d3hess(self):
        """Inspect the result of the D3hess calculation."""
        calculation = self.ctx.d3hess_calculation
        if not calculation.is_finished_ok:
            return self.exit_codes.ERROR_D3HESS_FAILED
        self.out_many(
            self.exposed_outputs(
                calculation, D3hessCalculation, namespace="d3hess"
            )
        )

    def run_phonon(self):
        """Run the phonon calculation workflow."""
        inputs = self.ctx.phonon_inputs
        inputs["clean_workdir"] = orm.Bool(False)
        inputs["ph"]["ph"][
            "parent_folder"
        ] = self.ctx.scf_workchain.outputs.remote_folder
        inputs["metadata"][
            "label"
        ] = f"{self.ctx.current_structure.label}_phonon"

        inputs = prepare_process_inputs(PhononWorkChain, inputs)
        future = self.submit(PhononWorkChain, **inputs)
        self.report("Launched PhononWorkChain for phonon calculations.")
        return ToContext(phonon_workchain=future)

    def inspect_phonon(self):
        """Inspect the result of the phonon calculation workflow."""
        workchain = self.ctx.phonon_workchain
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_PHONON_FAILED
        self.out_many(
            self.exposed_outputs(
                workchain, PhononWorkChain, namespace="phonon"
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
