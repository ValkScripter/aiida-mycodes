"""
Command line scripts to launch a `MyPwBaseWorkChain`.
"""

from aiida_quantumespresso.cli.utils import options
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
import click

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
from aiida.engine import submit

from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain

from .. import RUN, cmd_launch


def submit_workflow(
    code: orm.Code,
    structure: orm.StructureData,
    protocol: str,
    ppn: int,
    npool: int,
    run: bool = False,
):
    """Submit a ``MyPwBaseWorkChain`` to perform scf calculation."""

    builder = MyPwBaseWorkChain.get_builder_from_protocol(
        code, structure=structure, protocol=protocol
    )

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": ppn,
    }
    if npool:
        parallelization["npool"] = npool

    set_parallelization(builder, parallelization, process_class=MyPwBaseWorkChain)

    print_builder(builder)

    if run:
        submit(builder)


@cmd_launch.command("pw-base")
@options_core.CODE(
    required=True, type=types.CodeParamType(entry_point="quantumespresso.pw")
)
# @options.STRUCTURE(default=defaults.get_structure)
@options.STRUCTURE(required=True)
@click.option(
    "-p",
    "--protocol",
    type=str,
    default="moderate",
    help="Protocol name to use for the workflow builder [default='moderate']",
)
@click.option(
    "--ppn",
    type=int,
    default=1,
    help="Number of cores of the remote computer to run the job on [default=1]",
)
@click.option("--npool", type=int, default=None, help="Number of pools for the QE job")
@decorators.with_dbenv()
@RUN()
def launch_workflow(code, structure, protocol, ppn, npool, run):
    """
    Run a ``PwBandsWorkChain`` to perform scf calculation.
    """
    submit_workflow(code, structure, protocol, ppn, npool, run)
