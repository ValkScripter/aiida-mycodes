"""
Command line scripts to launch a `MyPwBaseWorkChain`.
"""

from aiida_quantumespresso.cli.utils import options
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
from aiida.engine import submit

from aiida_mycodes.cli.utils import options as own_options
from aiida_mycodes.cli.workflows import RUN, cmd_launch
from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain


def submit_workflow(
    code: orm.Code,
    structure: orm.StructureData,
    protocol: str,
    ppn: int,
    npool: int,
    pseudo_family: options.PseudoFamilyType,
    clean_workdir: bool = False,
    run: bool = False,
):
    """Submit a ``MyPwBaseWorkChain`` to perform scf calculation."""

    builder = MyPwBaseWorkChain.get_builder_from_protocol(
        code, structure=structure, protocol=protocol
    )

    builder["clean_workdir"] = clean_workdir
    builder["pseudo_family"] = pseudo_family

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": ppn,
    }
    if npool:
        parallelization["npool"] = npool
    parallelization["max_wallclock_seconds"] = builder["pw"]["metadata"]["options"][
        "max_wallclock_seconds"
    ]

    set_parallelization(builder, parallelization, process_class=PwBaseWorkChain)

    print_builder(builder)

    if run:
        submit(builder)


@cmd_launch.command("pw-base")
@options_core.CODE(
    required=True,
    type=types.CodeParamType(entry_point="quantumespresso.pw"),
)
@options.STRUCTURE(required=True)
@own_options.PROTOCOL()
@own_options.PPN()
@own_options.NPOOL()
@options.PSEUDO_FAMILY(
    required=False,
    default="SSSP/1.3/PBEsol/efficiency",
    show_default=True,
)
@options.CLEAN_WORKDIR()
@decorators.with_dbenv()
@RUN()
def launch_workflow(
    code, structure, protocol, ppn, npool, pseudo_family, clean_workdir, run
):
    """
    Run a ``MyPwBaseWorkChain`` to perform scf calculation.
    """
    submit_workflow(
        code, structure, protocol, ppn, npool, pseudo_family, clean_workdir, run
    )


if __name__ == "__main__":
    launch_workflow()  # pylint: disable=no-value-for-parameter
