"""
Command line scripts to launch a `MyPwBaseWorkChain`.
"""

from aiida_quantumespresso.cli.utils import options
from aiida_wannier90_workflows.utils.workflows.builder.serializer import (
    print_builder,
)

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
from aiida.engine import submit

from aiida_mycodes.cli.utils import options as own_options
from aiida_mycodes.cli.workflows import RUN, cmd_launch
from aiida_mycodes.utils.workflows.builder.setter import (
    set_parallelization,
)
from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain


def submit_workflow(
    code: orm.Code,
    structure: orm.StructureData,
    protocol: str,
    ppn: int,
    npool: int,
    pseudo_family: options.PseudoFamilyType,
    clean_workdir: bool = False,
    max_wallclock_seconds: int = 86400,
    run: bool = False,
):
    """Submit a ``MyPwBaseWorkChain`` to perform scf calculation."""

    overrides = dict(
        clean_workdir=clean_workdir,
        pseudo_family=pseudo_family.label,
    )

    builder = MyPwBaseWorkChain.get_builder_from_protocol(
        code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
    )

    # You can change parallelization here
    parallelization = {
        "max_wallclock_seconds": max_wallclock_seconds,
        "num_mpiprocs_per_machine": ppn,
    }
    if npool:
        parallelization["npool"] = npool

    set_parallelization(
        builder, parallelization, process_class=MyPwBaseWorkChain
    )

    print_builder(builder)

    if run:
        result = submit(builder)
        print(f"Submitted {result.process_label}<{result.pk}>")


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
@options.MAX_WALLCLOCK_SECONDS(default=86400)
@decorators.with_dbenv()
@RUN()
def launch_workflow(
    code,
    structure,
    protocol,
    ppn,
    npool,
    pseudo_family,
    clean_workdir,
    max_wallclock_seconds,
    run,
):
    """
    Run a ``MyPwBaseWorkChain`` to perform scf calculation.
    """
    submit_workflow(
        code,
        structure,
        protocol,
        ppn,
        npool,
        pseudo_family,
        clean_workdir,
        max_wallclock_seconds,
        run,
    )


if __name__ == "__main__":
    launch_workflow()  # pylint: disable=no-value-for-parameter
