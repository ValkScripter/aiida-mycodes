"""
Command line scripts to launch a `MyPdosWorkChain`.
"""

from aiida_quantumespresso.cli.utils import options
from aiida_wannier90_workflows.utils.code import (
    check_codes,
    identify_codes,
)
from aiida_wannier90_workflows.utils.workflows.builder.serializer import (
    print_builder,
)

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.utils import decorators
from aiida.engine import submit

from aiida_mycodes.cli.utils import options as own_options
from aiida_mycodes.cli.workflows import RUN, cmd_launch

from ...utils.workflows.builder.setter import set_parallelization
from ...workflows.pdos import MyPdosWorkChain


def submit_workflow(
    codes: list,
    structure: orm.StructureData,
    protocol: str,
    ppn: int,
    npool: int,
    pseudo_family: options.PseudoFamilyType,
    clean_workdir: bool = False,
    max_wallclock_seconds: int = 86400,
    run: bool = False,
):
    """
    Submit a ``MyPdosWorkChain`` to perform:
    - scf calculation to get the wfcs
    - nscf calculation to get eigenvalues on a denser k-point mesh
    - dos calculation to get total DoS
    - projwfc calculation to get Bloch wfc DoS projections on atomic orbitals
    """

    codes = identify_codes(codes)
    check_codes(codes, required_codes=["pw", "dos", "projwfc"])

    pw_overrides = dict(
        pseudo_family=pseudo_family.label,
    )

    overrides = {
        "clean_workdir": clean_workdir,
        "scf": pw_overrides,
        "nscf": pw_overrides,
    }

    builder = MyPdosWorkChain.get_builder_from_protocol(
        codes["pw"],
        codes["dos"],
        codes["projwfc"],
        structure=structure,
        protocol=protocol,
        overrides=overrides,
    )

    print(
        # pylint: disable=no-member
        "align_to_fermi: ",
        builder.align_to_fermi().value,
    )

    # You can change parallelization here
    parallelization = {
        "max_wallclock_seconds": max_wallclock_seconds,
        "num_mpiprocs_per_machine": ppn,
    }
    if npool:
        parallelization["npool"] = npool

    set_parallelization(
        builder, parallelization, process_class=MyPdosWorkChain
    )

    print_builder(builder)

    if run:
        result = submit(builder)
        print(f"Submitted {result.process_label}<{result.pk}>")


@cmd_launch.command("pdos")
@options_core.CODES(required=True)
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
    codes,
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
    Submit a ``MyPdosWorkChain`` to perform scf, nscf, dos, projwfc.
    """
    submit_workflow(
        codes,
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
