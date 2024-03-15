"""
Functions for changing builder.
"""

import typing as ty

from aiida_quantumespresso.calculations import BasePwCpInputGenerator
from aiida_quantumespresso.calculations.namelists import (
    NamelistsCalculation,
)
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannier90_workflows.utils.workflows.builder.setter import (
    get_metadata,
    get_settings_for_kpool,
)
from aiida_wannier90_workflows.workflows.bands import (
    Wannier90BandsWorkChain,
)
from aiida_wannier90_workflows.workflows.base.open_grid import (
    OpenGridBaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.projwfc import (
    ProjwfcBaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import (
    Wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.projwfcbands import (
    ProjwfcBandsWorkChain,
)

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import (
    CalcJob,
    ProcessBuilder,
    ProcessBuilderNamespace,
)

from aiida_mycodes.workflows.pdos import MyPdosWorkChain
from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain


def set_parallelization(
    builder: ty.Union[
        ProcessBuilder, ProcessBuilderNamespace, AttributeDict
    ],
    parallelization: dict = None,
    process_class: ty.Union[
        PwCalculation,
        PwBaseWorkChain,
        PwRelaxWorkChain,
        PwBandsWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
        CalcJob,
        MyPwBaseWorkChain,
        MyPdosWorkChain,
    ] = Wannier90BandsWorkChain,
) -> None:
    """Set parallelization for Wannier90BandsWorkChain.

    :param builder: a builder or its subport, or a ``AttributeDict`` which is the inputs for the builder.
    :type builder: ProcessBuilderNamespace
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    default_max_wallclock_seconds = 24 * 3600
    default_num_mpiprocs_per_machine = None
    default_npool = 1
    default_num_machines = 1
    default_queue_name = None
    default_account = None

    if parallelization is None:
        parallelization = {}

    max_wallclock_seconds = parallelization.get(
        "max_wallclock_seconds",
        default_max_wallclock_seconds,
    )
    num_mpiprocs_per_machine = parallelization.get(
        "num_mpiprocs_per_machine",
        default_num_mpiprocs_per_machine,
    )
    npool = parallelization.get(
        "npool",
        default_npool,
    )
    num_machines = parallelization.get(
        "num_machines",
        default_num_machines,
    )
    queue_name = parallelization.get(
        "queue_name",
        default_queue_name,
    )
    account = parallelization.get(
        "account",
        default_account,
    )

    # I need to prune the builder, otherwise e.g. initially builder.relax is
    # an empty dict but the following code will change it to non-empty,
    # leading to invalid process spec such as code not found for PwRelaxWorkChain.
    # pylint: disable=protected-access
    if isinstance(builder, dict):
        pruned_builder = builder
    else:
        pruned_builder = builder._inputs(prune=True)
    # pylint: enable=protected-access

    metadata = get_metadata(
        num_mpiprocs_per_machine=num_mpiprocs_per_machine,
        max_wallclock_seconds=max_wallclock_seconds,
        num_machines=num_machines,
        queue_name=queue_name,
        account=account,
    )
    settings = get_settings_for_kpool(npool=npool)

    # PwCalculation is a subclass of BasePwCpInputGenerator,
    # `parallelization` is defined in BasePwCpInputGenerator
    if issubclass(process_class, BasePwCpInputGenerator):
        if "parallelization" in builder:
            base_parallelization = builder["parallelization"].get_dict()
        else:
            base_parallelization = {}

        base_parallelization["npool"] = npool

        builder["parallelization"] = orm.Dict(base_parallelization)
        builder["metadata"] = metadata

    elif process_class == PwBaseWorkChain:
        set_parallelization(
            builder["pw"],
            parallelization=parallelization,
            process_class=BasePwCpInputGenerator,
        )

    elif process_class == MyPwBaseWorkChain:
        set_parallelization(
            builder["pw"],
            parallelization=parallelization,
            process_class=BasePwCpInputGenerator,
        )

    elif process_class == MyPdosWorkChain:
        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=MyPwBaseWorkChain,
            )

        if "nscf" in pruned_builder:
            set_parallelization(
                builder["nscf"],
                parallelization=parallelization,
                process_class=MyPwBaseWorkChain,
            )

    # Includes PwCalculation, OpenGridCalculation, ProjwfcCalculation
    elif issubclass(process_class, NamelistsCalculation):
        builder["metadata"] = metadata
        builder["settings"] = settings

    elif issubclass(process_class, CalcJob):
        builder["metadata"] = metadata

    elif process_class == OpenGridBaseWorkChain:
        set_parallelization(
            builder["open_grid"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )
        # For now open_grid.x has memory issue, I run it with less cores
        # open_grid_metadata = copy.deepcopy(metadata)
        # open_grid_metadata['options']['resources']['num_mpiprocs_per_machine'] = (num_mpiprocs_per_machine // npool)
        # builder.open_grid['open_grid']['metadata'] = open_grid_metadata

    elif process_class == ProjwfcBaseWorkChain:
        set_parallelization(
            builder["projwfc"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )

    elif process_class == Pw2wannier90BaseWorkChain:
        set_parallelization(
            builder["pw2wannier90"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )

    elif process_class == Wannier90Calculation:
        builder["metadata"] = metadata

    elif process_class == Wannier90BaseWorkChain:
        set_parallelization(
            builder["wannier90"],
            parallelization=parallelization,
            process_class=Wannier90Calculation,
        )

    elif process_class == PwRelaxWorkChain:
        if "base" in pruned_builder:
            set_parallelization(
                builder["base"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )
        if "base_final_scf" in pruned_builder:
            set_parallelization(
                builder["base_final_scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    elif process_class == PwBandsWorkChain:
        if "relax" in pruned_builder:
            set_parallelization(
                builder["relax"],
                parallelization=parallelization,
                process_class=PwRelaxWorkChain,
            )

        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "bands" in pruned_builder:
            set_parallelization(
                builder["bands"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    elif process_class == ProjwfcBandsWorkChain:
        if "relax" in pruned_builder:
            set_parallelization(
                builder["relax"],
                parallelization=parallelization,
                process_class=PwRelaxWorkChain,
            )

        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "bands" in pruned_builder:
            set_parallelization(
                builder["bands"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "projwfc" in pruned_builder:
            set_parallelization(
                builder["projwfc"],
                parallelization=parallelization,
                process_class=ProjwfcBaseWorkChain,
            )

    elif process_class == Wannier90BandsWorkChain:
        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "nscf" in pruned_builder:
            set_parallelization(
                builder["nscf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "open_grid" in pruned_builder:
            set_parallelization(
                builder["open_grid"],
                parallelization=parallelization,
                process_class=OpenGridBaseWorkChain,
            )

        if "projwfc" in pruned_builder:
            set_parallelization(
                builder["projwfc"],
                parallelization=parallelization,
                process_class=ProjwfcBaseWorkChain,
            )

        if "pw2wannier90" in pruned_builder:
            set_parallelization(
                builder["pw2wannier90"],
                parallelization=parallelization,
                process_class=Pw2wannier90BaseWorkChain,
            )

        if "wannier90" in pruned_builder:
            set_parallelization(
                builder["wannier90"],
                parallelization=parallelization,
                process_class=Wannier90BaseWorkChain,
            )
