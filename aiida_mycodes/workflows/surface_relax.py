"""
WorkChain to automatically perform a relaxation workflow for surface studies. Steps are the following:
1.  Bulk conventional cell vc-relax until volum convergence (PwRelaxWorkChain)
2.  Transform the nulk conventional cell into a slab-consistent bulk cell by cutting it and giving the Z periodicity
3.  Slab-consistent bulk cell vc-relax until volume convergence (PwRelaxWorkChain)
4.  Birch-Murnaghan fit of the EoS from energies of various SCF calculation
    with lattice parameters around previous one to get optimal lattice parameter
5.  Relaxation with Z-free top-slab and adsorbates at ideal in-plane positions with symmetries
6.  Relaxation with free top-slab and adsorbates with no symmetry
7.  Final SCF with symmetries to get the remaining ones
"""

# pylint: disable=protected-access
import pathlib

# from aiida_quantumespresso.common.types import (
#     ElectronicType,
#     RelaxType,
#     RestartType,
#     SpinType,
# )
# from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (  # recursive_merge,
    ProtocolMixin,
)

# from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
# from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from importlib_resources import files
import yaml

from aiida import orm

# from aiida.common import AttributeDict
# from aiida.common.lang import type_check
from aiida.engine.processes import (  # ProcessBuilder,; ToContext,; if_,
    WorkChain,
)
from aiida.orm.nodes.data.base import to_aiida_type

from . import protocols

# import typing as ty


class SurfaceRelaxWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain to automatically perform a relaxation workflow for surface studies. Steps are the following:
    1.  Bulk conventional cell vc-relax until volum convergence (PwRelaxWorkChain)
    2.  Transform the nulk conventional cell into a slab-consistent bulk cell by cutting it and giving the Z periodicity
    3.  Slab-consistent bulk cell vc-relax until volume convergence (PwRelaxWorkChain)
    4.  Birch-Murnaghan fit of the EoS from energies of various SCF calculation
        with lattice parameters around previous one to get optimal lattice parameter
    5.  Relaxation with Z-free top-slab and adsorbates at ideal in-plane positions with symmetries
    6.  Relaxation with free top-slab and adsorbates with no symmetry
    7.  Final SCF with symmetries to get the remaining ones
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.input(
            "bulk_structure",
            valid_type=orm.StructureData,
            help="The input bulk structure with conventional cell.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help=(
                "If True, work directories of all called calculation will be cleaned "
                "at the end of execution."
            ),
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        return files(protocols) / "surface_relax.yaml"

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for various input arguments of the ``get_builder_from_protocol()`` method."""
        path = files(protocols) / "overrides" / "surface_relax.yaml"
        with path.open() as file:
            return yaml.safe_load(file)
