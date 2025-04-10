"""Calculations as tools for structure manipulation"""

import numpy as np

from aiida import orm
from aiida.engine import calcfunction

__all__ = ["interpolate_structures", "scale_structure"]


@calcfunction
def interpolate_structures(
    origin: orm.StructureData,
    target: orm.StructureData,
    balance: orm.Float,
) -> orm.StructureData:
    """
    Calcfunction to generate a new structure from `origin` and `target` structures
    which linearly interpolates their atomic positions and cell parameters.
    (balance=0 gives `origin` and balance=1 gives `target` structure)

    :param origin: the origin `aiida.orm.StructureData`
    :param target: the target `aiida.orm.StructureData`
    :param balance: the interpolation coefficient between 0 and 1
    :return: the interpolated `aiida.orm.StructureData`
    """
    ase_origin = origin.get_ase()
    ase_target = target.get_ase()

    try:
        assert len(ase_origin) == len(ase_target)
    except Exception as exc:
        raise IndexError(
            "`origin` and `target` structures must have the same number of atoms"
        ) from exc
    try:
        assert (
            ase_origin.get_atomic_numbers()
            == ase_target.get_atomic_numbers()
        ).all()
    except Exception as exc:
        raise IndexError(
            "`origin` and `target` structures must have the same ordering of elements to interpolate"
        ) from exc
    try:
        assert 0 <= balance.value <= 1
    except Exception as exc:
        raise ValueError("`balance` must be between 0 and 1") from exc

    ase_interpolated = ase_origin.copy()
    ase_interpolated.set_positions(
        balance.value * ase_target.positions
        + (1.0 - balance.value) * ase_origin.positions
    )
    ase_interpolated.set_cell(
        balance.value * ase_target.cell
        + (1.0 - balance.value) * ase_origin.cell,
        scale_atoms=False,
    )
    interpolated = orm.StructureData(ase=ase_interpolated)

    return interpolated


@calcfunction
def scale_structure(
    structure: orm.StructureData,
    scale_factor: orm.Float,
    scale_a: orm.Bool,
    scale_b: orm.Bool,
    scale_c: orm.Bool,
) -> orm.StructureData:
    """
    Calcfunction to scale a structure (cell and positions) by a given factor along
    the a, b, and c lattice vectors (if the corresponding `scale_a`, `scale_b`, and
    `scale_c` flags are set to True).

    :param structure: the input `aiida.orm.StructureData`
    :param scale_factor: the scaling factor
    :param scale_a: flag to scale the a lattice vector
    :param scale_b: flag to scale the b lattice vector
    :param scale_c: flag to scale the c lattice vector
    :return: the scaled `aiida.orm.StructureData`
    """
    ase_structure = structure.get_ase()
    ase_scaled = ase_structure.copy()
    scales = [scale_a.value, scale_b.value, scale_c.value]
    scale_factors = [
        scale_factor.value if scale else 1.0 for scale in scales
    ]
    ase_scaled.set_cell(
        np.diag(scale_factors) @ ase_structure.cell, scale_atoms=True
    )

    return orm.StructureData(ase=ase_scaled)
