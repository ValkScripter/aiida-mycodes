"""Calculations as tools for structure manipulation"""

from aiida import orm
from aiida.engine import calcfunction

__all__ = ["interpolate_structures"]


@calcfunction
def interpolate_structures(
    origin: orm.StructureData,
    target: orm.StructureData,
    balance: orm.Float,
) -> orm.StructureData:
    """
    Calcfunction to generate a new structure from `origin` and `target` structures
    which linearly interpolates their atom positions.
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
    interpolated = orm.StructureData(ase=ase_interpolated)

    return interpolated
