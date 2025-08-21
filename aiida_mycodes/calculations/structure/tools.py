"""Calculations as tools for structure manipulation"""

from typing import Optional, Union

from ase import build
import numpy as np

from aiida import orm
from aiida.engine import calcfunction


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

    :param structure: the input Structure to scale
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


@calcfunction
def superposition(
    structure_bot: orm.StructureData,
    structure_top: orm.StructureData,
    fix: orm.Float = lambda: orm.Float(0.0),
    maxstrain: orm.Float = lambda: orm.Float(0.01),
    distance: Union[orm.Float, orm.List] = lambda: orm.Float(3.0),
    z_width: Optional[orm.Float] = None,
    vacuum: orm.Float = lambda: orm.Float(15.0),
    isolated: orm.Bool = lambda: orm.Bool(False),
):
    """
    This function performs a geometric superposition of two structures by aligning one
    (top structure) over another (bottom structure) with adjustable constraints.
    The function adds strain to the structures within the defined limits,
    and recalibrates the periodic boundary conditions (PBC). It also ensures
    adequate separation in the z-direction by extending the unit cell.
    The result includes both the stacked structure and the adjusted strained structures,
    along with their respective strain values.

    :param structure_bot: The bottom structure, provided as an AiiDA StructureData object.
    :type structure_bot: orm.StructureData
    :param structure_top: The top structure, provided as an AiiDA StructureData object.
    :type structure_top: orm.StructureData
    :param fix: The convex constant to interpolate the cell between the two structures.
    (fix = 0 means the stack cell matches the bottom cell, fix = 1 means the stack cell matches the top cell.)
    :type fix: float
    :param maxstrain: The maximum allowed strain for the alignment. Defaults to 0.01.
    :type maxstrain: float
    :param distance: The target interlayer distance used for stacking. Calculated as the distance
    between the closest atoms between the two structures. Defaults to 3.0 Angstrom.
    :type distance: float
    :param z_width: The width of the unit cell in the z-direction.
    If not provided, the z-width is calculated with at least 15 Ang of vaccum.
    :type z_width: float, optional
    :param vacuum: The total amount of vacuum to put on each side in the z-direction. Defaults to 15 Angstrom.
    :type vacuum: float
    :param isolated: If True, the z dim will be calculated for the use of 'assume_isolated' parameter. Defaults to False.
    :type isolated: bool, optional
    :return: A dictionary with the stacked structure, strained bottom structure,
        strained top structure, and their respective strain values.
    :rtype: dict
    """
    fix = fix.value
    maxstrain = maxstrain.value
    distance = (
        distance.value
        if isinstance(distance, orm.Float)
        else distance.get_list()[0]
    )
    if z_width is not None:
        z_width = z_width.value
    vacuum = vacuum.value

    ase_bot = structure_bot.get_ase()
    ase_top = structure_top.get_ase()

    stack, strained_bot, strained_top = build.stack(
        ase_bot,
        ase_top,
        fix=fix,
        maxstrain=maxstrain,
        distance=distance,
        output_strained=True,
    )

    for ase_structure in [stack, strained_bot, strained_top]:
        ase_structure.set_pbc([True, True, False])
        if z_width is not None:
            ase_structure.cell[2, 2] = z_width
            ase_structure.center(axis=2)
        elif isolated:
            DZ = (
                ase_structure.positions[:, 2].max()
                - ase_structure.positions[:, 2].min()
            )
            DZ = max(DZ + 15, 2 * (DZ + 6.0))
            ase_structure.cell[2, 2] = DZ
            ase_structure.center(axis=2)
        else:
            DZ = (
                ase_structure.positions[:, 2].max()
                - ase_structure.positions[:, 2].min()
            )
            ase_structure.cell[2, 2] = DZ + vacuum
            ase_structure.center(axis=2, vacuum=vacuum / 2)

    label = f"{structure_top.label}_ON_{structure_bot.label}_superposition_fix_{fix:.2f}_distance_{distance:.2f}A"

    strain_bot = (
        np.mean(
            strained_bot.cell.cellpar()[:2] / ase_bot.cell.cellpar()[:2]
        )
        - 1.0
    )
    strain_top = (
        np.mean(
            strained_top.cell.cellpar()[:2] / ase_top.cell.cellpar()[:2]
        )
        - 1.0
    )

    stack_structure = orm.StructureData(ase=stack, label=label)
    strained_bot_structure = orm.StructureData(
        ase=strained_bot, label=f"{label}_bot_strained_{strain_bot:.2f}"
    )
    strained_top_structure = orm.StructureData(
        ase=strained_top, label=f"{label}_top_strained_{strain_top:.2f}"
    )

    return {
        "stack_structure": stack_structure,
        "strained_bot_structure": strained_bot_structure,
        "strained_top_structure": strained_top_structure,
        "strain_bot": orm.Float(float(strain_bot)),
        "strain_top": orm.Float(float(strain_top)),
    }
