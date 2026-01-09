""" Create input files for unfolding tools 'unklist.x' and 'unfold.x' from Quantum ESPRESSO. """

from typing import Dict, List, Optional, Tuple

from aiida_quantumespresso.utils.convert import (
    conv_to_fortran,
    convert_input_to_namelist_entry,
)
from ase.cell import Cell
import numpy as np
from scipy.signal import find_peaks, peak_widths


def create_unklist_input(
    supercell: np.ndarray,
    primitive: np.ndarray,
    path: str,
    density: float = 66.7,
    round_decimals: int = 5,
) -> str:
    """
    Return the content of the input file for the 'unklist.x' tool from Quantum ESPRESSO.

    :param supercell: The supercell matrix.
    :param primitive: The primitive cell matrix.
    :param path: The path to the output file in the primitive cell to fold into the supercell.
    :param density: The density of k-points in the path.
    :param round_decimals: The number of decimals to round the transformation matrix to.
    :return: The content of the input file as a string.
    """
    input_file_content = ""
    input_file_content += "&input_unklist\n"
    input_file_content += convert_input_to_namelist_entry("ibrav", 0)
    input_file_content += "/\n\n"

    input_file_content += "CELL_PARAMETERS angstrom\n"
    for row in supercell:
        a, b, c = list(row)
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\n"
    input_file_content += "\n\n"

    input_file_content += "TRMAT\n"
    transformation = np.round(
        np.linalg.inv(primitive @ np.linalg.inv(supercell)),
        round_decimals,
    )
    for row in transformation:
        a, b, c = list(map(int, row))
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\n"
    input_file_content += "\n\n"

    input_file_content += "UNKPTS crystal\n"
    bandpath = Cell(primitive).bandpath(path, density=density)
    kpts = bandpath.kpts
    input_file_content += f"{len(kpts)}\n"
    for kpt in kpts:
        a, b, c = list(kpt)
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\t0.0\n"
    input_file_content += "\n"

    return input_file_content


def read_unklist_output(output_file: str) -> np.ndarray:
    """
    Read the output file of the 'unklist.x' tool from Quantum ESPRESSO and return the folded k-points.

    :param output_file: The path to the output file.
    :return: The folded k-points.
    """
    with open(output_file, encoding="utf-8") as file_handle:
        lines = file_handle.readlines()

    while lines.pop(0).strip() != "K_POINTS crystal":
        pass
    num_kpoints = int(lines.pop(0).split()[0])
    kpoints = []
    for _ in range(num_kpoints):
        kpoints.append(
            list(map(float, lines.pop(0).strip().split()[:3]))
        )

    return np.array(kpoints)


def create_unfold_input(
    supercell: np.ndarray,
    primitive: np.ndarray,
    path: str,
    fermi_energy: float,
    Emin: float = -5.0,
    Emax: float = 5.0,
    deltaE: float = 0.01,
    broadening: float = 0.1,
    proj_orbitals: Optional[List[int]] = None,
    density: float = 66.7,
    round_decimals: int = 5,
    fileout: str = "./unfold.dat",
) -> str:
    """
    Return the content of the input file for the 'unfold.x' tool from Quantum ESPRESSO.

    :param supercell: The supercell matrix.
    :type supercell: np.ndarray
    :param primitive: The primitive cell matrix.
    :type primitive: np.ndarray
    :param path: The path to the output file in the primitive cell to fold into the supercell.
    :type path: str
    :param fermi_energy: The Fermi energy in eV.
    :type fermi_energy: float
    :param Emin: The minimum energy in eV.
    :type Emin: float
    :param Emax: The maximum energy in eV.
    :type Emax: float
    :param deltaE: The energy step in eV.
    :type deltaE: float
    :param broadening: The broadening for delta function in spectral function.
    :type broadening: float
    :param proj_orbitals: The list of orbital indices to project onto, or None for all orbitals (no projection).
    :type proj_orbitals: Optional[List[int]]
    :param density: The density of k-points in the path.
    :type density: float
    :param round_decimals: The number of decimals to round the transformation matrix to.
    :type round_decimals: int
    :param fileout: The path to the output data file.
    :type fileout: str

    :return: The content of the input file as a string.
    :rtype: str
    """
    input_file_content = ""
    input_file_content += "&inputun\n"
    input_file_content += convert_input_to_namelist_entry(
        "prefix", "aiida"
    )
    input_file_content += convert_input_to_namelist_entry(
        "outdir", "./out/"
    )
    input_file_content += convert_input_to_namelist_entry(
        "Emin", fermi_energy + Emin
    )
    input_file_content += convert_input_to_namelist_entry(
        "Emax", fermi_energy + Emax
    )
    input_file_content += convert_input_to_namelist_entry(
        "DeltaE", deltaE
    )
    input_file_content += convert_input_to_namelist_entry(
        "w", broadening
    )
    input_file_content += convert_input_to_namelist_entry(
        "filout", fileout
    )
    input_file_content += convert_input_to_namelist_entry(
        "nscfklist", False
    )
    input_file_content += convert_input_to_namelist_entry(
        "symreduce", False
    )
    input_file_content += convert_input_to_namelist_entry(
        "verbose", False
    )
    input_file_content += convert_input_to_namelist_entry(
        "write_pkm", False
    )
    input_file_content += convert_input_to_namelist_entry(
        "use_atmproj", proj_orbitals is not None
    )
    input_file_content += "/\n\n"

    input_file_content += "CELL_PARAMETERS angstrom\n"
    for row in supercell:
        a, b, c = list(row)
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\n"
    input_file_content += "\n\n"

    input_file_content += "TRMAT\n"
    transformation = np.round(
        np.linalg.inv(primitive @ np.linalg.inv(supercell)),
        round_decimals,
    )
    for row in transformation:
        a, b, c = list(map(int, row))
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\n"
    input_file_content += "\n\n"

    if proj_orbitals is not None:
        input_file_content += "ATOMIC_ORBITALS explicit\n"
        input_file_content += f"{len(proj_orbitals)}\n"
        input_file_content += "\t".join(map(str, proj_orbitals)) + "\n"
        input_file_content += "\n"

    input_file_content += "UNKPTS crystal\n"
    bandpath = Cell(primitive).bandpath(path, density=density)
    kpts = bandpath.kpts
    input_file_content += f"{len(kpts)}\n"
    for kpt in kpts:
        a, b, c = list(kpt)
        input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\t0.0\n"
    input_file_content += "\n"

    return input_file_content


def read_unfold_output(
    output_file: str,
    nkpts: int,
    fermi_energy: float,
    prominence: float = 0.5,
    rel_height: float = 0.5,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read the output file of the 'unfold.x' tool from Quantum ESPRESSO and return the unfolded spectral function.

    :param output_file: The path to the output file.
    :type output_file: str
    :param nkpts: The number of k-points in the path.
    :type nkpts: int
    :param fermi_energy: The Fermi energy in eV.
    :type fermi_energy: float

    :return: Dictionary with the kpts, energies, and weights for ax.contourf plot under key "contourf"
    and the kpts, bands and sizes for ax.scatter plot under key "scatter" (to highlight the peaks).
    :rtype: Dict[str,Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    data = np.fromfile(output_file)

    energies = data[1::3].reshape((nkpts, -1)) - fermi_energy
    weights = data[2::3].reshape((nkpts, -1))
    kpts = np.meshgrid(
        range(energies.shape[1]), range(energies.shape[0])
    )[1]

    bands = []
    sizes = []

    for i in range(energies.shape[0]):
        peaks, _ = find_peaks(weights[i, :], prominence=prominence)
        widths, _, _, _ = peak_widths(
            weights[i, :], peaks, rel_height=rel_height
        )
        bands.append(energies[i, peaks])
        sizes.append(widths)

    max_bands = max(map(len, bands))

    bands = np.array(
        [
            np.pad(
                b,
                (0, max_bands - len(b)),
                mode="constant",
                constant_values=np.nan,
            )
            for b in bands
        ]
    )
    sizes = np.array(
        [
            np.pad(
                s,
                (0, max_bands - len(s)),
                mode="constant",
                constant_values=np.nan,
            )
            for s in sizes
        ]
    )

    return {
        "contourf": (kpts, energies, weights),
        "scatter": (kpts[:, :max_bands], bands, sizes),
    }
