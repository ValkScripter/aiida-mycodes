"""`Parser` implementation for the `unfold.x` code of UNFOLD-X."""

from pathlib import Path
from typing import TextIO

import numpy as np

from aiida import orm
from aiida.parsers.parser import Parser


class UnfoldParser(Parser):
    """
    Parser for the output of unfold.x.

    This parser produces:
    - 'output_parameters': a Dict containing parsed spectral function data from the unfold.x calculation.

    Assumes the presence of the text output file containing the spectral function data.
    """

    def parse(self, **kwargs):
        """Parse the retrieved temporary files into output nodes."""
        # Get the output data filename (the binary output file from unfold.x)
        output_data_filename = "unfold.dat"
        assert isinstance(output_data_filename, str)

        # Parse the spectral function from the binary output file
        try:
            retrieved_temporary_folder = kwargs[
                "retrieved_temporary_folder"
            ]
            with open(
                Path(retrieved_temporary_folder) / output_data_filename,
                "rb",
            ) as file_handler:
                output_data = np.fromfile(file_handler)
        except OSError as exc:
            self.logger.error(f"Error reading data file: {exc}")
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                f"Error parsing spectral function: {exception}"
            )
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE

        # Get the number of k-points from the input kpoints
        try:
            assert isinstance(self.node.inputs.kpoints, orm.KpointsData)
            nkpts = len(self.node.inputs.kpoints.get_kpoints())
        except Exception as exc:
            self.logger.error(
                f"Could not get number of k-points: {exc}"
            )
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE

        # Get fermi_energy from input
        try:
            assert isinstance(self.node.inputs.fermi_energy, orm.Float)
            fermi_energy = self.node.inputs.fermi_energy.value
            assert isinstance(fermi_energy, (int, float))
        except (AttributeError, AssertionError) as exc:
            self.logger.error(
                f"Could not get fermi_energy from inputs: {exc}"
            )
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE

        # Parse the spectral function data
        spectral_data = self.parse_spectral_function_from_data(
            output_data, nkpts, fermi_energy
        )

        # Create output Dict node
        output_contourf = orm.ArrayData()
        output_contourf.set_array(
            "kpts", np.array(spectral_data["contourf"]["kpts"])
        )
        output_contourf.set_array(
            "energies", np.array(spectral_data["contourf"]["energies"])
        )
        output_contourf.set_array(
            "weights", np.array(spectral_data["contourf"]["weights"])
        )
        self.out("output_contourf", output_contourf)

        output_scatter = orm.ArrayData()
        output_scatter.set_array(
            "kpts", np.array(spectral_data["scatter"]["kpts"])
        )
        output_scatter.set_array(
            "bands", np.array(spectral_data["scatter"]["bands"])
        )
        output_scatter.set_array(
            "sizes", np.array(spectral_data["scatter"]["sizes"])
        )
        self.out("output_scatter", output_scatter)

    def parse_spectral_function_from_data(
        self,
        data: np.ndarray,
        nkpts: int,
        fermi_energy: float,
        prominence: float = 0.5,
        rel_height: float = 0.5,
    ) -> dict:
        """
        Parse the binary data from unfold.x output and return the spectral function data.

        This method uses the logic from read_unfold_output in tools/unfold.py.

        :param data: Binary data array read from the output file.
        :param nkpts: Number of k-points.
        :param fermi_energy: Fermi energy in eV.
        :param prominence: Peak prominence parameter for peak detection.
        :param rel_height: Relative height parameter for peak width calculation.
        :return: Dictionary with spectral function data.
        """
        from scipy.signal import find_peaks, peak_widths

        # Reshape data according to unfold.x binary output format
        # The binary file contains triplets of (kpoint_index, energy, weight) for each point
        # in the spectral function grid. We extract energies and weights using slice patterns:
        # - data[1::3] gets every third element starting from index 1 (energies)
        # - data[2::3] gets every third element starting from index 2 (weights)
        # Then reshape to (nkpts, n_energy_points) matrix
        energies = data[1::3].reshape((nkpts, -1)) - fermi_energy
        weights = data[2::3].reshape((nkpts, -1))
        kpts = np.meshgrid(
            range(energies.shape[1]), range(energies.shape[0])
        )[1]

        # Find peaks in the spectral function
        bands = []
        sizes = []

        for i in range(energies.shape[0]):
            peaks, _ = find_peaks(weights[i, :], prominence=prominence)
            widths, _, _, _ = peak_widths(
                weights[i, :], peaks, rel_height=rel_height
            )
            bands.append(energies[i, peaks])
            sizes.append(widths)

        max_bands = max(map(len, bands)) if bands else 0

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

        # Return data in a format suitable for serialization
        return {
            "contourf": {
                "kpts": kpts.tolist(),
                "energies": energies.tolist(),
                "weights": weights.tolist(),
            },
            "scatter": {
                "kpts": kpts[:, :max_bands].tolist(),
                "bands": bands.tolist(),
                "sizes": sizes.tolist(),
            },
            "nkpts": nkpts,
            "fermi_energy": fermi_energy,
        }
