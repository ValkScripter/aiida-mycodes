"""`Parser` implementation for the `unklist.x` code of UNFOLD-X."""

from pathlib import Path
from typing import TextIO

import numpy as np

from aiida import orm
from aiida.parsers.parser import Parser


class UnklistParser(Parser):
    """
    Parser for the output of unklist.x.

    This parser produces two outputs:
    - 'output_parameters': a Dict containing parsed metadata from the standard output.
    - 'output_kpoints': a KpointsData node containing the folded k-points for band unfolding.

    Assumes the presence of the output file containing the k-point list in the standard unklist.x format.
    """

    def parse(self, **kwargs):
        """Parse the retrieved files into output nodes."""
        # Get the output filename from the calculation metadata
        try:
            output_filename = self.node.get_option("output_filename")
            assert isinstance(output_filename, str)
        except Exception:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        # Parse the k-points from the output file
        try:
            retrieved_temporary_folder = kwargs[
                "retrieved_temporary_folder"
            ]
            with open(
                Path(retrieved_temporary_folder) / output_filename
            ) as file_handler:
                kpoints_data = self.parse_kpoints(file_handler)
                self.out("output_kpoints", kpoints_data)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(f"Error parsing k-points: {exception}")
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE

    def parse_kpoints(self, file_handler: TextIO) -> orm.KpointsData:
        """
        Parse the k-points from the unklist.x output file.

        The output file contains k-points in crystal coordinates in the format:
        K_POINTS crystal
        <number of k-points>
        <kx> <ky> <kz> <weight>
        ...

        :param file_handler: File handler for the output file.
        :return: KpointsData node with the folded k-points.
        """
        lines = file_handler.readlines()

        # Find the K_POINTS line
        while lines and lines[0].strip() != "K_POINTS crystal":
            lines.pop(0)

        if not lines:
            self.logger.error(
                "K_POINTS crystal section not found in output file"
            )
            raise ValueError

        lines.pop(0)  # Remove the K_POINTS line

        # Get the number of k-points
        if not lines:
            self.logger.error(
                "Expected number of k-points but reached end of file"
            )
            raise ValueError
        num_kpoints = int(lines.pop(0).split()[0])

        # Parse k-points
        kpoints = []
        for _ in range(num_kpoints):
            if not lines:
                self.logger.error(
                    f"Expected {num_kpoints} k-points but found fewer lines"
                )
                raise ValueError
            kpoint_line = lines.pop(0).strip().split()
            kpoints.append(list(map(float, kpoint_line[:3])))

        kpoints_array = np.array(kpoints)

        # Create KpointsData node
        kpoints_data = orm.KpointsData()
        kpoints_data.set_cell_from_structure(
            self.node.inputs.supercell_structure
        )
        kpoints_data.set_kpoints(kpoints_array, cartesian=False)

        return kpoints_data
