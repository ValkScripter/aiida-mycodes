"""`CalcJob` implementation for the unklist.x code of UNFOLD-X."""

from aiida_quantumespresso.utils.convert import (
    conv_to_fortran,
    convert_input_to_namelist_entry,
)
import numpy as np

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob


class UnklistCalculation(CalcJob):
    """AiiDA calculation plugin for the `unklist.x` code of UNFOLD-X.

    This class defines the interface for running the `unklist.x` pre-processing code within the AiiDA framework.
    It specifies the required and optional inputs, expected outputs, and handles the preparation of input files
    and retrieval of output files for generating k-point lists for band unfolding.

    Inputs:
        - base_structure (StructureData, required): The primitive/base structure.
        - supercell_structure (StructureData, required): The supercell structure.
        - kpoints (KpointsData, required): The k-points for unfolding in the primitive cell.
        - metadata.options.input_filename (str, default="unklist.in"): Name of the input file.
        - metadata.options.output_filename (str, default="unklist.out"): Name of the output file.
        - metadata.options.parser_name (str, default="mycodes.unklist"): Name of the parser to use.
        - metadata.options.withmpi (bool, default=False): Whether to run with MPI.

    Outputs:
        - output_kpoints (KpointsData): The generated k-point list for band unfolding.

    Exit Codes:
        - 301: ERROR_NO_RETRIEVED_TEMPORARY_FOLDER - The retrieved temporary folder could not be accessed.
        - 302: ERROR_OUTPUT_STDOUT_MISSING - The retrieved folder did not contain the required stdout output file.
        - 310: ERROR_OUTPUT_STDOUT_READ - The stdout output file could not be read.
        - 311: ERROR_OUTPUT_STDOUT_PARSE - The stdout output file could not be parsed.
        - 312: ERROR_OUTPUT_STDOUT_INCOMPLETE - The stdout output file was incomplete.
        - 340: ERROR_OUT_OF_WALLTIME_INTERRUPTED - The calculation stopped prematurely due to walltime exhaustion.
        - 350: ERROR_UNEXPECTED_PARSER_EXCEPTION - The parser raised an unexpected exception.
    """

    _DEFAULT_INPUT_FILE = "unklist.in"
    _DEFAULT_OUTPUT_FILE = "unklist.out"

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # Input ports
        spec.input(
            "base_structure",
            valid_type=orm.StructureData,
            required=True,
            help="The primitive/base structure.",
        )
        spec.input(
            "supercell_structure",
            valid_type=orm.StructureData,
            required=True,
            help="The supercell structure.",
        )
        spec.input(
            "kpoints",
            valid_type=orm.KpointsData,
            required=True,
            help="The k-points for unfolding in the primitive cell.",
        )

        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls._DEFAULT_INPUT_FILE,
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="mycodes.unklist",
        )
        spec.input(
            "metadata.options.withmpi", valid_type=bool, default=False
        )

        # Output ports
        spec.output(
            "output_kpoints",
            valid_type=orm.KpointsData,
            help="The generated k-point list for band unfolding.",
        )

        # Standard exceptions
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_TEMPORARY_FOLDER",
            message="The retrieved temporary folder could not be accessed.",
        )
        spec.exit_code(
            302,
            "ERROR_OUTPUT_STDOUT_MISSING",
            message="The retrieved folder did not contain the required stdout output file.",
        )
        spec.exit_code(
            310,
            "ERROR_OUTPUT_STDOUT_READ",
            message="The stdout output file could not be read.",
        )
        spec.exit_code(
            311,
            "ERROR_OUTPUT_STDOUT_PARSE",
            message="The stdout output file could not be parsed.",
        )
        spec.exit_code(
            312,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The stdout output file was incomplete.",
        )
        spec.exit_code(
            340,
            "ERROR_OUT_OF_WALLTIME_INTERRUPTED",
            message="The calculation stopped prematurely because it ran out of walltime but the job was killed by the "
            "scheduler before the files were safely written to disk for a potential restart.",
        )
        spec.exit_code(
            350,
            "ERROR_UNEXPECTED_PARSER_EXCEPTION",
            message="The parser raised an unexpected exception: {exception}",
        )

    def prepare_for_submission(self, folder):
        """Prepare the calculation job for submission by transforming input nodes into input files.

        In addition to the input files being written to the sandbox folder, a `CalcInfo` instance will be returned that
        contains lists of files that need to be copied to the remote machine before job submission, as well as file
        lists that are to be retrieved after job completion.

        :param folder: a sandbox folder to temporarily write files on disk.
        :return: :class:`~aiida.common.datastructures.CalcInfo` instance.
        """
        # Get the cell matrices
        supercell = np.array(self.inputs.supercell_structure.cell)
        primitive = np.array(self.inputs.base_structure.cell)

        # Get k-points from KpointsData
        kpoints: orm.KpointsData = self.inputs.kpoints
        kpoints_array = kpoints.get_kpoints()

        # Code information
        codeinfo = datastructures.CodeInfo()
        codeinfo.stdin_name = (
            self.inputs.metadata.options.input_filename
        )
        codeinfo.stdout_name = (
            self.inputs.metadata.options.output_filename
        )
        codeinfo.code_uuid = self.inputs.code.uuid

        # Calculation information
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []

        # Retrieve by default the output file
        calcinfo.retrieve_temporary_list = [
            self.inputs.metadata.options.output_filename
        ]

        # Create the input file using the logic from create_unklist_input
        input_filename = self.inputs.metadata.options.input_filename
        with folder.open(input_filename, "w") as infile:
            # Write input file content
            input_file_content = ""
            input_file_content += "&input_unklist\n"
            input_file_content += convert_input_to_namelist_entry(
                "ibrav", 0
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
                5,
            )
            for row in transformation:
                a, b, c = list(map(int, row))
                input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\n"
            input_file_content += "\n\n"

            input_file_content += "UNKPTS crystal\n"
            # Use k-points from KpointsData input
            input_file_content += f"{len(kpoints_array)}\n"
            for kpt in kpoints_array:
                a, b, c = list(kpt)
                input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\t0.0\n"
            input_file_content += "\n"

            infile.write(input_file_content)

        return calcinfo
