"""`CalcJob` implementation for the unfold.x code of UNFOLD-X."""

import pathlib

import numpy as np

from aiida import orm
from aiida.common import datastructures, exceptions
from aiida_quantumespresso.utils.convert import (
    conv_to_fortran,
    convert_input_to_namelist_entry,
)

try:
    from aiida_quantumespresso.calculations.namelists import NamelistsCalculation
except ImportError:
    # Fallback for development/testing when aiida_quantumespresso is not installed
    # Note: This fallback uses CalcJob as base but UnfoldCalculation overrides
    # prepare_for_submission completely, so it doesn't depend on NamelistsCalculation-specific methods
    from aiida.engine import CalcJob as NamelistsCalculation


class UnfoldCalculation(NamelistsCalculation):
    """AiiDA calculation plugin for the `unfold.x` code of UNFOLD-X.

    This class defines the interface for running the `unfold.x` band unfolding code within the AiiDA framework.
    It specifies the required and optional inputs, expected outputs, and handles the preparation of input files
    and retrieval of output files for band unfolding spectral function calculations.

    Inputs:
        - base_structure (StructureData, required): The primitive/base structure.
        - supercell_structure (StructureData, required): The supercell structure.
        - kpoints (KpointsData, required): The k-point mesh for unfolding.
        - parent_folder (RemoteData/FolderData, required): Parent folder from pw bands calculation.
        - parameters (Dict, required): Calculation parameters including fermi_energy, Emin, Emax, deltaE, broadening.
        - metadata.options.input_filename (str, default="unfold.in"): Name of the input file.
        - metadata.options.output_filename (str, default="unfold.out"): Name of the output file.
        - metadata.options.parser_name (str, default="mycodes.unfold"): Name of the parser to use.
        - metadata.options.withmpi (bool, default=False): Whether to run with MPI.

    Outputs:
        - output_parameters (Dict): Parsed output data including spectral function data.

    Exit Codes:
        - 301: ERROR_NO_RETRIEVED_TEMPORARY_FOLDER - The retrieved temporary folder could not be accessed.
        - 302: ERROR_OUTPUT_STDOUT_MISSING - The retrieved folder did not contain the required stdout output file.
        - 310: ERROR_OUTPUT_STDOUT_READ - The stdout output file could not be read.
        - 311: ERROR_OUTPUT_STDOUT_PARSE - The stdout output file could not be parsed.
        - 312: ERROR_OUTPUT_STDOUT_INCOMPLETE - The stdout output file was incomplete.
        - 340: ERROR_OUT_OF_WALLTIME_INTERRUPTED - The calculation stopped prematurely due to walltime exhaustion.
        - 350: ERROR_UNEXPECTED_PARSER_EXCEPTION - The parser raised an unexpected exception.
    """

    _DEFAULT_INPUT_FILE = "unfold.in"
    _DEFAULT_OUTPUT_FILE = "unfold.out"
    _UNFOLD_DATA_FILE = "unfold.out"  # The binary output file to parse

    _default_namelists = ['INPUTUN']
    _blocked_keywords = []

    _internal_retrieve_list = []
    _default_parser = 'mycodes.unfold'

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # Override parent_folder to make it required
        spec.input(
            'parent_folder',
            valid_type=(orm.RemoteData, orm.FolderData),
            required=True,
            help='Use a remote or local folder from parent pw bands calculation.',
        )

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

        # Parameters contains: fermi_energy, Emin, Emax, deltaE, broadening, and optionally proj_orbitals
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            required=True,
            help='Calculation parameters: fermi_energy (required), Emin, Emax, deltaE, broadening, proj_orbitals (optional list).',
        )
        spec.input(
            'settings',
            valid_type=orm.Dict,
            required=False,
            help='Use an additional node for special settings'
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
            default=cls._default_parser,
        )
        spec.input(
            "metadata.options.withmpi", valid_type=bool, default=False
        )

        # Output ports
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Parsed spectral function data from unfold.x calculation.",
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

        # Get parameters
        params = self.inputs.parameters.get_dict()
        fermi_energy = params['fermi_energy']
        Emin = params.get('Emin', -5.0)
        Emax = params.get('Emax', 5.0)
        deltaE = params.get('deltaE', 0.01)
        broadening = params.get('broadening', 0.1)
        proj_orbitals = params.get('proj_orbitals', None)

        # Get the cell matrices
        supercell = np.array(self.inputs.supercell_structure.cell)
        primitive = np.array(self.inputs.base_structure.cell)

        # Get k-points from KpointsData
        kpoints_array = self.inputs.kpoints.get_kpoints()

        # Create the input file content using adapted logic from create_unfold_input
        input_filename = self.inputs.metadata.options.input_filename
        with folder.open(input_filename, "w") as infile:
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
                "filout", self._UNFOLD_DATA_FILE
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
                5,  # round_decimals fixed to 5
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
            input_file_content += f"{len(kpoints_array)}\n"
            for kpt in kpoints_array:
                a, b, c = list(kpt)
                input_file_content += f"{conv_to_fortran(a)}\t{conv_to_fortran(b)}\t{conv_to_fortran(c)}\t0.0\n"
            input_file_content += "\n"

            infile.write(input_file_content)

        # Handle parent_folder setup (similar to NamelistsCalculation)
        remote_copy_list = []
        local_copy_list = []
        remote_symlink_list = []

        # Get settings if provided
        settings = {}
        if 'settings' in self.inputs:
            settings = self.inputs.settings.get_dict()

        symlink = settings.get('PARENT_FOLDER_SYMLINK', False)
        remote_list = remote_symlink_list if symlink else remote_copy_list

        source = self.inputs.parent_folder

        if isinstance(source, orm.RemoteData):
            dirname = settings.get('PARENT_CALC_OUT_SUBFOLDER', './out/')
            dirpath = pathlib.Path(source.get_remote_path()) / dirname
            remote_list.append((source.computer.uuid, str(dirpath), './out/'))
        elif isinstance(source, orm.FolderData):
            local_copy_list.append((source.uuid, './out/', './out/'))

        # Code information
        codeinfo = datastructures.CodeInfo()
        codeinfo.stdin_name = self.inputs.metadata.options.input_filename
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid

        # Calculation information
        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        # Retrieve files
        calcinfo.retrieve_list = [
            self.inputs.metadata.options.output_filename,
            self._UNFOLD_DATA_FILE
        ]

        return calcinfo
