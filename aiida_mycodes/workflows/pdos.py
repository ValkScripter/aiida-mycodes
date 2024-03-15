"""
My Workchain to run a Quantum ESPRESSO pw.x calculation with:
- native automated error handling and restarts.
- custom protocol files in subdirectory `my_protocols`
"""

from aiida_quantumespresso.workflows.pdos import (
    PdosWorkChain,
    to_aiida_type,
    validate_dos,
    validate_inputs,
    validate_nscf,
    validate_projwfc,
    validate_scf,
)
from aiida_quantumespresso.workflows.protocols.utils import (
    recursive_merge,
)
from importlib_resources import files

from aiida import orm
from aiida.engine import if_
from aiida.plugins import (
    CalculationFactory,
    GroupFactory,
    WorkflowFactory,
)

if __name__ == "__main__":
    from aiida_mycodes.workflows import protocols
    from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain
else:
    from . import protocols
    from ..workflows.pw.base import MyPwBaseWorkChain

PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
DosCalculation = CalculationFactory("quantumespresso.dos")
ProjwfcCalculation = CalculationFactory("quantumespresso.projwfc")

PwCalculation = CalculationFactory("quantumespresso.pw")
SsspFamily = GroupFactory("pseudo.family.sssp")
PseudoDojoFamily = GroupFactory("pseudo.family.pseudo_dojo")
CutoffsPseudoPotentialFamily = GroupFactory("pseudo.family.cutoffs")


class MyPdosWorkChain(PdosWorkChain):
    """
    My PdosWorkChain calling my own protocol
    """

    @classmethod
    def define(cls, spec):
        # yapf: disable
        """Define the process specification."""
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, help='The input structure.')
        spec.input(
            'serial_clean',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help=('If ``True``, calculations will be run in serial, '
                  'and work directories will be cleaned before the next step.')
        )
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='If ``True``, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.input(
            'dry_run',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help='Terminate workchain steps before submitting calculations (test purposes only).'
        )
        spec.input(
            'align_to_fermi',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help=(
                'If true, Emin=>Emin-Efermi & Emax=>Emax-Efermi, where Efermi is taken from the `nscf` calculation. '
                'Note that it only makes sense to align `Emax` and `Emin` to the fermi level in case they are actually '
                'provided by in the `dos` and `projwfc` inputs, since otherwise the '
            )
        )
        spec.expose_inputs(
            MyPwBaseWorkChain,
            namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `MyPwBaseWorkChain` of the `scf` calculation.',
                'validator': validate_scf,
                'required': False,
                'populate_defaults': False,
            }
        )
        spec.expose_inputs(
            MyPwBaseWorkChain,
            namespace='nscf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `MyPwBaseWorkChain` of the `nscf` calculation.',
                'validator': validate_nscf
            }
        )
        spec.expose_inputs(
            DosCalculation,
            namespace='dos',
            exclude=('parent_folder',),
            namespace_options={
                'help': ('Input parameters for the `dos.x` calculation. Note that the `Emin`, `Emax` and `DeltaE` '
                         'values have to match with those in the `projwfc` inputs.'),
                'validator': validate_dos
            }
        )
        spec.expose_inputs(
            ProjwfcCalculation,
            namespace='projwfc',
            exclude=('parent_folder',),
            namespace_options={
                'help': ('Input parameters for the `projwfc.x` calculation. Note that the `Emin`, `Emax` and `DeltaE` '
                         'values have to match with those in the `dos` inputs.'),
                'validator': validate_projwfc
            }
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_nscf,
            cls.inspect_nscf,
            if_(cls.serial_clean)(
                cls.run_dos_serial,
                cls.inspect_dos_serial,
                cls.run_projwfc_serial,
                cls.inspect_projwfc_serial
            ).else_(
                cls.run_pdos_parallel,
                cls.inspect_pdos_parallel,
            ),
            cls.results,
        )

        spec.exit_code(202, 'ERROR_INVALID_INPUT_KPOINTS',
            message='Neither the `kpoints` nor the `kpoints_distance` input was specified for base or nscf namespaces.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_NSCF',
            message='the NSCF sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_DOS',
            message='the DOS sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_PROJWFC',
            message='the PROJWFC sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_BOTH',
            message='both the DOS and PROJWFC sub process failed')

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(DosCalculation, namespace='dos')
        spec.expose_outputs(ProjwfcCalculation, namespace='projwfc')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        return files(protocols) / "pdos.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        dos_code,
        projwfc_code,
        structure,
        protocol=None,
        overrides=None,
        options=None,
        **kwargs,
    ):  # pylint: disable=too-many-locals
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param dos_code: the ``Code`` instance configured for the ``quantumespresso.dos`` plugin.
        :param projwfc_code: the ``Code`` instance configured for the ``quantumespresso.projwfc`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """

        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)
        scf = MyPwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("scf", None),
            options=options,
            **kwargs,
        )
        scf["pw"].pop("structure", None)
        scf.pop("clean_workdir", None)
        nscf = MyPwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("nscf", None),
            options=options,
            **kwargs,
        )
        nscf["pw"].pop("structure", None)
        nscf["pw"]["parameters"]["SYSTEM"].pop("smearing", None)
        nscf["pw"]["parameters"]["SYSTEM"].pop("degauss", None)
        nscf.pop("clean_workdir", None)

        metadata_dos = inputs.get("dos", {}).get(
            "metadata", {"options": {}}
        )
        metadata_projwfc = inputs.get("projwfc", {}).get(
            "metadata", {"options": {}}
        )

        if options:
            metadata_dos["options"] = recursive_merge(
                metadata_dos["options"], options
            )
            metadata_projwfc["options"] = recursive_merge(
                metadata_projwfc["options"], options
            )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.scf = scf
        builder.nscf = nscf
        # pylint: disable=no-member
        builder.dos.code = dos_code
        builder.dos.parameters = orm.Dict(
            inputs.get("dos", {}).get("parameters")
        )
        builder.dos.metadata = metadata_dos
        builder.projwfc.code = projwfc_code
        builder.projwfc.parameters = orm.Dict(
            inputs.get("projwfc", {}).get("parameters")
        )
        builder.projwfc.metadata = metadata_projwfc
        # pylint: enable=no-member

        return builder


if __name__ == "__main__":
    print("hi")
    print(PdosWorkChain.get_protocol_filepath())
    print(MyPdosWorkChain.get_protocol_filepath())
