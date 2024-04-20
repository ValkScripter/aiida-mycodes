"""WorkChain to automatically launch PdosWorkChain with Ldos calculation."""

from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import (
    ProjwfcCalculation,
)
from aiida_quantumespresso.workflows.pdos import (
    PdosWorkChain,
    get_parameter_schema,
    jsonschema,
)
from aiida_quantumespresso.workflows.pdos import (
    validate_inputs as parent_validate_inputs,
)
from aiida_quantumespresso.workflows.protocols.utils import (
    recursive_merge,
)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from importlib_resources import files

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, if_

from . import protocols

# pylint: disable=no-value-for-parameter,too-many-locals,no-member


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `XdosWorkChain`."""

    # Call parent validator
    result = parent_validate_inputs(inputs)
    if result is not None:
        return result


def validate_ldos(value, _):
    """Validate LDoS parameters.

    - shared: Emin | Emax | DeltaE
    - dos.x only: ngauss | degauss | bz_sum
    - projwfc.x only: ngauss | degauss | pawproj | n_proj_boxes | irmin(3,n_proj_boxes) | irmax(3,n_proj_boxes)

    """
    jsonschema.validate(
        value["parameters"].get_dict()["PROJWFC"],
        get_parameter_schema(),
    )
    assert (
        "tdosinboxes" in value["parameters"].get_dict()["PROJWFC"]
    ), "Missing 'tdosinboxes' key in parameters"


class XdosWorkChain(PdosWorkChain):
    """
    WorkChain to launch PdosWorkChain with Ldos calculation.
    It will launch 5 calculations:
    1. SCF calculation (pw.x)
    2. Nscf calculation (pw.x)
    3. (DOS) Projwfc calculation (dos.x)
    4. (PDOS) Projwfc calculation (projwfc.x)
    5. (LDOS) Projwfc calculation (projwfc.x)
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.expose_inputs(
            PdosWorkChain,
        )

        spec.expose_inputs(
            ProjwfcCalculation,
            namespace="ldos",
            exclude=("parent_folder",),
            namespace_options={
                "help": (
                    "Input parameters for the `projwfc.x` calculation for LDoS. Note that the `Emin`, `Emax` and `DeltaE` "
                    "values have to match with those in the `dos` inputs."
                ),
                "validator": validate_ldos,
            },
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
                cls.inspect_projwfc_serial,
                cls.run_ldos_serial,
                cls.inspect_ldos_serial,
            ).else_(
                cls.run_pdos_parallel,
                cls.inspect_pdos_parallel,
            ),
            cls.results,
        )

        spec.exit_code(
            202,
            "ERROR_INVALID_INPUT_KPOINTS",
            message="Neither the `kpoints` nor the `kpoints_distance` input was specified for base or nscf namespaces.",
        )
        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="the SCF sub process failed",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_NSCF",
            message="the NSCF sub process failed",
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_DOS",
            message="the DOS sub process failed",
        )
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_PROJWFC",
            message="the PROJWFC sub process failed",
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_LDOS",
            message="the LDOS sub process failed",
        )
        spec.exit_code(
            406,
            "ERROR_SUB_PROCESS_FAILED_ALL",
            message="all DOS, PROJWFC and LDOS sub processes failed",
        )
        spec.exit_code(
            407,
            "ERROR_SUB_PROCESS_FAILED_TWO",
            message="two of the DOS, PROJWFC and LDOS sub processes failed",
        )

        spec.expose_outputs(PwBaseWorkChain, namespace="nscf")
        spec.expose_outputs(DosCalculation, namespace="dos")
        spec.expose_outputs(ProjwfcCalculation, namespace="projwfc")
        spec.expose_outputs(ProjwfcCalculation, namespace="ldos")

    @classmethod
    def get_protocol_filepath(cls):
        return files(protocols) / "xdos.yaml"

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
    ):
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
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get("scf", None),
            options=options,
            **kwargs,
        )
        scf["pw"].pop("structure", None)
        scf.pop("clean_workdir", None)
        nscf = PwBaseWorkChain.get_builder_from_protocol(
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
        metadata_ldos = inputs.get("ldos", {}).get(
            "metadata", {"options": {}}
        )

        if options:
            metadata_dos["options"] = recursive_merge(
                metadata_dos["options"], options
            )
            metadata_projwfc["options"] = recursive_merge(
                metadata_projwfc["options"], options
            )
            metadata_ldos["options"] = recursive_merge(
                metadata_ldos["options"], options
            )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.scf = scf
        builder.nscf = nscf
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
        builder.ldos.code = projwfc_code
        builder.ldos.parameters = orm.Dict(
            inputs.get("ldos", {}).get("parameters")
        )
        builder.ldos.metadata = metadata_ldos

        return builder

    def _generate_ldos_inputs(self):
        """Run Projwfc calculation, to generate Local Densities of State."""
        ldos_inputs = AttributeDict(
            self.exposed_inputs(ProjwfcCalculation, "ldos")
        )
        ldos_inputs.parent_folder = self.ctx.nscf_parent_folder
        ldos_parameters = self.inputs.ldos.parameters.get_dict()

        if ldos_parameters.pop("align_to_fermi", False):
            ldos_parameters["PROJWFC"]["Emin"] = (
                ldos_parameters.pop("Emin", None) + self.ctx.nscf_fermi
            )
            ldos_parameters["PROJWFC"]["Emax"] = (
                ldos_parameters.pop("Emax", None) + self.ctx.nscf_fermi
            )

        ldos_inputs.parameters = orm.Dict(ldos_parameters)
        ldos_inputs["metadata"]["call_link_label"] = "ldos"
        return ldos_inputs

    def run_ldos_serial(self):
        """Run LDOS Projwfc calculation."""
        ldos_inputs = self._generate_ldos_inputs()

        if self.ctx.dry_run:
            return ldos_inputs

        future_ldos = self.submit(ProjwfcCalculation, **ldos_inputs)
        self.report(
            f"launching LDOS ProjwfcCalculation<{future_ldos.pk}>"
        )
        return ToContext(calc_ldos=future_ldos)

    # pylint: disable=inconsistent-return-statements
    def inspect_ldos_serial(self):
        """Verify that the LDOS Projwfc calculation finished successfully, then clean its remote directory."""
        calculation = self.ctx.calc_ldos
        if not calculation.is_finished_ok:
            self.report(
                f"LDOS ProjwfcCalculation failed with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_LDOS

    def run_pdos_parallel(self):
        """Run DOS, Projwfc, and LDOS calculations in parallel."""
        dos_inputs = self._generate_dos_inputs()
        projwfc_inputs = self._generate_projwfc_inputs()
        ldos_inputs = self._generate_ldos_inputs()

        if self.ctx.dry_run:
            return dos_inputs, projwfc_inputs, ldos_inputs

        future_dos = self.submit(DosCalculation, **dos_inputs)
        self.report(f"launching DosCalculation<{future_dos.pk}>")
        self.to_context(**{"calc_dos": future_dos})

        future_projwfc = self.submit(
            ProjwfcCalculation, **projwfc_inputs
        )
        self.report(
            f"launching ProjwfcCalculation<{future_projwfc.pk}>"
        )
        self.to_context(**{"calc_projwfc": future_projwfc})

        future_ldos = self.submit(ProjwfcCalculation, **ldos_inputs)
        self.report(
            f"launching LDOS ProjwfcCalculation<{future_ldos.pk}>"
        )
        self.to_context(**{"calc_ldos": future_ldos})

    def inspect_pdos_parallel(self):
        """Verify that the DOS, Projwfc, and LDOS calculations finished successfully."""
        error_codes = []

        calculation = self.ctx.calc_dos
        if not calculation.is_finished_ok:
            self.report(
                f"DosCalculation failed with exit status {calculation.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_DOS
            )

        calculation = self.ctx.calc_projwfc
        if not calculation.is_finished_ok:
            self.report(
                f"ProjwfcCalculation failed with exit status {calculation.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC
            )

        calculation = self.ctx.calc_ldos
        if not calculation.is_finished_ok:
            self.report(
                f"LDOS ProjwfcCalculation failed with exit status {calculation.exit_status}"
            )
            error_codes.append(
                self.exit_codes.ERROR_SUB_PROCESS_FAILED_LDOS
            )

        if len(error_codes) == 3:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL
        if len(error_codes) == 2:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_TWO
        if len(error_codes) == 1:
            return error_codes[0]

    # pylint: enable=inconsistent-return-statements

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report("workchain successfully completed")

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_nscf,
                PwBaseWorkChain,
                namespace="nscf",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_dos, DosCalculation, namespace="dos"
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_projwfc,
                ProjwfcCalculation,
                namespace="projwfc",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_ldos, ProjwfcCalculation, namespace="ldos"
            )
        )
