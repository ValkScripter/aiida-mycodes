"""
WorkChain to compute unfolded projected bands.
To compute unfolded projected bands, one needs:
- a primitive / base structure
- a supercell structure under study, which is a supercell of the base structure
- a k-point list in the primitive cell, where the unfolded bands will be computed.
The workchain will then:
1) run the unklist calculation to generate the k-point list in the supercell corresponding to
   the input k-point list in the primitive cell (discarding redundant k-points due to folding);
2) run the projwfcbands workchain on the supercell structure using the generated k-point list to get
   the folded band structure, with wavefunction projections;
3) run the unfold calculation to compute the unfolded total band structure (with no projection);
4) run the unfold calculation for each wavefunction projection to get the unfolded projected band structures.
The user will them sum on his own the unfolded projected band structures to get the desired projected
band structure.
"""

from ase import Atoms

from aiida import orm
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida.tools.data.orbital import Orbital

from aiida_mycodes.calculations.unfold import UnfoldCalculation
from aiida_mycodes.calculations.unklist import UnklistCalculation
from aiida_mycodes.workflows.projwfcbands import MyProjwfcBandsWorkChain


class UnfoldedProjBandsWorkChain(WorkChain):
    """WorkChain to compute unfolded projected bands."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
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
            help="The supercell structure under study (a supercell of the base structure).",
        )
        spec.input(
            "bands_kpoints",
            valid_type=orm.KpointsData,
            required=True,
            help="The k-point list in the primitive cell, where the unfolded bands will be computed.",
        )
        spec.expose_inputs(
            process_class=UnklistCalculation,
            namespace="unklist",
            include=[
                "code",
                "metadata",
            ],
            namespace_options={
                "help": "Inputs for the unklist calculation to generate the k-point list in the supercell.",
            },
        )
        spec.expose_inputs(
            process_class=MyProjwfcBandsWorkChain,
            namespace="projwfcbands",
            exclude=[
                "clean_workdir",
                "structure",
                "bands_kpoints",
                "bands_kpoints_distance",
            ],
            namespace_options={
                "help": "Inputs for the MyProjwfcBands workchain to compute the folded band structure with wavefunction projections.",
            },
        )
        spec.expose_inputs(
            process_class=UnfoldCalculation,
            namespace="unfold",
            include=[
                "code",
                "broadening",
                "metadata",
            ],
            namespace_options={
                "help": "Inputs for the unfold calculation to compute the unfolded band structures.",
            },
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Whether to clean the work directories of the calculations spawned by this workchain.",
        )

        # Output ports
        spec.expose_outputs(
            process_class=MyProjwfcBandsWorkChain,
            namespace="projwfcbands",
            namespace_options={
                "help": "Outputs from the MyProjwfcBands workchain.",
            },
        )
        spec.output(
            "unfolded_bands_contourf",
            valid_type=orm.ArrayData,
            help="The unfolded total band structure (contourf format).",
        )
        spec.output(
            "unfolded_bands_scatter",
            valid_type=orm.ArrayData,
            help="The unfolded total band structure (scatter format).",
        )
        spec.output(
            "unfolded_projected_bands_contourf",
            valid_type=orm.ArrayData,
            help="The unfolded projected band structures (contourf format).",
        )
        spec.output(
            "unfolded_projected_bands_scatter",
            valid_type=orm.ArrayData,
            help="The unfolded projected band structures (scatter format).",
        )

        # WorkChain outline
        spec.outline(  # pyright: ignore[reportAttributeAccessIssue]
            cls.setup,
            cls.run_unklist,
            cls.inspect_unklist,
            cls.run_projwfcbands,
            cls.inspect_projwfcbands,
            cls.run_unfold_total,
            cls.inspect_unfold_total,
            cls.run_unfold_projections,
            cls.inspect_unfold_projections,
        )

        # Exit codes
        spec.exit_code(
            401,
            "ERROR_UNKLIST_FAILED",
            message="The UnklistCalculation to generate the k-point list in the supercell failed.",
        )
        spec.exit_code(
            402,
            "ERROR_PROJWFCBANDS_FAILED",
            message="The MyProjwfcBandsWorkChain to compute the folded band structure failed.",
        )
        spec.exit_code(
            403,
            "ERROR_UNFOLD_TOTAL_FAILED",
            message="The UnfoldCalculation to compute the unfolded total band structure failed.",
        )
        spec.exit_code(
            404,
            "ERROR_UNFOLD_PROJECTIONS_FAILED",
            message="One of the UnfoldCalculation to compute the unfolded projected band structures failed.",
        )

    def setup(self):
        """Setup the workchain."""
        assert isinstance(self.inputs.base_structure, orm.StructureData)
        self.ctx.base_structure = self.inputs.base_structure
        assert isinstance(
            self.inputs.supercell_structure, orm.StructureData
        )
        self.ctx.supercell_structure = self.inputs.supercell_structure
        assert isinstance(self.inputs.bands_kpoints, orm.KpointsData)
        self.ctx.bands_kpoints = self.inputs.bands_kpoints
        self.ctx.folded_bands_kpoints = None
        self.ctx.projections = None

    def run_unklist(self):
        """Run the unklist calculation to generate the k-point list in the supercell."""
        unklist_inputs = self.exposed_inputs(
            UnklistCalculation,
            namespace="unklist",
        )
        unklist_inputs["base_structure"] = self.ctx.base_structure
        unklist_inputs["supercell_structure"] = (
            self.ctx.supercell_structure
        )
        unklist_inputs["kpoints"] = self.ctx.bands_kpoints

        self.report(
            "Submitting UnklistCalculation to generate k-point list in supercell."
        )

        return ToContext(
            unklist_calc=self.submit(
                UnklistCalculation, **unklist_inputs
            )
        )

    def inspect_unklist(self):
        """Inspect the output of the unklist calculation."""
        unklist_calc = self.ctx.unklist_calc
        assert isinstance(unklist_calc, orm.CalcJobNode)

        if not unklist_calc.is_finished_ok:
            self.report(
                f"UnklistCalculation<{unklist_calc.pk}> failed with exit status {unklist_calc.exit_status}"
            )
            return self.exit_codes.ERROR_UNKLIST_FAILED
        else:
            self.report(
                f"UnklistCalculation<{unklist_calc.pk}> completed successfully."
            )

        assert isinstance(
            unklist_calc.outputs.output_kpoints, orm.KpointsData
        )
        self.ctx.folded_bands_kpoints = (
            unklist_calc.outputs.output_kpoints
        )

    def run_projwfcbands(self):
        """Run the MyProjwfcBands workchain on the supercell structure using the generated k-point list."""
        assert self.ctx.folded_bands_kpoints is not None

        projwfcbands_inputs = self.exposed_inputs(
            MyProjwfcBandsWorkChain,
            namespace="projwfcbands",
        )
        projwfcbands_inputs["clean_workdir"] = orm.Bool(False)
        projwfcbands_inputs["structure"] = self.ctx.supercell_structure
        projwfcbands_inputs["bands_kpoints"] = (
            self.ctx.folded_bands_kpoints
        )
        projwfcbands_inputs.pop("bands_kpoints_distance", None)

        self.report(
            "Submitting MyProjwfcBandsWorkChain on supercell structure."
        )

        return ToContext(
            projwfcbands_wc=self.submit(
                MyProjwfcBandsWorkChain, **projwfcbands_inputs
            )
        )

    def inspect_projwfcbands(self):
        """Inspect the output of the MyProjwfcBands workchain."""
        projwfcbands_wc = self.ctx.projwfcbands_wc
        assert isinstance(projwfcbands_wc, orm.WorkChainNode)

        if not projwfcbands_wc.is_finished_ok:
            self.report(
                f"MyProjwfcBandsWorkChain<{projwfcbands_wc.pk}> failed with exit status {projwfcbands_wc.exit_status}"
            )
            return self.exit_codes.ERROR_PROJWFCBANDS_FAILED
        else:
            self.report(
                f"MyProjwfcBandsWorkChain<{projwfcbands_wc.pk}> completed successfully."
            )

        # Expose outputs from projwfcbands workchain
        self.out_many(
            self.exposed_outputs(
                node=projwfcbands_wc,
                process_class=MyProjwfcBandsWorkChain,
                namespace="projwfcbands",
            )
        )

        self.ctx.projections = (
            self.ctx.projwfcbands_wc.outputs.projwfc.projections
        )
        assert isinstance(self.ctx.projections, orm.ProjectionData)

    def run_unfold_total(self):
        """Run the unfold calculation to compute the unfolded total band structure."""
        unfold_inputs = self.exposed_inputs(
            UnfoldCalculation,
            namespace="unfold",
        )
        unfold_inputs["base_structure"] = self.ctx.base_structure
        unfold_inputs["supercell_structure"] = (
            self.ctx.supercell_structure
        )
        unfold_inputs["kpoints"] = self.ctx.bands_kpoints
        unfold_inputs.setdefault(
            "broadening", orm.Float(0.1)
        )  # default broadening
        unfold_inputs["fermi_energy"] = (
            self.ctx.projwfcbands_wc.outputs.scf_parameters.dict.fermi_energy
        )
        unfold_inputs["parent_folder"] = (
            self.ctx.projwfcbands_wc.outputs.bands_remote_folder
        )
        unfold_inputs["energy_min"] = (
            self.ctx.projwfcbands_wc.inputs.Emin
        )
        unfold_inputs["energy_max"] = (
            self.ctx.projwfcbands_wc.inputs.Emax
        )
        projwfc = self.ctx.projwfcbands_wc.inputs.projwfc.projwfc.parameters.get_dict()[
            "PROJWFC"
        ]
        unfold_inputs["energy_step"] = projwfc.get(
            "deltae"
        ) or projwfc.get("DeltaE")

        self.report(
            "Submitting UnfoldCalculation for total band structure."
        )

        return ToContext(
            unfold_total_calc=self.submit(
                UnfoldCalculation, **unfold_inputs
            )
        )

    def inspect_unfold_total(self):
        """Inspect the output of the unfold calculation for the total band structure."""
        unfold_total_calc = self.ctx.unfold_total_calc
        assert isinstance(unfold_total_calc, orm.CalcJobNode)

        if not unfold_total_calc.is_finished_ok:
            self.report(
                f"UnfoldCalculation (total) <{unfold_total_calc.pk}> failed with exit status {unfold_total_calc.exit_status}"
            )
            return self.exit_codes.ERROR_UNFOLD_TOTAL_FAILED
        else:
            self.report(
                f"UnfoldCalculation (total) <{unfold_total_calc.pk}> completed successfully."
            )

        assert isinstance(
            unfold_total_calc.outputs.output_contourf, orm.ArrayData
        )
        assert isinstance(
            unfold_total_calc.outputs.output_scatter, orm.ArrayData
        )
        self.out(
            "unfolded_bands_contourf",
            unfold_total_calc.outputs.output_contourf,
        )
        self.out(
            "unfolded_bands_scatter",
            unfold_total_calc.outputs.output_scatter,
        )

    def run_unfold_projections(self):
        """Run the projcted unfold calculation for each group of orbitals (defined by elements) to compute the unfolded projected band structures."""
        unfold_inputs = self.exposed_inputs(
            UnfoldCalculation,
            namespace="unfold",
        )
        unfold_inputs["base_structure"] = self.ctx.base_structure
        structure = self.ctx.supercell_structure
        assert isinstance(structure, orm.StructureData)
        ase_structure = structure.get_ase()
        assert isinstance(ase_structure, Atoms)

        unfold_inputs["supercell_structure"] = structure
        unfold_inputs["kpoints"] = self.ctx.bands_kpoints
        unfold_inputs.setdefault(
            "broadening", orm.Float(0.1)
        )  # default broadening
        unfold_inputs["fermi_energy"] = (
            self.ctx.projwfcbands_wc.outputs.scf_parameters.dict.fermi_energy
        )
        unfold_inputs["parent_folder"] = (
            self.ctx.projwfcbands_wc.outputs.projwfc_remote_folder
        )
        unfold_inputs["energy_min"] = (
            self.ctx.projwfcbands_wc.inputs.Emin
        )
        unfold_inputs["energy_max"] = (
            self.ctx.projwfcbands_wc.inputs.Emax
        )
        projwfc = self.ctx.projwfcbands_wc.inputs.projwfc.projwfc.parameters.get_dict()[
            "PROJWFC"
        ]
        unfold_inputs["energy_step"] = projwfc.get(
            "deltae"
        ) or projwfc.get("DeltaE")

        projections = self.ctx.projections
        assert isinstance(projections, orm.ProjectionData)

        unique_elements = sorted(
            set(ase_structure.get_chemical_symbols())
        )
        self.ctx.unique_elements = unique_elements
        self.report(f"Unique elements in structure: {unique_elements}")

        all_orbitals: list[Orbital] = projections.get_orbitals()
        element_indices = {element: [] for element in unique_elements}
        for i, orb in enumerate(all_orbitals):
            orb_dict = orb.get_orbital_dict()
            kind_name = orb_dict["kind_name"]
            if kind_name in element_indices:
                element_indices[kind_name].append(i)

        # Log the number of orbitals found for each element
        for element in unique_elements:
            self.report(
                f"Found {len(element_indices[element])} {element} orbitals."
            )

        for num, element in enumerate(unique_elements):
            group = element_indices[element]
            unfold_inputs["proj_orbitals"] = orm.List(
                [i + 1 for i in group]
            )  # 1-based indexing
            inputs = unfold_inputs.copy()
            self.report(
                f"Submitting UnfoldCalculation for projected band structure of element '{element}' ({num+1}/{len(unique_elements)})."
            )
            unfold_calc = self.submit(UnfoldCalculation, **inputs)
            self.to_context(
                unfold_projections_calcs=append_(unfold_calc)
            )

    def inspect_unfold_projections(self):
        """Inspect the outputs of the unfold calculations for the projected band structures."""
        unfold_projections_calcs = self.ctx.unfold_projections_calcs

        unfolded_proj_contourf = orm.ArrayData()
        unfolded_proj_scatter = orm.ArrayData()

        n_groups = len(unfold_projections_calcs)

        for i, (unfold_calc, element) in enumerate(
            zip(unfold_projections_calcs, self.ctx.unique_elements)
        ):
            assert isinstance(unfold_calc, orm.CalcJobNode)

            if not unfold_calc.is_finished_ok:
                self.report(
                    f"UnfoldCalculation (projections) for element '{element}' # {i+1}/{n_groups} <{unfold_calc.pk}> failed with exit status {unfold_calc.exit_status}"
                )
                return self.exit_codes.ERROR_UNFOLD_PROJECTIONS_FAILED
            else:
                self.report(
                    f"UnfoldCalculation (projections) for element '{element}' # {i+1}/{n_groups} <{unfold_calc.pk}> completed successfully."
                )

            assert isinstance(
                unfold_calc.outputs.output_contourf, orm.ArrayData
            )
            contourf = unfold_calc.outputs.output_contourf
            assert isinstance(
                unfold_calc.outputs.output_scatter, orm.ArrayData
            )
            scatter = unfold_calc.outputs.output_scatter
            for name in contourf.get_arraynames():
                array = contourf.get_array(name)
                unfolded_proj_contourf.set_array(
                    f"{name}_element_{element}", array
                )
            for name in scatter.get_arraynames():
                array = scatter.get_array(name)
                unfolded_proj_scatter.set_array(
                    f"{name}_element_{element}", array
                )

        unfolded_proj_contourf.store()
        unfolded_proj_scatter.store()

        self.out(
            "unfolded_projected_bands_contourf",
            unfolded_proj_contourf,
        )
        self.out(
            "unfolded_projected_bands_scatter",
            unfolded_proj_scatter,
        )

    def on_terminated(self):
        """Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )


def clean_calcjob_remote(node):
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        cleaned = True
    except (OSError, KeyError):
        pass
    return cleaned


def clean_workchain_calcs(workchain):
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            if clean_calcjob_remote(called_descendant):
                cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs
