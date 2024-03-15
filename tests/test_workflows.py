""" Tests for workflows."""

from ase.build import bulk
from importlib_resources import files

from aiida.orm import StructureData

from aiida_mycodes.workflows.protocols import pw as pw_protocols
from aiida_mycodes.workflows.pw.base import MyPwBaseWorkChain

# from aiida_pseudo.cli.install import cmd_install_sssp


# pylint: disable=attribute-defined-outside-init
class TestMyPwBaseWorkChain:
    """
    Test the `MyPwBaseWorkChain` workflow.
    """

    def setup_method(self):
        """Prepare nodes for cli tests."""
        # cmd_install_sssp()
        self.structure = StructureData(ase=bulk("Si"))

    def test_builder(self):
        """Test the builder of the `MyPwBaseWorkChain` workflow."""
        builder = MyPwBaseWorkChain.get_builder()
        builder["pw"]["structure"] = self.structure
        assert builder["pw"]["structure"].get_formula() == "Si2"

    def test_protocol_filepath(self):
        """Test the protocol filepath of the `MyPwBaseWorkChain` workflow."""
        filepath = MyPwBaseWorkChain.get_protocol_filepath()
        assert filepath == files(pw_protocols) / "my_pw_base_protocol.yaml"
