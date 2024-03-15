"""My Workchain to run a Quantum ESPRESSO pw.x calculation with:
- native automated error handling and restarts.
- custom protocol files in subdirectory `my_protocols`"""

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from importlib_resources import files

from aiida.plugins import CalculationFactory, GroupFactory

from ..protocols import pw as pw_protocols

PwCalculation = CalculationFactory("quantumespresso.pw")
SsspFamily = GroupFactory("pseudo.family.sssp")
PseudoDojoFamily = GroupFactory("pseudo.family.pseudo_dojo")
CutoffsPseudoPotentialFamily = GroupFactory("pseudo.family.cutoffs")


class MyPwBaseWorkChain(PwBaseWorkChain):
    """
    My PwBaseWorkChain calling my own protocol
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        return files(pw_protocols) / "base.yaml"

    @classmethod
    def get_builder_from_protocol(cls, *args, **kwargs):
        builder = super().get_builder_from_protocol(*args, **kwargs)
        return builder


if __name__ == "__main__":
    print("hi")
    print(PwBaseWorkChain.get_protocol_filepath())
    print(MyPwBaseWorkChain.get_protocol_filepath())
