"""My Workchain to run a Quantum ESPRESSO pw.x calculation with:
- native automated error handling and restarts.
- custom protocol files in subdirectory `my_protocols`"""

from aiida_quantumespresso.workflows.pdos import PdosWorkChain
from importlib_resources import files

from aiida.plugins import CalculationFactory, GroupFactory

if __name__ == "__main__":
    from aiida_mycodes.workflows import protocols
else:
    from . import protocols

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
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        return files(protocols) / "pdos.yaml"

    @classmethod
    def get_builder_from_protocol(cls, *args, **kwargs):
        builder = super().get_builder_from_protocol(*args, **kwargs)
        return builder


if __name__ == "__main__":
    print("hi")
    print(PdosWorkChain.get_protocol_filepath())
    print(MyPdosWorkChain.get_protocol_filepath())
