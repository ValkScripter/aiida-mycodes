"""
Command line scripts to launch a `MyPwBaseWorkChain`.
"""
from aiida_quantumespresso.cli.utils import defaults, launch, options

from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
from aiida.plugins import WorkflowFactory

from .. import cmd_launch

# import click


@cmd_launch.command("pw-base")
@options_core.CODE(
    required=True, type=types.CodeParamType(entry_point="quantumespresso.pw")
)
@options.STRUCTURE(default=defaults.get_structure)
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(code, structure, daemon):
    """Run a `MyPwBaseWorkChain`."""

    builder = WorkflowFactory("mycodes.pw.base").get_builder_from_protocol(
        code, structure, protocol="moderate"
    )

    launch.launch_process(builder, daemon)
