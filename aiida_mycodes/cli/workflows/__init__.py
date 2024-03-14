"""
Module with CLI commands for the various work chain implementations.
"""
# pylint: disable=cyclic-import,reimported,unused-import,wrong-import-position

from .. import cmd_root


@cmd_root.group("workflow")
def cmd_workflow():
    """Commands to launch and interact with workflows."""


@cmd_workflow.group("launch")
def cmd_launch():
    """Launch workflows."""


# Import the sub commands to register them with the CLI
from .pw.base import launch_workflow

# from aiida_quantumespresso.cli.workflows.pw.base import launch_workflow
