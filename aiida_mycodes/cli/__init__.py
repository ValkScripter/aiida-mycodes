"""
Command line interface (cli) for aiida_mycodes.

Register new commands either via the "scripts" entry point or plug them
directly into the 'verdi' command by using AiiDA-specific entry points like
"aiida.cmdline.data" (both in the setup.json file).
"""
# pylint: disable=wrong-import-position

import click

from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options, types


@click.group(
    "aiida-mycodes",
    cls=VerdiCommandGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@options.PROFILE(type=types.ProfileParamType(load_profile=True), expose_value=False)
def cmd_root():
    """CLI for the `aiida-mycodes` plugin."""


from .workflows import cmd_workflow
