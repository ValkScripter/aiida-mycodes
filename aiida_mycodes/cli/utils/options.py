"""
My own pre-defined overridable options for command line interface parameters.
"""
import click

from aiida.cmdline.params.options import OverridableOption

PROTOCOL = OverridableOption(
    "-p",
    "--protocol",
    type=click.STRING,
    required=False,
    default="moderate",
    show_default=True,
    help="Protocol name to use for the workflow builder",
)

PPN = OverridableOption(
    "--ppn",
    type=click.INT,
    required=False,
    default=1,
    show_default=True,
    help="Number of cores of the remote computer to run the job on",
)

NPOOL = OverridableOption(
    "--npool",
    type=click.INT,
    required=False,
    default=None,
    show_default=False,
    help="Number of pools for the QE job",
)
