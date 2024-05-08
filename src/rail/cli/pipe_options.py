import enum
from functools import partial
from typing import Any, Type, TypeVar

import click

from rail.cli.options import (
    EnumChoice,
    PartialOption,
    PartialArgument,
)


__all__ = [
    "RunMode",
    "config_path",
    "input_dir",
    "maglim",
    "model_dir",
    "model_name",
    "model_path",
    "pdf_dir",
    "run_mode",
    "train_dir",
    "train_file",
]


class RunMode(enum.Enum):
    """Choose the run mode"""

    dry_run = 0
    bash = 1
    slurm = 2


config_path = PartialOption(
    "--config_path",
    help="Path to configuration file",
    type=click.Path(),
)

input_dir = PartialOption(
    "--input_dir",
    help="Input Directory",
    type=click.Path(),
)


maglim = PartialOption(
    "--maglim",
    help="Magnitude limit",
    type=float,
    default=25.5,
)


model_dir = PartialOption(
    "--model_dir",
    help="Path to directory with model files",
    type=click.Path(),
)


model_path = PartialOption(
    "--model_path",
    help="Path to model file",
    type=click.Path(),
)


model_name = PartialOption(
    "--model_name",
    help="Model Name",
    type=str,
)


pdf_dir = PartialOption(
    "--pdf_dir",
    help="Path to directory with p(z) files",
    type=click.Path(),
)


run_mode = PartialOption(
    "--run_mode",
    type=EnumChoice(RunMode),
    default="dry_run",
    help="Mode to run script",
)


size = PartialOption(
    "--size",
    type=int,
    default=100_000,
    help="Number of objects in file",
)


train_dir = PartialOption(
    "--train_dir",
    type=click.Path(),
    help="Path to directory for training file for ML algorithms",
)


train_file = PartialOption(
    "--train_file",
    type=click.Path(),
    help="Training file for ML algorithms",
)
