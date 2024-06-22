import click

from rail.core import __version__
from rail.cli import options, scripts, pipe_options, pipe_scripts
from rail.cli.reduce_roman_rubin_data import reduce_roman_rubin_data


@click.group()
@click.version_option(__version__)
def pipe_cli() -> None:
    """RAIL pipeline scripts"""


@pipe_cli.command(name="inspect")
@pipe_options.config_file()
def inspect(config_file):
    """Inspect a rail pipeline project config"""
    return pipe_scripts.inspect(config_file)


@pipe_cli.command()
@pipe_options.config_file()
@pipe_options.flavor()
def build_pipelines(config_file, flavor, **kwargs):
    """Reduce the roman rubin simulations for PZ analysis"""
    pipe_scripts.build_pipelines(config_file, flavor, **kwargs)

    
@pipe_cli.command()
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.run_mode()
def reduce_roman_rubin(config_file, **kwargs):
    """Reduce the roman rubin simulations for PZ analysis"""
    return reduce_roman_rubin_data(config_file, **kwargs)


@pipe_cli.command(name="truth-to-observed")
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.flavor()
@pipe_options.run_mode()
def truth_to_observed_pipeline(config_file, **kwargs):
    """Run the truth-to-observed data pipeline"""
    return pipe_scripts.truth_to_observed_pipeline(config_file, **kwargs)


@pipe_cli.command(name="subsample")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.label()
@pipe_options.run_mode()
def subsample_data(config_file, **kwargs):
    """Make a training data set by randomly selecting objects"""
    return pipe_scripts.subsample_data(config_file, **kwargs)


@pipe_cli.command(name="inform")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def inform(config_file, **kwargs):
    """Inform the model for a single algorithm"""
    return pipe_scripts.inform_pipeline(config_file, **kwargs)


@pipe_cli.command(name="estimate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def estimate_single(config_file, **kwargs):
    """Run the estimation stage on a single file"""
    return pipe_scripts.estimate_single(config_file, **kwargs)


@pipe_cli.command(name="evaluate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def evaluate_single(config_file, **kwargs):
    """Run the estimation stage for a single algorithm"""
    return pipe_scripts.evaluate_single(config_file, **kwargs)


