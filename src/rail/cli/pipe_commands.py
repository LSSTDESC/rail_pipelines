import click

from rail.core import __version__
from rail.cli import options, scripts, pipe_options, pipe_scripts
from rail.cli.reduce_roman_rubin_data import reduce_roman_rubin_data


@click.group()
@click.version_option(__version__)
def pipe_cli() -> None:
    """RAIL pipeline scripts"""


@pipe_cli.command(name="truth-to-observed")
@pipe_options.input_dir()
@pipe_options.config_path()
@options.outdir()
@pipe_options.run_mode()
def truth_to_observed_pipeline(input_dir, config_path, outdir, run_mode):
    """Run the truth-to-observed data pipeline"""
    return pipe_scripts.truth_to_observed_pipeline(input_dir, config_path, outdir, run_mode)


@pipe_cli.command(name="inform")
@pipe_options.train_file()
@pipe_options.config_path()
@pipe_options.model_dir()
@pipe_options.run_mode()
def inform_single(train_file, config_path, model_dir, run_mode):
    """Inform the model for a single algorithm"""
    return pipe_scripts.inform_single(train_file, config_path, model_dir, run_mode)


@pipe_cli.command(name="estimate")
@pipe_options.input_dir()
@pipe_options.config_path()
@pipe_options.pdf_dir()
@pipe_options.model_name()
@pipe_options.model_path()
@pipe_options.run_mode()
def estimate_single(input_dir, config_path, pdf_dir, model_name, model_path, run_mode):
    """Run the estimation stage for a single algorithm"""
    return pipe_scripts.estimate_single(input_dir, config_path, pdf_dir, model_name, model_path, run_mode)

@pipe_cli.command(name="estimate-all")
@pipe_options.input_dir()
@pipe_options.config_path()
@pipe_options.pdf_dir()
@pipe_options.model_dir()
@pipe_options.run_mode()
def estimate_all(input_dir, config_path, pdf_dir, model_dir, run_mode):
    """Run the estimation stage for all algorithms"""
    return pipe_scripts.estimate_all(input_dir, config_path, pdf_dir, model_dir, run_mode)


@pipe_cli.command()
@pipe_options.input_dir()
@pipe_options.train_dir()
@pipe_options.train_file()
@pipe_options.size()
def make_training_data(input_dir, train_dir, train_file, size):
    """Make a training data set by randomly selecting objects"""
    return pipe_scripts.make_training_data(input_dir, train_dir, train_file, size)



@pipe_cli.command()
@pipe_options.input_dir()
@pipe_options.maglim()
def reduce_roman_rubin(input_dir, maglim):
    """Reduce the roman rubin simulations for PZ analysis"""
    reduce_roman_rubin_data(input_dir, maglim)


