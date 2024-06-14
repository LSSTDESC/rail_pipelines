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

@pipe_cli.command(name="truth-to-observed")
@pipe_options.config_file()
@pipe_options.input_dir()
@pipe_options.config_path()
@options.outdir()
@pipe_options.run_mode()
def truth_to_observed_pipeline(config_file, input_dir, config_path, outdir, run_mode):
    """Run the truth-to-observed data pipeline"""
    return pipe_scripts.truth_to_observed_pipeline(config_file, input_dir, config_path, outdir, run_mode)


@pipe_cli.command(name="inform")
@pipe_options.input_file()
@pipe_options.config_path()
@pipe_options.model_dir()
@pipe_options.run_mode()
def inform_single(input_file, config_path, model_dir, run_mode):
    """Inform the model for a single algorithm"""
    return pipe_scripts.inform_single(input_file, config_path, model_dir, run_mode)


@pipe_cli.command(name="estimate")
@pipe_options.input_file()
@pipe_options.config_path()
@pipe_options.pdf_dir()
@pipe_options.model_name()
@pipe_options.model_path()
@pipe_options.run_mode()
def estimate_single(input_file, config_path, pdf_dir, model_name, model_path, run_mode):
    """Run the estimation stage for a single algorithm"""
    return pipe_scripts.estimate_single(input_file, config_path, pdf_dir, model_name, model_path, run_mode)


@pipe_cli.command(name="estimate-all")
@pipe_options.input_dir()
@pipe_options.input_file()
@pipe_options.config_path()
@pipe_options.pdf_dir()
@pipe_options.model_dir()
@pipe_options.run_mode()
def estimate_all(input_dir, input_file, config_path, pdf_dir, model_dir, run_mode):
    """Run the estimation stage for all algorithms"""
    return pipe_scripts.estimate_all(input_dir, input_file, config_path, pdf_dir, model_dir, run_mode)


@pipe_cli.command(name="evaluate")
@pipe_options.config_path()
@pipe_options.pdf_path()
@pipe_options.truth_path()
@pipe_options.output_dir()
@pipe_options.run_mode()
def evaluate_single(config_path, pdf_path, truth_path, output_dir, run_mode):
    """Run the estimation stage for a single algorithm"""
    return pipe_scripts.evaluate_single(config_path, pdf_path, truth_path, output_dir, run_mode)


@pipe_cli.command(name="make-training-data")
@pipe_options.input_dir()
@pipe_options.output_dir()
@pipe_options.output_file()
@pipe_options.size()
@pipe_options.seed()
def subsample_data(input_dir, output_dir, output_file, size, seed):
    """Make a training data set by randomly selecting objects"""
    return pipe_scripts.subsample_data(input_dir, output_dir, output_file, size, seed=seed, label="train")


@pipe_cli.command(name="make-testing-data")
@pipe_options.input_dir()
@pipe_options.output_dir()
@pipe_options.output_file()
@pipe_options.size()
@pipe_options.seed()
def subsample_data(input_dir, output_dir, output_file, size, seed):
    """Make a testing data set by randomly selecting objects"""
    return pipe_scripts.subsample_data(input_dir, output_dir, output_file, size, seed=seed, label="test")


# NOTE brief testing shows that using all of the data isn't necessarily better
# than using a subset. This is quite expensive, so commenting out for now.
# @pipe_cli.command()
# @pipe_options.input_dir()
# @pipe_options.output_dir()
# @pipe_options.output_file()
# def make_som_data(input_dir, output_dir, output_file):
#     """Make a training data set for a som from all objects"""
#     return pipe_scripts.make_som_data(input_dir, output_dir, output_file)


@pipe_cli.command()
@pipe_options.input_dir()
@pipe_options.maglim()
def reduce_roman_rubin(input_dir, maglim):
    """Reduce the roman rubin simulations for PZ analysis"""
    reduce_roman_rubin_data(input_dir, maglim)


