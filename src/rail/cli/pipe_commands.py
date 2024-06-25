import click

from rail.core import __version__

from rail.utils.project import RailProject
from rail.cli import pipe_options, pipe_scripts
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
def build_pipelines(config_file, **kwargs):
    """Reduce the roman rubin simulations for PZ analysis"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.build_pipelines(project, **kw, **kwargs)
    return ok


@pipe_cli.command()
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.run_mode()
def reduce_roman_rubin(config_file, **kwargs):
    """Reduce the roman rubin simulations for PZ analysis"""
    project = RailProject.load_config(config_file)
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= reduce_roman_rubin_data(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="truth-to-observed")
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.flavor()
@pipe_options.run_mode()
def truth_to_observed_pipeline(config_file, **kwargs):
    """Run the truth-to-observed data pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.truth_to_observed_pipeline(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="spec-selection")
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.flavor()
@pipe_options.run_mode()
def spectroscopic_selection_pipeline(config_file, **kwargs):
    """Run the truth-to-observed data pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.spectroscopic_selection_pipeline(project, **kw, **kwargs)
    return ok



@pipe_cli.command(name="subsample")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.label()
@pipe_options.run_mode()
def subsample_data(config_file, **kwargs):
    """Make a training data set by randomly selecting objects"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.subsample_data(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="inform")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def inform(config_file, **kwargs):
    """Run the inform pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.inform_pipeline(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="estimate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def estimate_single(config_file, **kwargs):
    """Run the estimation pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.estimate_single(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="evaluate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def evaluate_single(config_file, **kwargs):
    """Run the evaluation pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.evaluate_single(project, **kw, **kwargs)
    return ok


@pipe_cli.command(name="pz")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def pz_single(config_file, **kwargs):
    """Run the pz pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.pz_single(project, **kw, **kwargs)
    return ok
