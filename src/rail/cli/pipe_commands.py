from typing import Any

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
def inspect(config_file: str) -> int:
    """Inspect a rail pipeline project config"""
    return pipe_scripts.inspect(config_file)


@pipe_cli.command()
@pipe_options.config_file()
@pipe_options.flavor()
def build_pipelines(config_file: str, **kwargs: Any) -> int:
    """Build the ceci pipeline configuraiton files"""
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
def reduce_roman_rubin(config_file: str, **kwargs: Any) -> int:
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
def truth_to_observed_pipeline(config_file: str, **kwargs: Any) -> int:
    """Run the truth-to-observed analysis pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    pipeline_name = "truth_to_observed"

    pipeline_catalog_config = pipe_scripts.TruthToObservedPipelineCatalogConfiguration(
        project, source_catalog_tag='reduced', sink_catalog_tag='degraded',
    )
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_catalog(
            project, pipeline_name,
            pipeline_catalog_config,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="spec-selection")
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.flavor()
@pipe_options.run_mode()
def spectroscopic_selection_pipeline(config_file: str, **kwargs: Any) -> int:
    """Run the spectroscopic selection analysis pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    pipeline_name = "spec_selection"
    pipeline_info = project.get_pipeline(pipeline_name)
    input_catalog_name = pipeline_info['InputCatalogTag']

    pipeline_catalog_config = pipe_scripts.SpectroscopicPipelineCatalogConfiguration(
        project,
        source_catalog_tag=input_catalog_name,
        sink_catalog_tag='degraded',
        source_catalog_basename="output_dereddener_errors.pq",
    )

    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_catalog(
            project, pipeline_name,
            pipeline_catalog_config,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="blending")
@pipe_options.config_file()
@pipe_options.selection()
@pipe_options.flavor()
@pipe_options.run_mode()
def blending_pipeline(config_file: str, **kwargs: Any) -> int:
    """Run the blending analysis pipeline"""
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    pipeline_name = "blending"
    pipeline_info = project.get_pipeline(pipeline_name)
    input_catalog_name = pipeline_info['InputCatalogTag']
    pipeline_catalog_config = pipe_scripts.BlendingPipelineCatalogConfiguration(
        project,
        source_catalog_tag=input_catalog_name,
        sink_catalog_tag='degraded',
    )

    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_catalog(
            project, pipeline_name,
            pipeline_catalog_config,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="subsample")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.label()
@pipe_options.run_mode()
def subsample_data(config_file: str, **kwargs: Any) -> int:
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
def inform(config_file: str, **kwargs: Any) -> int:
    """Run the inform pipeline"""
    pipeline_name = "inform"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.inform_input_callback,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="estimate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def estimate_single(config_file: str, **kwargs: Any) -> int:
    """Run the estimation pipeline"""
    pipeline_name = "estimate"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.estimate_input_callback,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="evaluate")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def evaluate_single(config_file: str, **kwargs: Any) -> int:
    """Run the evaluation pipeline"""
    pipeline_name = "evaluate"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.evaluate_input_callback,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="pz")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def pz_single(config_file: str, **kwargs: Any) -> int:
    """Run the pz pipeline"""
    pipeline_name = "pz"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.pz_input_callback,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="tomography")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def tomography_single(config_file : str, **kwargs: Any) -> int:
    """Run the tomography pipeline"""
    pipeline_name = "tomography"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.tomography_input_callback,
            **kw, **kwargs,
        )
    return ok


@pipe_cli.command(name="sompz")
@pipe_options.config_file()
@pipe_options.flavor()
@pipe_options.selection()
@pipe_options.run_mode()
def sompz_single(config_file: str, **kwargs: Any) -> int:
    """Run the sompz pipeline"""
    pipeline_name = "sompz"
    project = RailProject.load_config(config_file)
    flavors = project.get_flavor_args(kwargs.pop('flavor'))
    selections = project.get_selection_args(kwargs.pop('selection'))
    iter_kwargs = project.generate_kwargs_iterable(flavor=flavors, selection=selections)
    ok = 0
    for kw in iter_kwargs:
        ok |= pipe_scripts.run_pipeline_on_single_input(
            project, pipeline_name,
            pipe_scripts.sompz_input_callback,
            **kw, **kwargs,
        )
    return ok
