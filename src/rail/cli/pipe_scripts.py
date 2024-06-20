import copy
import os
import sys
import glob
import subprocess
from pathlib import Path
import pprint
import time
import functools
import itertools

import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import yaml

from rail.utils import name_utils, catalog_utils
from rail.core.stage import RailPipeline
from .pipe_options import RunMode
from .project import RailProject


def handle_command(run_mode, command_line):
    print("subprocess:", *command_line)
    _start_time = time.time()
    print(">>>>>>>>")
    if run_mode == RunMode.dry_run:
        # print(command_line)
        command_line.insert(0, "echo")
        finished = subprocess.run(command_line)
    elif run_mode == RunMode.bash:
        # return os.system(command_line)
        finished = subprocess.run(command_line)
    elif run_mode == RunMode.slurm:        
        raise RuntimeError("handle_command should not be called with run_mode == RunMode.slurm")

    returncode = finished.returncode
    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    print("<<<<<<<<")
    print(f"subprocess completed with status {returncode} in {_elapsed_time} seconds\n")
    return returncode


def handle_commands(run_mode, command_lines, script_path=None):
    
    if run_mode in [RunMode.dry_run, RunMode.bash]:
        for command_ in command_lines:
            retcode = handle_command(run_mode, command_)
            if retcode:
                return retcode
        return 0
    # At this point we are using slurm and need a script to send to batch
    if script_path is None:
        raise ValueError(
            "handle_commands with run_mode == RunMode.slurm requires a path to a script to write",
        )
    with open(script_path, 'w') as fout:
        fout.write("#!/usr/bin/bash\n\n")
        for command_ in command_lines:
            com_line = ' '.join(command_)
            fout.write(f"{com_line}\n")

    script_log = script_path.replace('.sh', '.log')
    try:
        with subprocess.Popen(
                ["sbatch", "-o", script_log, "--mem", "16448", "-p", "milano", "--parsable", script_path],
                stdout=subprocess.PIPE,
        ) as sbatch:
            assert sbatch.stdout
            line = sbatch.stdout.read().decode().strip()
            return line.split("|")[0]
    except TypeError as msg:
        raise TypeError(f"Bad slurm submit: {msg}") from msg


def inspect(config_file):
    project = RailProject.load_config(config_file)
    printable_config = pprint.pformat(project.config, compact=True)
    print(f"RAIL Project: {project}")
    print(f">>>>>>>>")
    print(printable_config)
    print("<<<<<<<<")
    return 0


def truth_to_observed_pipeline(
    config_file,
    run_mode=RunMode.bash,
):
    project = RailProject.load_config(config_file)
    pipeline_name = "truth_to_observed"
    pipeline_info = project.get_pipeline(pipeline_name)
    pipeline_path = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor="baseline")
    catalog_tag = pipeline_info['CatalogTag']
    input_catalog_name = pipeline_info['InputCatalog']
    input_catalog = project.get_catalogs().get(input_catalog_name)
    
    # Loop through all possible combinations of the iteration variables that are
    # relevant to this pipeline
    if (iteration_vars := input_catalog.get("IterationVars")) is not None:
        iterations = itertools.product(
            *[
                project.config.get("IterationVars").get(iteration_var)
                for iteration_var in iteration_vars
            ]
        )
        for iteration_args in iterations:
            iteration_kwargs = {
                iteration_vars[i]: iteration_args[i]
                for i in range(len(iteration_vars))
            }

            source_catalog = project.get_catalog('reduced', flavor='baseline', **iteration_kwargs)
            sink_catalog = project.get_catalog('degraded', flavor='baseline', **iteration_kwargs)
            sink_dir = os.path.dirname(sink_catalog)
            script_path = os.path.join(sink_dir, f"submit_{pipeline_name}.sh")
            
            if not os.path.isfile(source_catalog):
                raise ValueError(f"Input file {source_catalog} not found")
            try:
                handle_commands(
                    run_mode,
                    [
                        ["mkdir", "-p", f"{sink_dir}"],
                        ["ceci", f"{pipeline_path}", f"inputs.input={source_catalog}", f"output_dir={sink_dir}", f"log_dir={sink_dir}"],
                        ["tables-io", "convert", "--input", f"{sink_dir}/output_dereddener_errors.pq", "--output", f"{sink_dir}/output.hdf5"]
                    ],
                    script_path,
                )
            except Exception as msg:
                print(msg)
                return 1
        return 0

    # FIXME need to get catalogs even if iteration not specified; this return fallback isn't ideal
    return 1


def inform_pipeline(
    config_file,
    selection="maglim_25.5",
    flavor="baseline",
    label="train_file",    
    run_mode=RunMode.bash,
):

    project = RailProject.load_config(config_file)
    pipeline_name = "inform"
    pipeline_info = project.get_pipeline(pipeline_name)
    pipeline_path = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor=flavor)
    pipeline_config = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor=flavor).replace('.yaml', '_config.yml')
    catalog_tag = pipeline_info['CatalogTag']
    input_file_alias = pipeline_info['InputFileAlias']
    input_file = project.get_file_for_flavor('baseline', label, selection=selection)

    # FIXME where is this specified in the project config?
    # we should provide an interface instead of just grabbing from the internal config dict...
    sink_dir = project.get_path_template('ceci_output_dir', selection=selection, flavor=flavor)
    script_path = os.path.join(sink_dir, f"submit_{pipeline_name}.sh")

    command_line = [
        f"ceci",
        f"{pipeline_path}",
        f"config={pipeline_config}",
        f"inputs.input={input_file}",
        f"output_dir={sink_dir}",
        f"log_dir={sink_dir}/logs",
    ]
    try:
        handle_commands(run_mode, [command_line], script_path)
    except Exception as msg:
        print(msg)
        return 1
    return 0


def estimate_single(
    config_file,
    selection="maglim_25.5",
    flavor="baseline",
    label="test_file",    
    run_mode=RunMode.bash,
):

    project = RailProject.load_config(config_file)
    pipeline_name = "estimate"
    pipeline_info = project.get_pipeline(pipeline_name)
    pipeline_path = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor=flavor)
    pipeline_config = pipeline_path.replace('.yaml', '_config.yml')
    catalog_tag = pipeline_info['CatalogTag']
    input_file_alias = pipeline_info['InputFileAlias']
    input_file = project.get_file_for_flavor('baseline', label, selection=selection)

    sink_dir = project.get_path_template('ceci_output_dir', selection=selection, flavor=flavor)
    script_path = os.path.join(sink_dir, f"submit_{pipeline_name}.sh")

    pz_algorithms = project.get_pzalgorithms()
    model_overrides = [
        f"inputs.model_{pz_algo_}={sink_dir}/model_inform_{pz_algo_}.pkl" for pz_algo_ in pz_algorithms.keys()]

    command_line = [
        f'ceci',
        f"{pipeline_path}",
        f"config={pipeline_config}",
        f'inputs.input={input_file}',
        f'output_dir={sink_dir}',
        f'log_dir={sink_dir}/logs',
    ]
    command_line += model_overrides
    try:
        handle_commands(run_mode, [command_line], script_path)
    except Exception as msg:
        print(msg)
        return 1
    return 0


def estimate_all(
    input_dir,
    input_file,
    config_path,
    pdf_dir,
    model_dir,
    run_mode=RunMode.bash,
):
    config_name = Path(config_path).stem
    config_dir = Path(config_path).parent
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config_file = config_dict["config"]

    model_path = Path(model_dir)
    model_commands = []
    for model_file in model_path.glob("*"):
        model_name = model_file.stem.lower()
        model_commands.append(
            f'inputs.{model_name}={model_file}'
        )

    if input_file:
        output_dir=f'{pdf_dir}/{config_name}'
        input_path = input_file
        command_line = [
            f'ceci',
            f'{config_path}',
            f'config={config_dir}/{config_file}',
            f'inputs.input={input_path}',
            f'inputs.spec_input={input_path}',
            f'output_dir={output_dir}',
            f'log_dir={output_dir}',
        ]
        command_line += model_commands
        try:
            handle_command(run_mode, command_line)
        except Exception as msg:
            print(msg)
            return 1

    input_dirs = glob.glob(f'{input_dir}/*')
    for input_dir_ in input_dirs:
        healpixel = os.path.basename(input_dir_)
        output_dir=f'{pdf_dir}/{config_name}/{healpixel}'
        input_path = f'{input_dir_}/output.hdf5'
        command_line = [
            f'ceci',
            f'{config_path}',
            f'config={config_dir}/{config_file}',
            f'inputs.input={input_path}',
            f'inputs.spec_input={input_path}',
            f'output_dir={output_dir}',
            f'log_dir={output_dir}',
        ]
        command_line += model_commands
        try:
            handle_command(run_mode, command_line)
        except Exception as msg:
            print(msg)
            return 1
    return 0


def evaluate_single(
    config_file,
    selection="maglim_25.5",
    flavor="baseline",
    label="test_file",    
    run_mode=RunMode.bash,
):
    project = RailProject.load_config(config_file)
    pipeline_name = "evaluate"
    pipeline_info = project.get_pipeline(pipeline_name)
    pipeline_path = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor=flavor)
    pipeline_config = pipeline_path.replace('.yaml', '_config.yml')
    catalog_tag = pipeline_info['CatalogTag']
    input_file_alias = pipeline_info['InputFileAlias']
    input_file = project.get_file_for_flavor('baseline', label, selection=selection)

    sink_dir = project.get_path_template('ceci_output_dir', selection=selection, flavor=flavor)
    script_path = os.path.join(sink_dir, f"submit_{pipeline_name}.sh")

    pz_algorithms = project.get_pzalgorithms()
    model_overrides = [
        f"inputs.input_evaluate_{pz_algo_}={sink_dir}/output_estimate_{pz_algo_}.hdf5" for pz_algo_ in pz_algorithms.keys()]

    command_line = [
        f'ceci',
        f"{pipeline_path}",
        f"config={pipeline_config}",
        f'inputs.truth={input_file}',
        f'output_dir={sink_dir}',
        f'log_dir={sink_dir}/logs',
    ]
    command_line += model_overrides
    try:
        handle_commands(run_mode, [command_line], script_path)
    except Exception as msg:
        print(msg)
        return 1
    return 0


def subsample_data(
    config_file,
    source_tag="degraded",
    selection="maglim_25.5",
    flavor="baseline",
    label="train_file",
    run_mode=RunMode.bash,
):

    project = RailProject.load_config(config_file)
    hdf5_output = project.get_file_for_flavor(flavor, label, selection=selection)
    output = hdf5_output.replace('.hdf5', '.parquet')
    output_metadata = project.get_file_metadata_for_flavor(flavor, label)
    output_dir = os.path.dirname(output)
    size = output_metadata.get("NumObjects")
    seed = output_metadata.get("Seed")
    catalog_metadata = project.get_catalogs()['degraded']
    iteration_vars = catalog_metadata['IterationVars']

    iterations = itertools.product(
        *[
            project.config.get("IterationVars").get(iteration_var)
            for iteration_var in iteration_vars
        ]
    )
    sources = []
    for iteration_args in iterations:
        iteration_kwargs = {
            iteration_vars[i]: iteration_args[i]
            for i in range(len(iteration_vars))
        }

        source_catalog = project.get_catalog(source_tag, selection=selection, flavor=flavor, **iteration_kwargs)    
        sources.append(source_catalog.replace('output.hdf5','output_dereddener_errors.pq'))

    if run_mode == RunMode.slurm:
        raise NotImplementedError("subsample_data not set up to run under slurm")

    dataset = ds.dataset(sources)
    num_rows = dataset.count_rows()
    print("num rows", num_rows)
    rng = np.random.default_rng(seed)
    print("sampling", size)
    indices = rng.choice(num_rows, size=size, replace=False)
    subset = dataset.take(indices)
    print("writing", output)

    if run_mode == RunMode.bash:    
        os.makedirs(output_dir, exist_ok=True)    
        pq.write_table(
            subset,
            output,
        )
    print("done")
    handle_command(run_mode, ["tables-io", "convert", "--input", f"{output}", "--output", f"{hdf5_output}"])
    return 0



def build_pipelines(config_file, catalog_tag=None, **kwargs):

    project = RailProject.load_config(config_file)
    output_dir = project.get_common_path('project_scratch_dir')
    namer = name_utils.NameFactory.build_from_yaml(config_file, relative=True)
        
    for pipeline_name, pipeline_info in project.get_pipelines().items():

        output_yaml = project.get_path_template('pipeline_path', pipeline=pipeline_name, flavor='baseline')
        pipe_out_dir = os.path.dirname(output_yaml)

        try:
            os.makedirs(pipe_out_dir)
        except:
            pass
        
        pipeline_class = pipeline_info['PipelineClass']
        catalog_tag = pipeline_info['CatalogTag']

        if catalog_tag:
            catalog_utils.apply_defaults(catalog_tag)
        
        tokens = pipeline_class.split('.')
        module = '.'.join(tokens[:-1])
        class_name = tokens[-1]
        log_dir = f"{output_dir}/logs/{pipeline_name}"
        
        __import__(module)        
        RailPipeline.build_and_write(class_name, output_yaml, namer, kwargs, output_dir, log_dir)

