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

from rail.utils import name_utils
from .pipe_options import RunMode


class RAILProject:
    config_template = {
        "IterationVars": {},
        "CommonPaths": {},
        "PathTemplates": {},
        "InputCatalog": {},
        "ReducedCatalog": {},
        "ObservedCatalog": {},
        "Files": {},
        "Pipelines": {},
        "Flavors": {},
        "Selections": {},
        "PZAlgorithms": {},
        "NZAlgorithms": {},
    }

    def __init__(self, project, config_dict):
        self.project = project
        self._config_dict = config_dict
        self.config = copy.deepcopy(self.config_template)
        for k in self.config.keys():
            if (v := self._config_dict.get(k)) is not None:
                self.config[k] = v
        self.interpolants = self.get_interpolants()
        self.name_factory = name_utils.NameFactory(
            project=self.project,
            config=self.config,
        )

    @staticmethod
    def load_config(config_file):
        project_name = Path(config_file).stem
        with open(config_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        project = RAILProject(project_name, config_dict)
        project.resolve_common()
        return project

    def get_interpolants(self):
        interpolants = {}

        if (common_dict := self.config.get("CommonPaths")) is not None:
            for key, value in common_dict.items():
                new_value = value.format(**interpolants)
                interpolants[key] = new_value

        return interpolants

    def resolve_common(self):
        resolved = {}
        for target in ["CommonPaths"]:
            done = self.name_factory.resolve(
                self.config.get(target),
                self.interpolants,
            )
            resolved[target] = done
            self.config.get(target).update(done)

        return resolved

    def get_partial_templates(self):
        resolved_templates = {}
        path_templates = self.config.get("PathTemplates")
        for k, v in path_templates.items():
            resolved_templates[k] = functools.partial(
                v.format
            )

        return resolved_templates

    def get_flavors(self):
        return self.config.get("Flavors")

    def get_flavor(self, name):
        flavors = self.get_flavors()
        flavor = flavors.get(name, None)
        if flavor is None:
            raise ValueError(f"flavor '{name}' not found in {self.project}")
        return flavor

    def get_selections(self):
        return self.config.get("Selections")

    def get_selection(self, name):
        selections = self.get_selections()
        selection = selections.get(name, None)
        if selection is None:
            raise ValueError(f"selection '{name}' not found in {self.project}")
        return selection

    def get_pzalgorithms(self):
        return self.config.get("PZAlgorithms")

    def get_pzalgorithm(self, name):
        pzalgorithms = self.get_pzalgorithms()
        pzalgorithm = pzalgorithms.get(name, None)
        if pzalgorithm is None:
            raise ValueError(f"pz algorithm '{name}' not found in {self.project}")
        return pzalgorithm

    def get_nzalgorithms(self):
        return self.config.get("NZAlgorithms")

    def get_nzalgorithm(self, name):
        nzalgorithms = self.get_nzalgorithms()
        nzalgorithm = nzalgorithms.get(name, None)
        if nzalgorithm is None:
            raise ValueError(f"nz algorithm '{name}' not found in {self.project}")
        return nzalgorithm

    def _get_catalog(self, catalog, **kwargs):
        if (input_catalog := self.config.get(catalog)) is not None:
            if (path_template := input_catalog.get("PathTemplate")) is not None:
                path = path_template.format(
                    **self.config["CommonPaths"],
                    **kwargs,
                )
        else:
            path = None

        return path

    def get_input_catalog(self, **kwargs):
        return self._get_catalog("InputCatalog", **kwargs)

    def get_reduced_catalog(self, **kwargs):
        return self._get_catalog("ReducedCatalog", **kwargs)

    def get_observed_catalog(self, **kwargs):
        return self._get_catalog("ObservedCatalog", **kwargs)

    def get_pipelines(self):
        return self.config.get("Pipelines")

    def get_pipeline(self, name):
        pipelines = self.get_pipelines()
        pipeline_template = pipelines.get(name, None)
        if pipeline_template is None:
            raise ValueError(f"pipeline '{name}' not found in {self.project}")
        pipeline = {}
        for k, v in pipeline_template.items():
            match k:
                case "IterationVars":
                    # continue
                    pipeline["IterationVars"] = v
                case "PipelinePathTemplate":
                    pipeline["PipelinePath"] = v.format(**self.config["CommonPaths"])
                case "ConfigPathTemplate":
                    pipeline["ConfigPath"] = v.format(**self.config["CommonPaths"])
                case _:
                    # pipeline[k] = v.format(**self.config["CommonPaths"])
                    raise ValueError(f"Unexpected key '{k}' found in {self.project} pipeline {name}")

        return pipeline


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
        raise NotImplementedError("Connection to slurm not yet implemented")

    returncode = finished.returncode
    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    print("<<<<<<<<")
    print(f"subprocess completed with status {returncode} in {_elapsed_time} seconds")
    return returncode


def inspect(config_file):
    project = RAILProject.load_config(config_file)
    printable_config = pprint.pformat(project.config)
    print("RAIL Project")
    print(f">>>>>>>>")
    print(printable_config)
    print("<<<<<<<<")
    return 0


def truth_to_observed_pipeline(
    config_file,
    input_dir=None,
    config_path=None,
    output_dir=None,
    run_mode=RunMode.bash,
):
    project = RAILProject.load_config(config_file)
    pipeline = project.get_pipeline("truth_to_observed")

    source_catalogs = []
    sink_catalogs = []
    catalogs = []
    if (iteration_vars := pipeline.get("IterationVars")) is not None:
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

            source_catalog = project.get_reduced_catalog(**iteration_kwargs)
            sink_catalog = project.get_observed_catalog(**iteration_kwargs)
            sink_dir = os.path.dirname(sink_catalog)

            source_catalogs.append(source_catalog)
            sink_catalogs.append(sink_catalog)

            catalogs.append((source_catalog, sink_catalog))

            try:
                handle_command(run_mode, ["mkdir", "-p", f"{sink_dir}"])
                handle_command(run_mode, ["ceci", f"{pipeline['PipelinePath']}", f"config={pipeline['ConfigPath']}", f"inputs.input={source_catalog}", f"output_dir={sink_dir}", f"log_dir={sink_dir}"])
                handle_command(run_mode, ["tables-io", "convert", f"{sink_dir}/output_dereddener.pq", f"{sink_dir}/output.hdf5"])
            except Exception as msg:
                print(msg)
                return 1
            return 0

    # config_name = os.path.splitext(os.path.basename(config_path))[0]
    # config_name = Path(config_path).stem
    # config_dir = Path(config_path).parent
    # with open(config_path, "r") as fp:
    #     config_dict = yaml.safe_load(fp)
    #     config_file = config_dict["config"]
    # if output_dir is None:
    #     input_path = Path(input_dir)
    #     input_base = input_path.parent
    #     input_name = input_path.name
    #     # output_name = input_name + "_curated"
    #     output_name = input_name + "_" + config_name
    #     _output_dir = str(input_base / output_name)
    #     # output_dir = f"{data_dir}/{config_name}"

    # for healpix_path in glob.glob(f"{input_dir}/*"):
    #     healpix=os.path.basename(healpix_path)
    #     for input_path in glob.glob(f"{healpix_path}/*.parquet"):
    #         output_dir = f"{_output_dir}/{healpix}"

    #     try:
    #         # handle_command(run_mode, f"mkdir -p {output_dir}")
    #         # handle_command(run_mode, f"ceci {config_path} inputs.input={input_path} output_dir={output_dir} log_dir={output_dir}")
    #         # handle_command(run_mode, f"convert-table {output_dir}/output_dereddener.pq {output_dir}/output.hdf5")
    #         handle_command(run_mode, ["mkdir", "-p", f"{output_dir}"])
    #         # handle_command(run_mode, ["ceci", f"{config_path}", f"inputs.input={input_path}", f"output_dir={output_dir}", f"log_dir={output_dir}"])
    #         handle_command(run_mode, ["ceci", f"{config_path}", f"config={config_dir}/{config_file}", f"inputs.input={input_path}", f"output_dir={output_dir}", f"log_dir={output_dir}"])
    #         handle_command(run_mode, ["tables-io", "convert", f"{output_dir}/output_dereddener.pq", f"{output_dir}/output.hdf5"])
    #     except Exception as msg:
    #         print(msg)
    #         return 1
    # return 0


def inform_single(
    input_file,
    config_path,
    model_dir,
    run_mode=RunMode.bash,
):
    config_name = Path(config_path).stem
    config_dir = Path(config_path).parent
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config_file = config_dict["config"]
    output_dir=f'{model_dir}'
    command_line = [
        f'ceci',
        f'{config_path}',
        f'config={config_dir}/{config_file}',
        f'inputs.input={input_file}',
        f'output_dir={output_dir}',
        f'log_dir={output_dir}/logs',
    ]
    try:
        handle_command(run_mode, command_line)
    except Exception as msg:
        print(msg)
        return 1
    return 0


def estimate_single(
    input_file,
    config_path,
    pdf_dir,
    model_name,
    model_path,
    run_mode=RunMode.bash,
):
    config_name = Path(config_path).stem
    config_dir = Path(config_path).parent
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config_file = config_dict["config"]
    output_dir=f'{pdf_dir}'
    model_name = Path(model_path).stem.lower()
    command_line = [
        f'ceci',
        f'{config_path}',
        f'config={config_dir}/{config_file}',
        f'inputs.input={input_file}',
        f'inputs.spec_input={input_file}',
        f'inputs.{model_name}={model_path}',
        f'output_dir={output_dir}',
        f'log_dir={output_dir}',
    ]
    try:
        handle_command(run_mode, command_line)
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
    config_path,
    pdf_path,
    truth_path,
    output_dir,
    run_mode=RunMode.bash,
):
    config_name = Path(config_path).stem
    config_dir = Path(config_path).parent
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config_file = config_dict["config"]

    command_line = [
        f'ceci',
        f'{config_path}',
        f'config={config_dir}/{config_file}',
        f'inputs.input={pdf_path}',
        f'inputs.truth={truth_path}',
        f'output_dir={output_dir}',
        f'log_dir={output_dir}',
    ]
    try:
        handle_command(run_mode, command_line)
    except Exception as msg:
        print(msg)
        return 1
    return 0


def subsample_data(
    input_dir,
    output_dir,
    output_file,
    size,
    seed=None,
    label="",
):
    input_path = Path(input_dir)
    input_name = input_path.name
    input_base = input_path.parent
    if output_dir is None:
        output_dir = input_base
    output_path = Path(output_dir)
    if output_file is None:
        output_file = input_name + f"_{label}-seed{seed}-size{size}.parquet"

    output = str(output_path / output_file)

    sources = []
    for healpix in input_path.glob("*"):
        for source in healpix.glob("output_dereddener.pq"):
            print(source)
            sources.append(source)

    dataset = ds.dataset(sources)
    num_rows = dataset.count_rows()
    print("num rows", num_rows)
    rng = np.random.default_rng(seed)
    print("sampling", size)
    indices = rng.choice(num_rows, size=size, replace=False)
    subset = dataset.take(indices)
    print("writing", output)
    os.makedirs(output_dir, exist_ok=True)
    pq.write_table(
        subset,
        output,
    )
    print("done")
    return 0


def make_som_data(
    input_dir,
    output_dir,
    output_file,
):
    input_path = Path(input_dir)
    input_name = input_path.name
    input_base = input_path.parent
    if output_dir is None:
        train_path = input_base
    else:
        train_path = Path(output_dir)
    if output_file is None:
        output_file = input_name + f"_train-som.parquet"

    train_output = str(train_path / output_file)

    paths = []
    for healpix in input_path.glob("*"):
        for output in healpix.glob("output_dereddener.pq"):
            print(output)
            paths.append(output)

    dataset = ds.dataset(paths)
    num_rows = dataset.count_rows()
    schema = dataset.schema
    print("num rows", num_rows)
    print("writing", train_output)
    with pq.ParquetWriter(train_output, schema) as writer:
        for i, batch in enumerate(dataset.to_batches()):
            ii = i + 1
            print(f"writing batch {ii}", end="\r")
            writer.write_batch(batch)
        print("")
    print("done")
    return 0
