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
 

    def __init__(self, name, config_dict):
        self.name = name
        self._config_dict = config_dict
        self.config = copy.deepcopy(self.config_template)
        for k in self.config.keys():
            if (v := self._config_dict.get(k)) is not None:
                self.config[k] = v
        # self.interpolants = self.get_interpolants()
        self.name_factory = name_utils.NameFactory(
            config=self.config,
            templates=config_dict.get('PathTemplates', {}),
            interpolants=self.config.get("CommonPaths", {})
        )
        self.name_factory.resolve_from_config(
            self.config.get("CommonPaths")
        )

    def __repr__(self):
        return f"{self.name}"

    @staticmethod
    def load_config(config_file):
        project_name = Path(config_file).stem
        with open(config_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        project = RAILProject(project_name, config_dict)
        # project.resolve_common()
        return project

    # def get_interpolants(self):
    #     interpolants = {}

    #     if (common_dict := self.config.get("CommonPaths")) is not None:
    #         for key, value in common_dict.items():
    #             new_value = value.format(**interpolants)
    #             interpolants[key] = new_value

    #     return interpolants

    # def resolve_common(self):
    #     # resolved = {}
    #     # for target in ["CommonPaths"]:
    #     #     done = self.name_factory.resolve(
    #     #         self.config.get(target),
    #     #         self.interpolants,
    #     #     )
    #     #     resolved[target] = done
    #     #     self.config.get(target).update(done)

    #     # return resolved

    #     resolved = self.name_factory.resolve_config(
    #         self.config.get("CommonPaths")
    #     )
    #     return resolved

    def get_partial_templates(self):
        resolved_templates = {}
        path_templates = self.config.get("PathTemplates")
        for k, v in path_templates.items():
            resolved_templates[k] = functools.partial(
                v.format
            )

        return resolved_templates

    def get_files(self):
        return self.config.get("Files")

    def _get_file(self, name, **kwargs):
        files = self.get_files()
        file_dict = files.get(name, None)
        if file_dict is None:
            raise ValueError(f"file '{name}' not found in {self}")
        path = self.name_factory.resolve_path(file_dict, "PathTemplate", **kwargs)
        return path

    def get_file(self, name, **kwargs):
        files = self.get_files()
        file_dict = files.get(name, None)
        if file_dict is None:
            raise ValueError(f"file '{name}' not found in {self}")
        path = self.name_factory.resolve_path(file_dict, "PathTemplate", **kwargs)
        return path

    def get_flavors(self):
        flavors = self.config.get("Flavors")
        baseline = flavors.get("baseline", {})
        for k, v in flavors.items():
            if k != "baseline":
                flavors[k] = baseline | v

        return flavors

    def get_flavor(self, name):
        flavors = self.get_flavors()
        flavor = flavors.get(name, None)
        if flavor is None:
            raise ValueError(f"flavor '{name}' not found in {self}")
        return flavor

    def get_selections(self):
        return self.config.get("Selections")

    def get_selection(self, name):
        selections = self.get_selections()
        selection = selections.get(name, None)
        if selection is None:
            raise ValueError(f"selection '{name}' not found in {self}")
        return selection

    def get_pzalgorithms(self):
        return self.config.get("PZAlgorithms")

    def get_pzalgorithm(self, name):
        pzalgorithms = self.get_pzalgorithms()
        pzalgorithm = pzalgorithms.get(name, None)
        if pzalgorithm is None:
            raise ValueError(f"pz algorithm '{name}' not found in {self}")
        return pzalgorithm

    def get_nzalgorithms(self):
        return self.config.get("NZAlgorithms")

    def get_nzalgorithm(self, name):
        nzalgorithms = self.get_nzalgorithms()
        nzalgorithm = nzalgorithms.get(name, None)
        if nzalgorithm is None:
            raise ValueError(f"nz algorithm '{name}' not found in {self}")
        return nzalgorithm

    def get_catalog(self, catalog, **kwargs):
        # if (input_catalog := self.config.get(catalog)) is not None:
        #     if (path_template := input_catalog.get("PathTemplate")) is not None:
        #         path = path_template.format(
        #             **self.config["CommonPaths"],
        #             **kwargs,
        #         )
        # else:
        #     path = None
        catalog_dict = self.config.get(catalog, {})
        path = self.name_factory.resolve_path(catalog_dict, "PathTemplate", **kwargs)

        return path

    def get_input_catalog(self, **kwargs):
        return self.get_catalog("InputCatalog", **kwargs)

    def get_reduced_catalog(self, **kwargs):
        return self.get_catalog("ReducedCatalog", **kwargs)

    def get_observed_catalog(self, **kwargs):
        return self.get_catalog("ObservedCatalog", **kwargs)

    def get_pipelines(self):
        return self.config.get("Pipelines")

    def get_pipeline(self, name, **kwargs):
        pipelines = self.get_pipelines()
        pipeline_template = pipelines.get(name, None)
        if pipeline_template is None:
            raise ValueError(f"pipeline '{name}' not found in {self}")
        pipeline = {}
        for k, v in pipeline_template.items():
            match k:
                case "IterationVars":
                    # continue
                    pipeline["IterationVars"] = v
                case "PipelinePathTemplate":
                    pipeline["PipelinePath"] = self.name_factory.resolve_path(
                        pipeline_template,
                        "PipelinePathTemplate",
                        **kwargs,
                    )
                case "ConfigPathTemplate":
                    pipeline["ConfigPath"] = self.name_factory.resolve_path(
                        pipeline_template,
                        "ConfigPathTemplate",
                        **kwargs,
                    )
                case _:
                    # pipeline[k] = v.format(**self.config["CommonPaths"])
                    raise ValueError(f"Unexpected key '{k}' found in {self} pipeline {name}")

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
    print(f"subprocess completed with status {returncode} in {_elapsed_time} seconds\n")
    return returncode


def inspect(config_file):
    project = RAILProject.load_config(config_file)
    printable_config = pprint.pformat(project.config, compact=True)
    print(f"RAIL Project: {project}")
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

    # Loop through all possible combinations of the iteration variables that are
    # relevant to this pipeline
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

            if not os.path.isfile(source_catalog):
                raise ValueError(f"Input file {source_catalog} not found")
            try:
                handle_command(run_mode, ["mkdir", "-p", f"{sink_dir}"])
                handle_command(run_mode, ["ceci", f"{pipeline['PipelinePath']}", f"config={pipeline['ConfigPath']}", f"inputs.input={source_catalog}", f"output_dir={sink_dir}", f"log_dir={sink_dir}"])
                handle_command(run_mode, ["tables-io", "convert", "--input", f"{sink_dir}/output_dereddener.pq", "--output", f"{sink_dir}/output.hdf5"])
            except Exception as msg:
                print(msg)
                return 1
        return 0

    # FIXME need to get catalogs even if iteration not specified; this return fallback isn't ideal
    return 1


def inform_pipeline(
    config_file,
    input_file=None,
    config_path=None,
    model_dir=None,
    run_mode=RunMode.bash,
):
    # config_name = Path(config_path).stem
    # config_dir = Path(config_path).parent
    # with open(config_path, "r") as fp:
    #     config_dict = yaml.safe_load(fp)
    #     config_file = config_dict["config"]
    # output_dir=f'{model_dir}'
    # command_line = [
    #     f'ceci',
    #     f'{config_path}',
    #     f'config={config_dir}/{config_file}',
    #     f'inputs.input={input_file}',
    #     f'output_dir={output_dir}',
    #     f'log_dir={output_dir}/logs',
    # ]
    # try:
    #     handle_command(run_mode, command_line)
    # except Exception as msg:
    #     print(msg)
    #     return 1
    # return 0

    project = RAILProject.load_config(config_file)
    
    flavors = project.get_flavors()
    pzalgorithms = project.get_pzalgorithms()
    nzalgorithms = project.get_nzalgorithms()

    iteration_vars = list(project.config.get("IterationVars").keys())
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

        for flavor_k, flavor_v in flavors.items():
            for pz_k, pz_v in pzalgorithms.items():
                kwargs = {
                    "flavor": flavor_k,
                    "pzalgorithm": pz_k,
                }
                # FIXME all of the current algorithms in the config have something like
                # 'pzalgorithm', but I think we also want to specify some for nz algos.
                # will need to sort that out...

                pipeline = project.get_pipeline("inform", **kwargs)

                train_file = flavor_v.get("FileAliases").get("TrainFile")
                source_catalog = project.get_file(train_file, **kwargs, **iteration_kwargs)

                # FIXME where is this specified in the project config?
                # we should provide an interface instead of just grabbing from the internal config dict...
                sink_dir = project.config.get("CommonPaths").get("project_dir")

                command_line = [
                    f"ceci",
                    f"{pipeline['PipelinePath']}",
                    f"config={pipeline['ConfigPath']}",
                    f"inputs.input={source_catalog}",
                    f"output_dir={sink_dir}",
                    f"log_dir={sink_dir}/logs",
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
    config_file,
    input_dir=None,
    output_dir=None,
    output_file=None,
    size=None,
    seed=None,
    label="",
):
    # input_path = Path(input_dir)
    # input_name = input_path.name
    # input_base = input_path.parent
    # if output_dir is None:
    #     output_dir = input_base
    # output_path = Path(output_dir)
    # if output_file is None:
    #     output_file = input_name + f"_{label}-seed{seed}-size{size}.parquet"

    # output = str(output_path / output_file)

    # sources = []
    # for healpix in input_path.glob("*"):
    #     for source in healpix.glob("output_dereddener.pq"):
    #         print(source)
    #         sources.append(source)

    # dataset = ds.dataset(sources)
    # num_rows = dataset.count_rows()
    # print("num rows", num_rows)
    # rng = np.random.default_rng(seed)
    # print("sampling", size)
    # indices = rng.choice(num_rows, size=size, replace=False)
    # subset = dataset.take(indices)
    # print("writing", output)
    # os.makedirs(output_dir, exist_ok=True)
    # pq.write_table(
    #     subset,
    #     output,
    # )
    # print("done")
    # return 0

    file_tag = ""
    match label:
        case "train":
            file_tag = "train_file_100k"
        case "test":
            file_tag = "test_file_100k"
        case _:
            raise ValueError(f"Subsample '{label}' not valid")

    project = RAILProject.load_config(config_file)

    project_files = project.get_files()
    file_metadata = project_files.get(file_tag)

    source_tag = file_metadata.get("Input")
    size = file_metadata.get("NumObjects")

    iteration_vars = list(project.config.get("IterationVars").keys())
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

        source_catalog = project.get_catalog(source_tag, **iteration_kwargs)
        sink_catalog = project.get_file(file_tag, **iteration_kwargs)
        raise ValueError("SM: not finished implementation. We need to sort out when to subselect the data and when to do file conversion; this function is much easier to impelemnt over parquet datasets, but right now the config assumes we're working on the ObservedCatalog, which is an hdf5 file predicated on a healpixe...")



# def make_som_data(
#     input_dir,
#     output_dir,
#     output_file,
# ):
#     input_path = Path(input_dir)
#     input_name = input_path.name
#     input_base = input_path.parent
#     if output_dir is None:
#         train_path = input_base
#     else:
#         train_path = Path(output_dir)
#     if output_file is None:
#         output_file = input_name + f"_train-som.parquet"
# 
#     train_output = str(train_path / output_file)
# 
#     paths = []
#     for healpix in input_path.glob("*"):
#         for output in healpix.glob("output_dereddener.pq"):
#             print(output)
#             paths.append(output)
# 
#     dataset = ds.dataset(paths)
#     num_rows = dataset.count_rows()
#     schema = dataset.schema
#     print("num rows", num_rows)
#     print("writing", train_output)
#     with pq.ParquetWriter(train_output, schema) as writer:
#         for i, batch in enumerate(dataset.to_batches()):
#             ii = i + 1
#             print(f"writing batch {ii}", end="\r")
#             writer.write_batch(batch)
#         print("")
#     print("done")
#     return 0
