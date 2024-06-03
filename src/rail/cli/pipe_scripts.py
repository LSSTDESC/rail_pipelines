import os
import sys
import glob
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import yaml

from .pipe_options import RunMode


# data_dir=${PZ_BASE_AREA}/data
# model_top = os.path.expandvars("${PZ_BASE_AREA}/models")
# pdf_dir = os.path.expandvars("${PZ_BASE_AREA}/pdfs/roman_rubin_2023_v1.1.3_curated")
# for healpix in Path("/sdf/data/rubin/shared/pz/data/roman_rubin_2023_v1.1.3_curated").glob("*"):
# train_file = "/sdf/data/rubin/shared/pz/data/training/roman_rubin_2023_v1.1.3_parquet_healpixel_maglim_25.5_observed_100k.pq",


def handle_command(run_mode, command_line):
    print(f"subprocess:", *command_line)
    print(f">>>>>>>>")
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
    print(f"<<<<<<<<")
    print(f"subprocess completed with status {returncode}")
    return returncode


def truth_to_observed_pipeline(
    input_dir,
    config_path,
    output_dir=None,
    run_mode=RunMode.bash,
):
    # config_name = os.path.splitext(os.path.basename(config_path))[0]
    config_name = Path(config_path).stem
    config_dir = Path(config_path).parent
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config_file = config_dict["config"]
    if output_dir is None:
        input_path = Path(input_dir)
        input_base = input_path.parent
        input_name = input_path.name
        # output_name = input_name + "_curated"
        output_name = input_name + "_" + config_name
        _output_dir = str(input_base / output_name)
        # output_dir = f"{data_dir}/{config_name}"

    for healpix_path in glob.glob(f"{input_dir}/*"):
        healpix=os.path.basename(healpix_path)
        for input_path in glob.glob(f"{healpix_path}/*.parquet"):
            output_dir = f"{_output_dir}/{healpix}"

        try:
            # handle_command(run_mode, f"mkdir -p {output_dir}")
            # handle_command(run_mode, f"ceci {config_path} inputs.input={input_path} output_dir={output_dir} log_dir={output_dir}")
            # handle_command(run_mode, f"convert-table {output_dir}/output_dereddener.pq {output_dir}/output.hdf5")
            handle_command(run_mode, ["mkdir", "-p", f"{output_dir}"])
            # handle_command(run_mode, ["ceci", f"{config_path}", f"inputs.input={input_path}", f"output_dir={output_dir}", f"log_dir={output_dir}"])
            handle_command(run_mode, ["ceci", f"{config_path}", f"config={config_dir}/{config_file}", f"inputs.input={input_path}", f"output_dir={output_dir}", f"log_dir={output_dir}"])
            handle_command(run_mode, ["tables-io", "convert", f"{output_dir}/output_dereddener.pq", f"{output_dir}/output.hdf5"])
        except Exception as msg:
            print(msg)
            return 1
    return 0


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

    print("done")
    return 0
