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
        run_config = config_dict["config"]
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
            handle_command(run_mode, ["ceci", f"{config_path}", f"config={config_dir}/{run_config}", f"inputs.input={input_path}", f"output_dir={output_dir}", f"log_dir={output_dir}"])
            handle_command(run_mode, ["convert-table", f"{output_dir}/output_dereddener.pq", f"{output_dir}/output.hdf5"])
        except Exception as msg:
            print(msg)
            return 1
    return 0


def inform_single(
    train_file,
    config_path,
    model_dir,
    run_mode=RunMode.bash,
):
    # config_name = os.path.splitext(os.path.basename(config_path))[0]
    config_name = Path(config_path).stem
    config_file = os.path.join(os.path.dirname(config_path), 'inform_roman_rubin_config.yml')
    output_dir=f'{model_dir}'
    command_line = [
        f'ceci',
        f'{config_path}',
        f'config={config_file}',
        f'inputs.input={train_file}',
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
    input_dir,
    config_path,
    pdf_dir,
    model_name,
    model_path,
    run_mode=RunMode.bash,
):
    # config_name = os.path.splitext(os.path.basename(config_path))[0]
    config_name = Path(config_path).stem
    config_file = os.path.join(os.path.dirname(config_path), 'estimate_roman_rubin_config.yml')
    input_dirs = glob.glob(f'{input_dir}/*')
    for input_dir_ in input_dirs:
        healpixel = os.path.basename(input_dir_)
        output_dir=f'{pdf_dir}/{config_name}/{healpixel}'
        input_path = f'{input_dir_}/output.hdf5'
        command_line = [
            f'ceci',
            f'{config_path}',
            f'config={config_file}',
            f'inputs.input={input_path}',
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
    config_path,
    pdf_dir,
    model_dir,
    run_mode=RunMode.bash,
):
    # config_name = os.path.splitext(os.path.basename(config_path))[0]
    config_name = Path(config_path).stem
    config_file = os.path.join(os.path.dirname(config_path), 'estimate_roman_rubin_config.yml')
    input_dirs = glob.glob(f'{input_dir}/*')
    model_paths = Path(model_dir).glob("*")
    for input_dir_ in input_dirs:
        healpixel = os.path.basename(input_dir_)
        output_dir=f'{pdf_dir}/{config_name}/{healpixel}'
        input_path = f'{input_dir_}/output.hdf5'
        command_line = [
            f'ceci',
            f'{config_path}',
            f'config={config_file}',
            f'inputs.input={input_path}',
            f'inputs.spec_input={input_path}',
            f'output_dir={output_dir}',
            f'log_dir={output_dir}',
        ]
        for model_path in model_paths:
            model_name = model_path.stem
            command_line.append(
                f'inputs.{model_name.lower()}={model_path}'
            )
        try:
            handle_command(run_mode, command_line)
        except Exception as msg:
            print(msg)
            return 1
    return 0


def make_training_data(
    input_dir,
    train_dir,
    train_file,
    size,
):
    input_path = Path(input_dir)
    input_name = input_path.name
    input_base = input_path.parent
    if train_dir is None:
        train_path = input_base
    else:
        train_path = Path(train_dir)
    if train_file is None:
        train_file = input_name + f"_train-{size}.parquet"

    train_output = str(train_path / train_file)

    paths = []
    for healpix in input_path.glob("*"):
        for output in healpix.glob("output_dereddener.pq"):
            print(output)
            paths.append(output)

    dataset = ds.dataset(paths)
    num_rows = dataset.count_rows()
    print("num rows", num_rows)
    rng = np.random.default_rng(1)
    print("sampling", size)
    indices = rng.choice(num_rows, size=size, replace=False)
    small = dataset.take(indices)
    print("writing", train_output)
    pq.write_table(
        small,
        train_output,
    )
    print("done")
    return 0
