import math
from pathlib import Path
import itertools
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import acero

from rail.cli.pipe_scripts import RAILProject


COLUMNS = [
    "galaxy_id",
    "ra",
    "dec",
    "redshift",
    "LSST_obs_u",
    "LSST_obs_g",
    "LSST_obs_r",
    "LSST_obs_i",
    "LSST_obs_z",
    "LSST_obs_y",
    "ROMAN_obs_F184",
    "ROMAN_obs_J129",
    "ROMAN_obs_H158",
    "ROMAN_obs_W146",
    "ROMAN_obs_Z087",
    "ROMAN_obs_Y106",
    "ROMAN_obs_K213",
    "ROMAN_obs_R062",
    "totalEllipticity",
    "totalEllipticity1",
    "totalEllipticity2",
    "diskHalfLightRadiusArcsec",
    "spheroidHalfLightRadiusArcsec",
    "bulge_frac",
    # "healpix",
]

PROJECTIONS = [
    {
        "mag_u_lsst": pc.field("LSST_obs_u"),
        "mag_g_lsst": pc.field("LSST_obs_g"),
        "mag_r_lsst": pc.field("LSST_obs_r"),
        "mag_i_lsst": pc.field("LSST_obs_i"),
        "mag_z_lsst": pc.field("LSST_obs_z"),
        "mag_y_lsst": pc.field("LSST_obs_y"),
        "totalHalfLightRadiusArcsec": pc.add(
            pc.multiply(
                pc.field("diskHalfLightRadiusArcsec"),
                pc.subtract(pc.scalar(1), pc.field("bulge_frac")),
            ),
            pc.multiply(
                pc.field("spheroidHalfLightRadiusArcsec"),
                pc.field("bulge_frac"),
            )
        ),
        "_orientationAngle": pc.atan2(pc.field("totalEllipticity2"), pc.field("totalEllipticity1")),
    },
    {
        "major": pc.divide(
            pc.field("totalHalfLightRadiusArcsec"),
            pc.sqrt(pc.field("totalEllipticity")),
        ),
        "minor": pc.multiply(
            pc.field("totalHalfLightRadiusArcsec"),
            pc.sqrt(pc.field("totalEllipticity")),
        ),
        "orientationAngle": pc.multiply(
            pc.scalar(0.5),
            pc.subtract(
                pc.field("_orientationAngle"),
                pc.multiply(
                    pc.floor(
                        pc.divide(
                            pc.field("_orientationAngle"),
                            pc.scalar(2 * math.pi)
                        )
                    ),
                    pc.scalar(2 * math.pi)
                )
            )
        ),
    }
]


def reduce_roman_rubin_data(
    config_file,
    input_dir=None,
    maglim=25.5,
):
    project = RAILProject.load_config(config_file)

    source_catalogs = []
    sink_catalogs = []
    catalogs = []
    predicates = []

    # FIXME
    iteration_vars = list(project.config.get("IterationVars").keys())
    if iteration_vars is not None:
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

            source_catalog = project.get_input_catalog(**iteration_kwargs)
            sink_catalog = project.get_reduced_catalog(**iteration_kwargs)
            sink_dir = os.path.dirname(sink_catalog)
            if (selection_tag := iteration_kwargs.get("selection")) is not None:
                selection = project.get_selection(selection_tag)
                predicate = pc.field("LSST_obs_i") < selection["maglim_i"][1]
            else:
                predicate = None

            if not os.path.isfile(source_catalog):
                raise ValueError(f"Input file {source_catalog} not found")

            # FIXME properly warn
            if os.path.isfile(sink_catalog):
                # raise ValueError(f"Input file {source_catalog} not found")
                print(f"Warning: output file {sink_catalog} found; may be rewritten...")

            source_catalogs.append(source_catalog)
            sink_catalogs.append(sink_catalog)

            catalogs.append((source_catalog, sink_catalog))

            predicates.append(predicate)

            dataset = ds.dataset(
                source_catalog,
                # source_catalogs,
                # input_dir,
                format="parquet",
                # partitioning=["healpix"],
            )

            scan_node = acero.Declaration(
                "scan",
                acero.ScanNodeOptions(
                    dataset,
                    columns=COLUMNS,
                    filter=predicate,
                ),
            )

            filter_node = acero.Declaration(
                "filter",
                acero.FilterNodeOptions(
                    predicate,
                ),
            )

            column_projection = {
                k: pc.field(k)
                for k in COLUMNS
            }
            projection = column_projection
            project_nodes = []
            for _projection in PROJECTIONS:
                for k, v in _projection.items():
                    projection[k] = v
                project_node = acero.Declaration(
                    "project",
                    acero.ProjectNodeOptions(
                        [v for k, v in projection.items()],
                        names=[k for k, v in projection.items()],
                    )
                )
                project_nodes.append(project_node)

            seq = [
                scan_node,
                filter_node,
                *project_nodes,
            ]
            plan = acero.Declaration.from_sequence(seq)
            print(plan)

            # batches = plan.to_reader(use_threads=True)
            table = plan.to_table(use_threads=True)
            print(f"writing dataset to {sink_catalog}")
            os.makedirs(sink_dir, exist_ok=True)
            pq.write_table(table, sink_catalog)

            # sink_path = source_path.parent / (source_path.name + f"_healpixel_maglim_{maglim}")
            # sink_dir = sink_path.as_posix()
            # print(f"writing dataset to {sink_catalog}")
            # ds.write_dataset(
            #     batches,
            #     # sink_dir,
            #     sink_catalog,
            #     format="parquet",
            #     # partitioning=["healpix"],
            #     # max_rows_per_group=1024,
            #     # max_rows_per_file=1024 * 100,
            #     # max_rows_per_file=1024**2 * 100,
            # )
            # with pq.ParquetWriter(sink_catalog, schema) as writer:
            #     for batch in batches:
            #         writer.write_batch(batch)

        print(f"writing completed")
