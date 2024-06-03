import math
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero


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
    "healpix",
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
    input_dir,
    maglim = 25.5,
):

    predicate = pc.field("LSST_obs_i") < maglim

    source_path = Path(input_dir)
    dataset = ds.dataset(
        input_dir,
        format="parquet",
        partitioning=["healpix"],
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

    batches = plan.to_reader(use_threads=True)

    sink_path = source_path.parent / (source_path.name + f"_healpixel_maglim_{maglim}")
    sink_dir = sink_path.as_posix()
    print(f"writing dataset to {sink_dir}")
    ds.write_dataset(
        batches,
        sink_dir,
        format="parquet",
        partitioning=["healpix"],
        # max_rows_per_group=1024,
        # max_rows_per_file=1024 * 100,
        # max_rows_per_file=1024**2 * 100,
    )
    print(f"writing completed")
