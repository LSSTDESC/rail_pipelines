import os
from rail.utils.testing_utils import build_and_read_pipeline
import ceci

import pytest


@pytest.mark.parametrize(
    "pipeline_class",
    [
        'rail.pipelines.estimation.estimate_all.EstimatePipeline',
        'rail.pipelines.estimation.inform_all.InformPipeline',
        'rail.pipelines.estimation.pz_all.PzPipeline',
        'rail.pipelines.estimation.tomography.TomographyPipeline',
        'rail.pipelines.utils.prepare_observed.PrepareObservedPipeline',
        'rail.pipelines.examples.goldenspike.goldenspike.GoldenspikePipeline',
        'rail.pipelines.examples.survey_nonuniformity.survey_nonuniformity.SurveyNonuniformDegraderPipeline',
    ]
)
def test_build_and_read_pipeline(pipeline_class):
    #build_and_read_pipeline(pipeline_class)
    from rail.cli.rail.scripts import build_pipeline
    short_name = pipeline_class.split(".")[-1]
    yaml_file = f"{short_name}.yml"
    config_yaml_file = f"{short_name}_config.yml"
    build_pipeline(pipeline_class, yaml_file, "rubin")
    try:
        _pr = ceci.Pipeline.read(yaml_file)
    except:
        with open(config_yaml_file) as fin:
            print(fin.read())
        raise
    os.unlink(yaml_file)
    os.unlink(config_yaml_file)



