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
        'rail.pipelines.examples.goldenspike.goldenspike.GoldenspikePipeline',
        'rail.pipelines.examples.survey_nonuniformity.survey_nonuniformity.SurveyNonuniformDegraderPipeline',
    ]
)
def test_build_and_read_pipeline(pipeline_class):
    build_and_read_pipeline(pipeline_class)

