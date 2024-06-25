#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os

from rail.core.stage import RailStage, RailPipeline
from rail.evaluation.single_evaluator import SingleEvaluator
from rail.utils.project import PZ_ALGORITHMS


shared_stage_opts = dict(
    metrics=['all'],
    exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
    hdf5_groupname="",
    limits=[0, 3.5],
    truth_point_estimates=['redshift'],
    point_estimates=['mode'],
)


class EvaluationPipeline(RailPipeline):

    default_input_dict={}

    def __init__(self, self, namer, algorithms=None, selection="default", flavor="baseline"):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = PZ_ALGORITHMS

        for key in algorithms.keys():
            the_eval = SingleEvaluator.make_and_connect(
                name=f'evaluate_{key}',
                aliases=dict(input=f"input_evaluate_{key}"),
                **shared_stage_opts,
            )
            pdf_path = namer.resolve_path_template(
                "ceci_file_path",                
                stage=f'estimate_{key}',
                tag='output',
                suffix='.hdf5',
            )            
            self.default_input_dict[f"input_evaluate_{key}"] = os.path.join(pdfs_dir, pdf_path)
            self.add_stage(the_eval)
