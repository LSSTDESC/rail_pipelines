#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os

from rail.core.stage import RailStage, RailPipeline
from rail.evaluation.single_evaluator import SingleEvaluator
from rail.utils.algo_library import PZ_ALGORITHMS


shared_stage_opts = dict(
    metrics=['all'],
    exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
    hdf5_groupname="",
    limits=[0, 3.5],
    truth_point_estimates=['redshift'],
    point_estimates=['zmode'],
)


class EvaluationPipeline(RailPipeline):

    default_input_dict=dict(truth='dummy.in')

    def __init__(self, algorithms=None, pdfs_dir='.'):
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
            pdf_path = f'estimate_output_{key}.hdf5'
            self.default_input_dict[f"input_evaluate_{key}"] = os.path.join(pdfs_dir, pdf_path)
            self.add_stage(the_eval)
