#!/usr/bin/env python
# coding: utf-8

import ceci

# Various rail modules
from rail.core.stage import RailStage, RailPipeline
from rail.evaluation.single_evaluator import SingleEvaluator
from rail.utils.project import PZ_ALGORITHMS


input_file = 'rubin_dm_dc2_example.pq'


eval_shared_stage_opts = dict(
    metrics=['all'],
    exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
    hdf5_groupname="",
    limits=[0, 3.5],
    truth_point_estimates=['redshift'],
    point_estimates=['zmode'],
)


class PzPipeline(RailPipeline):

    default_input_dict={
        'input_train':'dummy.in',
        'input_test':'dummy.in',
    }

    def __init__(self, algorithms=None):        
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = PZ_ALGORITHMS

        for key, val in algorithms.items():
            inform_class = ceci.PipelineStage.get_stage(val['Inform'], val['Module'])
            the_informer = inform_class.make_and_connect(
                name=f'inform_{key}',
                aliases=dict(input='input_train'),
                hdf5_groupname='',
            )
            self.add_stage(the_informer)

            estimate_class = ceci.PipelineStage.get_stage(val['Estimate'], val['Module'])
            the_estimator = estimate_class.make_and_connect(
                name=f'estimate_{key}',
                aliases=dict(input='input_test'),
                connections=dict(
                    model=the_informer.io.model,
                ),
                hdf5_groupname='',
            )
            self.add_stage(the_estimator)

            the_evaluator = SingleEvaluator.make_and_connect(
                name=f'evaluate_{key}',
                aliases=dict(truth='input_test'),
                connections=dict(
                    input=the_estimator.io.output,
                ),
                point_estimates=['mode'],
                truth_point_estimates=["redshift"],
                metrics=["all"],
                metric_config=dict(brier=dict(limits=[0., 3.5])),
                exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
                hdf5_groupname='',
            )
            self.add_stage(the_evaluator)
