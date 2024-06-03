#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.utils.name_utils import NameFactory, DataType, CatalogType, ModelType, PdfType, MetricType
from rail.core.stage import RailStage, RailPipeline

import ceci


namer = NameFactory()
from rail.utils.path_utils import RAILDIR

input_file = 'rubin_dm_dc2_example.pq'


ALL_ALGORITHMS = dict(
    train_z=dict(Inform='TrainZInformer', Estimate='TrainZEstimator'),
    simplenn=dict(Inform='SklNeurNetInformer', Estimate='SklNeurNetEstimator'),
    knn=dict(Inform='KNearNeighInformer', Estimate='KNearNeighEstimator'),
    bpz=dict(Inform='BPZliteInformer', Estimate='BPZliteEstimator'),
    fzboost=dict(Inform='FlexZBoostInformer', Estimate='FlexZBoostEstimator'),
    gpz=dict(Inform='GPzInformer', Estimate='GPzEstimator'),
    tpz=dict(Inform='TPZliteInformer', Estimate='TPZliteEstimator'),
    #lephare=dict(Inform='LephareInformer', Estimate='LephareEstimator'),
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
            algorithms = ALL_ALGORITHMS

        for key, val in algorithms.items():
            inform_class = ceci.PipelineStage.get_stage(val['Inform'])
            the_informer = inform_class.make_and_connect(
                name=f'inform_{key}',
                aliases=dict(input='input_train'),
                model=os.path.join(namer.get_data_dir(DataType.models, ModelType.estimator), "model_knn.pkl"),
                hdf5_groupname='',
            )
            self.add_stage(the_informer)

            estimate_class = ceci.PipelineStage.get_stage(val['Estimate'])
            the_estimator = estimate_class.make_and_connect(
                name=f'estimate_{key}',
                aliases=dict(input='input_test'),
                connections=dict(
                    model=the_informer.io.model,
                ),
                output=os.path.join(namer.get_data_dir(DataType.pdfs, PdfType.pz), f"output_{key}.hdf5"),
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
                output=os.path.join(namer.get_data_dir(DataType.metrics, MetricType.per_object), "output_trainz.hdf5"),
                summary=os.path.join(namer.get_data_dir(DataType.metrics, MetricType.summary_value), "summary_trainz.hdf5"),
                single_distribution_summary=os.path.join(namer.get_data_dir(DataType.metrics, MetricType.summary_value), "single_distribution_summary_trainz.hdf5"),
                hdf5_groupname='',
            )
            self.add_stage(the_evaluator)
       
   
