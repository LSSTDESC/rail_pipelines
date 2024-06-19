#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.core.stage import RailStage, RailPipeline

import ceci


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
            algorithms = ALL_ALGORITHMS

        path_kwargs = dict(
            selection=selection,
            flavor=flavor,
        )
               
        for key in algorithms.keys():
            the_eval = SingleEvaluator.make_and_connect(
                name=f'evaluate_{key}',
                aliases=dict(input=f"input_evalute_{key}"),
                output=namer.resolve_path_template(
                    'per_object_metrics_path',
                    algorithm=key,
                    **path_kwargs,
                ),
                summary=namer.resolve_path_template(
                    'summary_value_metrics_path',
                    algorithm=key,
                    **path_kwargs,
                ),
                single_distribution_summary=namer.resolve_path_template(
                    'summary_pdf_metrics_path',
                    algorithm=key,
                    **path_kwargs,
                ),               
                **shared_stage_opts,                
            )
            self.default_input_dict[f"input_evaluate_{key}"] = namer.resolve_path_template(
                "pz_pdf_path",
                algorithm=key,
                **path_kwargs,
            )
            self.add_stage(the_eval)


