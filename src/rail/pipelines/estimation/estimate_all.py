#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.utils.name_utils import NameFactory
from rail.core.stage import RailStage, RailPipeline

import ceci



input_file = 'rubin_dm_dc2_example.pq'


ALL_ALGORITHMS = dict(
    train_z=dict(Estimate='TrainZEstimator'),
    simplenn=dict(Estimate='SklNeurNetEstimator'),
    knn=dict(Estimate='KNearNeighEstimator'),
    bpz=dict(Estimate='BPZliteEstimator'),
    fzboost=dict(Estimate='FlexZBoostEstimator'),
    gpz=dict(Estimate='GPzEstimator'),
    tpz=dict(Estimate='TPZliteEstimator'),
    #lephare=dict(Estimate='LephareEstimator'),
)
    

class EstimatePipeline(RailPipeline):

    default_input_dict={'input':'dummy.in'}
    
    def __init__(self, namer=None, algorithms=None, selection='default', flavor='baseline'):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = ALL_ALGORITHMS.copy()

        if namer is None:
            namer = NameFactory()

        models_dir = namer.resolve_path_template(
            'ceci_output_dir',
            selection=selection,
            flavor=flavor,
        )
        
        for key, val in algorithms.items():
            the_class = ceci.PipelineStage.get_stage(val['Estimate'])
            the_estimator = the_class.make_and_connect(
                name=f'estimate_{key}',
                aliases=dict(model=f"model_{key}"),
                hdf5_groupname='',
            )
            model_path = namer.resolve_path_template(
                "ceci_file_path",                
                stage=f'inform_{key}',
                tag='model',
                suffix='.pkl',
            )
            self.default_input_dict[f"model_{key}"] = os.path.join(models_dir, model_path)
            self.add_stage(the_estimator)

            
