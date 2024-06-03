#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.utils.name_utils import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline

import ceci


namer = NameFactory()
from rail.utils.path_utils import RAILDIR

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
    
    def __init__(self, algorithms=None):

        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = ALL_ALGORITHMS

        for key, val in algorithms.items():
            the_class = ceci.PipelineStage.get_stage(val['Estimate'])
            the_estimator = the_class.make_and_connect(
                name=f'estimate_{key}',
                aliases=dict(model=f"model_{key}"),
                output=os.path.join(namer.get_data_dir(DataType.pdfs, PdfType.pz), f"output_{key}.hdf5"),
                hdf5_groupname='',
            )
            self.default_input_dict[f"model_{key}"] = os.path.join(namer.get_data_dir(DataType.models, ModelType.estimator), f"model_{key}.pkl")
            self.add_stage(the_estimator)

            
