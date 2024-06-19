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

from rail.utils.path_utils import RAILDIR

input_file = 'rubin_dm_dc2_example.pq'


ALL_ALGORITHMS = dict(
    train_z=dict(Inform='TrainZInformer'),
    simplenn=dict(Inform='SklNeurNetInformer'),
    knn=dict(Inform='KNearNeighInformer'),
    bpz=dict(Inform='BPZliteInformer'),
    fzboost=dict(Inform='FlexZBoostInformer'),
    gpz=dict(Inform='GPzInformer'),
    tpz=dict(Inform='TPZliteInformer'),
    #lephare=dict(Inform='LephareInformer'),
)


class InformPipeline(RailPipeline):

    default_input_dict={'input':'dummy.in'}

    def __init__(self, namer, algorithms=None, selection="default", flavor="baseline"):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = ALL_ALGORITHMS

        path_kwargs = dict(
            selection=selection,
            flavor=flavor,
        )

        for key, val in algorithms.items():
            the_class = ceci.PipelineStage.get_stage(val['Inform'])
            the_informer = the_class.make_and_connect(
                name=f'inform_{key}',
                model=namer.resolve_path_template(
                    "estimator_model_path",
                    algorithm=key,
                    model_suffix='.pkl',
                    **path_kwargs,
                ),
                hdf5_groupname='',
            )
            self.add_stage(the_informer)
