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

from rail.utils.path_utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class SurveyNonuniformDegraderPipeline(RailPipeline):

    default_input_dict = dict(model=flow_file)
  
    def __init__(self):
        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        ### Creation steps:
        bands = ['u','g','r','i','z','y']
        rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
        
        # This may be changed later
        self.flow_engine_train = FlowCreator.build(
            model=flow_file,
            n_samples=10,
        )
        
        self.obs_condition = ObsCondition.build(
            connections=dict(input=self.flow_engine_train.io.output), 
        )
        
        self.col_remapper = ColumnMapper.build(
            connections=dict(input=self.obs_condition.io.output),
            columns=rename_dict,
        )
        
        ### Estimation steps:
        self.deredden = Dereddener.build(
            connections=dict(input=self.col_remapper.io.output),
            dustmap_dir=".",
        )
        
        ### convert table into hdf5 format for estimation
        self.table_conv = TableConverter.build(
            connections=dict(input=self.deredden.io.output),
            output_format='numpyDict',
        )
        
        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv.io.output),
            hdf5_groupname='',
            nt_array=[8],
            mmax=26.,
            type_file='',
        )
        
        self.estimate_bpz = BPZliteEstimator.build(
            connections=dict(input=self.table_conv.io.output,
                            model=self.inform_bpz.io.model,),
            hdf5_groupname='',
        )
        
        ### Tomographic binning
        self.tomopraphy = UniformBinningClassifier.build(
            connections=dict(input=self.estimate_bpz.io.output),
        )
