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
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class SurveyNonuniformDegraderPipeline(RailPipeline):

    default_input_dict = dict(model=flow_file)
    
    def __init__(self, namer=None):
        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if namer is None:
            namer = NameFactory()
        truth_catalog_dir = namer.resolve_path_template('truth_catalog_path')
        degraded_catalog_dir = namer.resolve_path_template('degraded_catalog_path', selection="default", flavor="baseline")
        path_kwargs = dict(
            selection="default",
            flavor="baseline",
        )

        ### Creation steps:
        bands = ['u','g','r','i','z','y']
        rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
        
        # This may be changed later
        self.flow_engine_train = FlowCreator.build(
            model=flow_file,
            n_samples=10,
            output=os.path.join(truth_catalog_dir, "output_flow_engine_train.pq"),
        )
        
        self.obs_condition = ObsCondition.build(
            connections=dict(input=self.flow_engine_train.io.output), 
            output=os.path.join(degraded_catalog_dir, "output_obscondition.pq"),
        )
        
        self.col_remapper = ColumnMapper.build(
            connections=dict(input=self.obs_condition.io.output),
            columns=rename_dict,
            output=os.path.join(degraded_catalog_dir, "output_col_remapper.pq"),
        )
        
        ### Estimation steps:
        self.deredden = Dereddener.build(
            connections=dict(input=self.col_remapper.io.output),
            dustmap_dir=".",
            output=os.path.join(degraded_catalog_dir, "output_deredden.pq"),
        )
        
        ### convert table into hdf5 format for estimation
        self.table_conv = TableConverter.build(
            connections=dict(input=self.deredden.io.output),
            output_format='numpyDict',
            output=os.path.join(degraded_catalog_dir, "output_table_conv.hdf5"),
        )
        
        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv.io.output),
            model=namer.resolve_path_template('estimator_model_path', algorithm='bpz', model_suffix='.pkl', **path_kwargs),
            hdf5_groupname='',
            nt_array=[8],
            mmax=26.,
            type_file='',
        )
        
        self.estimate_bpz = BPZliteEstimator.build(
            connections=dict(input=self.table_conv.io.output,
                            model=self.inform_bpz.io.model,),
            hdf5_groupname='',
            output=namer.resolve_path_template('pz_pdf_path', algorithm='bpz', **path_kwargs),
        )
        
        ### Tomographic binning
        self.tomopraphy = UniformBinningClassifier.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=namer.resolve_path_template('tomography_path', tomomethod='uniform_bin', **path_kwargs),
        )
