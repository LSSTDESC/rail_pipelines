#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.pipelines.utils.name_factory import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline

import ceci

namer = NameFactory()
from rail.utils.path_utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class SurveyNonuniformDegraderPipeline(RailPipeline):
    
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
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.created), "output_flow_engine_train.pq"),
        )
        
        self.obs_condition = ObsCondition.build(
            connections=dict(input=self.flow_engine_train.io.output), 
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_obscondition.pq"),
        )
        
        self.col_remapper = ColumnMapper.build(
            connections=dict(input=self.obs_condition.io.output),
            columns=rename_dict,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_col_remapper.pq"),
        )
        
        ### Estimation steps:
        self.deredden = Dereddener.build(
            connections=dict(input=self.col_remapper.io.output),
            dustmap_dir=".",
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_deredden.pq"),
        )
        
        ### convert table into hdf5 format for estimation
        self.table_conv = TableConverter.build(
            connections=dict(input=self.deredden.io.output),
            output_format='numpyDict',
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_table_conv.hdf5"),
        )
        
        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv.io.output),
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), 'trained_BPZ.pkl'),
            hdf5_groupname='',
            nt_array=[8],
            mmax=26.,
            type_file='',
        )
        
        self.estimate_bpz = BPZliteEstimator.build(
            connections=dict(input=self.table_conv.io.output,
                            model=self.inform_bpz.io.model,),
            hdf5_groupname='',
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_bpz.hdf5"),
        )
        
        ### Tomographic binning
        self.tomopraphy = UniformBinningClassifier.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_tomography.hdf5"),
        )
        
        
if __name__ == '__main__':
    pipe = SurveyNonuniformDegraderPipeline()
    pipe.initialize(dict(model=flow_file), dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_survey_nonuniformity.yml')
