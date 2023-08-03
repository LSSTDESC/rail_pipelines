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
from rail.core.utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class SurveyNonuniformDegraderPipeline(RailPipeline):
    
    def __init__(self):
        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        
        bands = ['u','g','r','i','z','y']
        #band_dict = {band:f'mag_{band}_lsst' for band in bands}
        #rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
        
        ### Creation steps:
        
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
        
        ### Estimation steps:
        
        self.deredden = Dereddener.build(
            connections=dict(input=self.obs_condition.io.output),
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_deredden.pq"),
        )
        
        self.estimate_bpz = BPZliteEstimator.build(
            connections=dict(input=self.deredden.io.output,),
            hdf5_groupname='',
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_bpz.hdf5"),
        )
        
        """
        # some sort of point estimates for pz
        self.point_estimate_test = PointEstimateHist.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_point_estimate_test.hdf5"),
            single_NZ=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "single_NZ_point_estimate_test.hdf5"),
        )
        """
        
        ### Tomographic binning
        
        self.tomopraphy = UniformBinningClassifier.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_tomography.hdf5"),
        )
        
        
if __name__ == '__main__':
    pipe = SurveyNonuniformDegraderPipeline()
    pipe.flow_engine_train.config.update(n_samples=5)
    pipe.initialize(dict(model=flow_file), dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_survey_nonuniformity.yml')
