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

input_file = 'rubin_dm_dc2_example.pq'


class EstimatePipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        #band_list = [f'mag_{band}_lsst' for band in bands] + [f'mag_err_{band}_lsst' for band in bands]

        self.estimate_trainz = TrainZEstimator.build(
            aliases=dict(model="model_trainz"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_trainz.hdf5"),
            hdf5_groupname='',
        )

        self.estimate_simplenn = SklNeurNetEstimator.build(
            aliases=dict(model="model_simplenn"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_simplenn.hdf5"),
            hdf5_groupname='',
        )
        
        self.estimate_knn = KNearNeighEstimator.build(
            aliases=dict(model="model_knn"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_knn.hdf5"),
            hdf5_groupname='',
        )
        
        self.estimate_simplesom = MiniSOMSummarizer.build(
            aliases=dict(model="model_simplesom"),            
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_simplesom.hdf5"),            
             hdf5_groupname='',
        )

        self.estimate_somoclu = SOMocluSummarizer.build(
            aliases=dict(model="model_somoclu"),            
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_somoclu.hdf5"),            
            hdf5_groupname='',
        )

        self.estimate_bpz = BPZliteEstimator.build(
            aliases=dict(model="model_bpz"),            
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_bpz.hdf5"),            
            hdf5_groupname='',
        )
        
        """
        self.estimate_delight = Estimate_DelightPZ.build(
            aliases=dict(model="model_delight"),            
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_delight.hdf5"),            
            hdf5_groupname='',

        )
        """
        
        self.estimate_fzboost = FlexZBoostEstimator.build(
            aliases=dict(model="model_fzboost"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_FZBoost.hdf5"),
            hdf5_groupname='',
        )

        self.estimate_gpz = GPzEstimator.build(
            aliases=dict(model="model_gpz"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_gpz.pkl"),
            hdf5_groupname='',
        )

        self.estimate_pzflow = PZFlowEstimator.build(
            aliases=dict(model="model_pzflow"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_pzflow.pkl"),
            hdf5_groupname='',
        )
        
        self.estimate_tpz = TPZliteEstimator.build(
            aliases=dict(model="model_tpz"),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_tpz.pkl"),
            hdf5_groupname='',
        )
        
        self.estimate_lephare = LephareEstimateer.build(
           aliases=dict(model="model_lephare"),
           output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_lephare.pkl"),
           hdf5_groupname='',
        )
         

if __name__ == '__main__':    
    pipe = EstimatePipeline()
    input_dict = dict(
        model_knn=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_knn.pkl"),
        model_simplenn=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_simplenn.pkl"),
        model_simplesom=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_simplesom.pkl"),
        model_somoclu=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_somoclu.pkl"),
        model_fzboost=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_FZBoost.hdf5"), #_fzboost
        model_trainz=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_trainz.pkl"),        
        model_bpz=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_bpz.hdf5"),        
        model_gpz=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_gpz.pkl"),        
        model_pzflow=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_pzflow.pkl"),        
        model_tpz=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_tpz.pkl"),        
        model_lephare=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_lephare.pkl"),        
        input=input_file,
        spec_input = input_file,
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_estimate_all.yml')
