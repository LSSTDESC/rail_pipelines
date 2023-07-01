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

input_file = 'rubin_dm_dc2_example.pq'


class InformPipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        #band_list = [f'mag_{band}_lsst' for band in bands] + [f'mag_err_{band}_lsst' for band in bands]
        
        self.inform_trainz = TrainZInformer.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_trainz.pkl"),
            hdf5_groupname='',
        )
        
        self.inform_simplenn = Inform_SimpleNN.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_simplenn.pkl"),
            hdf5_groupname='',
        )
        
        self.inform_knn = KNearNeighInformer.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_knn.pkl"),
            hdf5_groupname='',
        )
        
        self.inform_simplesom = Inform_SimpleSOMSummarizer.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_simplesom.pkl"),
             hdf5_groupname='',
        )

        self.inform_somoclu = Inform_somocluSOMSummarizer.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_somoclu.pkl"),
            hdf5_groupname='',
        )


        self.inform_bpz = Inform_BPZ_lite.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_bpz.hdf5"),
             hdf5_groupname='',
        )
        
        """
        self.inform_delight = Inform_DelightPZ.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_delight.hdf5"),
             hdf5_groupname='',
        )
        """
        self.inform_fzboost = Inform_FZBoost.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_FZBoost.hdf5"),
            hdf5_groupname='',
        )
        
        self.inform_gpz = Inform_GPz_v1.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_gpz.hdf5"),
             hdf5_groupname='',
        )

        

if __name__ == '__main__':    
    pipe = InformPipeline()
    input_dict = dict(
        input=input_file,
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_inform_all.yml')
