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

truth_file = 'dered_223501_sz_match_pdr3_dud_NONDET.hdf5'

class EvaluationPipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        #band_list = [f'mag_{band}_lsst' for band in bands] + [f'mag_err_{band}_lsst' for band in bands]
        
        self.evaluate_trainz = Evaluator.build(
            aliases=dict(input="input_trainz"),            
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_trainz"),
            hdf5_groupname='',
        )
        
        self.evaluate_simplenn = Evaluator.build(
            aliases=dict(input="input_simplenn"),                        
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_simplenn"),
            hdf5_groupname='',
        )
        
        self.evaluate_knn = Evaluator.build(
            aliases=dict(input="input_knn"),                        
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_knn"),
            hdf5_groupname='',
        )

        """
        self.evaluate_simplesom = Evaluator.build(
            aliases=dict(input="input_simplesom"),                        
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_simplesom"),
             hdf5_groupname='',
        )
        """

        """
        self.evaluate_somoclu = Evaluator.build(
            aliases=dict(input="input_somoclu"),                        
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_somoclu"),
            hdf5_groupname='',
        )
        """

        """
        self.evaluate_bpz = Evaluator.build(
            aliases=dict(input="input_bpz"),            
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_bpz"),
             hdf5_groupname='',
        )
        """
        
        """
        self.evaluate_delight = Evaluator.build(
            aliases=dict(input="input_trainz"),            
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_delight"),
             hdf5_groupname='',
        )
        """
        
        self.evaluate_fzboost = Evaluator.build(
            aliases=dict(input="input_fzboost"),            
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_FZBoost"),
            hdf5_groupname='',
        )
        
        """
        self.evaluate_gpz = Evaluator.build(
            aliases=dict(input="input_gpz"),            
            output=os.path.join(namer.get_data_dir(DataType.metric, ModelType.estimator), "output_evaluate_gpz"),
             hdf5_groupname='',
        )
        """

        
if __name__ == '__main__':    
    pipe = EvaluatePipeline()
    input_dict = dict(
        input_knn=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_knn.hdf5"),
        input_simplenn=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_simplenn.hdf5"),
        input_simplesom=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_simplesom.hdf5"),
        input_somoclu=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_somoclu.hdf5"),
        input_fzboost=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_FZBoost.hdf5"), #_fzboost
        input_trainz=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_trainz.hdf5"),      
        truth=truth_file,
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_evaluate_all.yml')
