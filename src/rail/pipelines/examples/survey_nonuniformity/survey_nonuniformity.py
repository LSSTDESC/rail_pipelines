#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Extra packages used by this pipeline
from astropy.io import fits
import healpy as hp
import pickle
import pandas as pd
from collections import OrderedDict
import yaml

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.pipelines.utils.name_factory import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline

import ceci

# other RAIL modules:
import tables_io

from rail.core.data import TableHandle
from rail.core.stage import RailStage

#import pzflow
#from pzflow import Flow
from rail.creation.engines.flowEngine import FlowCreator

from rail.creation.degradation import observing_condition_degrader
from rail.creation.degradation.observing_condition_degrader import ObsCondition

from rail.estimation.algos.flexzboost import Inform_FZBoost, FZBoost
from rail.estimation.algos.bpz_lite import BPZ_lite

# also need to import the reddening stage
from rail.core.utilStages import Dereddener

namer = NameFactory()
#from rail.core.utils import RAILDIR
# for now we use MYDIR, change to something else later
MYDIR = ""
flow_file = os.path.join(MYDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')



class SurveyNonuniformPipeline(RailPipeline):
    
    def __init__(self):
        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        
        bands = ['u','g','r','i','z','y']
        band_dict = {band:f'mag_{band}_lsst' for band in bands}
        rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
        
        # load pretrained flowmodel
        self.flow_engine_train = FlowCreator.build(
            model=flow_file,
            n_samples=50,
            seed=1235,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.created), "output_flow_engine_train.pq"),
        )
        
        # Here we need to convert semi major minor axies
        
        # apply LSST error model with maps
        self.obs_condition_train = ObsCondition.build(
            connections=dict(input=self.flow_engine_train.io.output),    
            bandNames=band_dict, seed=29,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_lsst_error_model_train.pq"),
        )
        
        # deredden
        self.deredden = Dereddener.build(
            connections=dict(input=self.obs_condition_train.io.output),
            bandNames=band_dict, 
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_lsst_error_model_deredden_train.pq"),
        )
        
        # use BPZ to estimate redshifts:
        self.estimate_bpz = BPZ_lite.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_bpz.io.model,
            ),
            hdf5_groupname='',
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_bpz.hdf5"),
        )
        
        # some sort of point estimates for pz
        self.point_estimate_test = PointEstimateHist.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_point_estimate_test.hdf5"),
            single_NZ=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "single_NZ_point_estimate_test.hdf5"),
        )
        
        # a stage that assign objects into tomographic bins
        
    
        # for each set of pixels in the depth bin, check the mean and width of the tomographic bin

        
        
