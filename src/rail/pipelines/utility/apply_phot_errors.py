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
from rail.core.utils import RAILDIR

input_file = 'rubin_dm_dc2_example.pq'


class ApplyPhotErrorsPipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        #band_list = [f'mag_{band}_lsst' for band in bands] + [f'mag_err_{band}_lsst' for band in bands]
        
        self.reddener = Redenner.build(
            input= dummy,
        )
        
        self.phot_errors = Redenner.build(
            input= dummy,
        )

        self.dereddener_errors = Redenner.build(
            input= dummy,
        )




if __name__ == '__main__':    
    pipe = ApplyPhotErrorsPipeline()
    input_dict = dict(
        input=input_file,
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_apply_phot_errors.yml')
