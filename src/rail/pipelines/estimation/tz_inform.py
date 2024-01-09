#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
# import rail.stages
# rail.stages.import_and_attach_all()
# from rail.stages import *
from rail.estimation.algos import train_z

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

        self.inform_trainz = train_z.TrainZInformer.build(
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_trainz.pkl"),
            hdf5_groupname='',
        )


if __name__ == '__main__':    
    pipe = InformPipeline()
    input_dict = dict(
        input=input_file,
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_inform_tz.yml')
