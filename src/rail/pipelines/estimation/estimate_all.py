#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import ceci


from rail.core.stage import RailStage, RailPipeline
from rail.utils.project import PZ_ALGORITHMS


input_file = 'rubin_dm_dc2_example.pq'


class EstimatePipeline(RailPipeline):

    default_input_dict={'input':'dummy.in'}
    
    def __init__(self, algorithms=None, models_dir='.'):

        RailPipeline.__init__(self)
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = PZ_ALGORITHMS.copy()

        for key, val in algorithms.items():
            the_class = ceci.PipelineStage.get_stage(val['Estimate'])
            the_estimator = the_class.make_and_connect(
                name=f'estimate_{key}',
                aliases=dict(model=f"model_{key}"),
                hdf5_groupname='',
            )
            model_path = namer.resolve_path_template(
                "ceci_file_path",                
                stage=f'inform_{key}',
                tag='model',
                suffix='.pkl',
            )
            self.default_input_dict[f"model_{key}"] = os.path.join(models_dir, model_path)
            self.add_stage(the_estimator)
