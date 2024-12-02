#!/usr/bin/env python
# coding: utf-8

import ceci

from rail.core.stage import RailStage, RailPipeline
from rail.utils.algo_library import PZ_ALGORITHMS


input_file = 'rubin_dm_dc2_example.pq'


class InformPipeline(RailPipeline):

    default_input_dict={'input':'dummy.in'}

    def __init__(self, algorithms: dict | None=None):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = PZ_ALGORITHMS

        for key, val in algorithms.items():
            the_class = ceci.PipelineStage.get_stage(val['Inform'], val['Module'])
            the_informer = the_class.make_and_connect(
                name=f'inform_{key}',
                hdf5_groupname='',
            )
            self.add_stage(the_informer)
