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


shared_stage_opts = dict(
    metrics=['all'], 
    exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
    hdf5_groupname="", 
    limits=[0, 3.5], 
    truth_point_estimates=['redshift'],
    point_estimates=['mode'],
)



class EvaluationPipeline(RailPipeline):

    def __init__(self, estimate_list):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        
        metric_outdir = namer.get_data_dir(DataType.metric, PdfType.pz)
        
        for estimate_ in estimate_list:
            the_eval = SingleEvaluator.make_and_connect(
                name=f'evaluate_{estimate_}',
                aliases=dict(input=f"input_evalute_{estimate_}"),
                output=os.path.join(metric_outdir, f"evaluate_output_{estimate_}.pq"),
                summary=os.path.join(metric_outdir, f"evaluate_summary_{estimate_}.pq"),
                single_distribution_summary=os.path.join(metric_outdir, f"evaluate_single_distribution_summary_{estimate_}.hdf5"),
                **shared_stage_opts,
            )
            self.add_stage(the_eval)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=f"Set up evaluation pipeline")

    parser.add_argument('-t', "--truth", action='store', default="dummy.in")
    parser.add_argument('-e', "--estimates", action='append')

    args = parser.parse_args()
    
    pipe = EvaluationPipeline(args.estimates)
    input_dict = dict(
        truth=args.truth,        
    )

    for estimate_ in args.estimates:
        input_dict[f"input_evalute_{estimate_}"] = "dummy.in"
        
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_evaluate_all.yml')
