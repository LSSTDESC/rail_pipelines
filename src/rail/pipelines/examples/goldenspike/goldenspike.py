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
from rail.utils.path_utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class GoldenspikePipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        band_dict = {band:f'mag_{band}_lsst' for band in bands}
        rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}

        self.flow_engine_train = FlowCreator.build(
            model=flow_file,
            n_samples=50,
            seed=1235,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.created), "output_flow_engine_train.pq"),
        )

        self.lsst_error_model_train = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_train.io.output),    
            renameDict=band_dict, seed=29,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_lsst_error_model_train.pq"),
        )

        self.inv_redshift = InvRedshiftIncompleteness.build(
            connections=dict(input=self.lsst_error_model_train.io.output),
            pivot_redshift=1.0,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_inv_redshift.pq"),
        )

        self.line_confusion = LineConfusion.build(
            connections=dict(input=self.inv_redshift.io.output),
            true_wavelen=5007., wrong_wavelen=3727., frac_wrong=0.05,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_line_confusion.pq"),
        )

        self.quantity_cut = QuantityCut.build(
            connections=dict(input=self.line_confusion.io.output),
            cuts={'mag_i_lsst': 25.0},
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_quantity_cut.pq"),
        )

        self.col_remapper_train = ColumnMapper.build(
            connections=dict(input=self.quantity_cut.io.output),
            columns=rename_dict,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_col_remapper_train.pq"),
        )

        self.table_conv_train = TableConverter.build(
            connections=dict(input=self.col_remapper_train.io.output),
            output_format='numpyDict',
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_table_conv_train.hdf5"),
        )

        self.flow_engine_test = FlowCreator.build(
            model=flow_file,
            n_samples=50,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_flow_engine_test.pq"),
        )

        self.lsst_error_model_test = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_test.io.output),
            bandNames=band_dict,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_lsst_error_model_test.pq"),
        )

        self.col_remapper_test = ColumnMapper.build(
            connections=dict(input=self.lsst_error_model_test.io.output),
            columns=rename_dict,
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_col_remapper_test.pq"),
        )

        self.table_conv_test = TableConverter.build(
            connections=dict(input=self.col_remapper_test.io.output),
            output_format='numpyDict',
            output=os.path.join(namer.get_data_dir(DataType.catalog, CatalogType.degraded), "output_table_conv_test.hdf5"),
        )

        self.inform_knn = KNearNeighInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            nondetect_val=np.nan,
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), 'knnpz.pkl'),
            hdf5_groupname=''
        )

        self.inform_fzboost = FlexZBoostInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), 'fzboost.pkl'),
            hdf5_groupname=''
        )

        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), 'trained_BPZ.pkl'),
            hdf5_groupname='',
            nt_array=[8],
            mmax=26.,
            type_file='',
        )

        self.estimate_bpz = BPZliteEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_bpz.io.model,
            ),
            hdf5_groupname='',
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_bpz.hdf5"),
        )

        self.estimate_knn = KNearNeighEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_knn.io.model,
            ),
            hdf5_groupname='',
            nondetect_val=np.nan,
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_knn.hdf5"),
        )

        self.estimate_fzboost = FlexZBoostEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_fzboost.io.model,
            ),
            nondetect_val=np.nan,
            hdf5_groupname='',
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_estimate_FZBoost.hdf5"),
        )

        eval_dict = dict(bpz=self.estimate_bpz, fzboost=self.estimate_fzboost, knn=self.estimate_knn)
        for key, val in eval_dict.items():
            the_eval = DistToPointEvaluator.make_and_connect(
                name=f'{key}_dist_to_point',
                connections=dict(
                    input=val.io.output,
                    truth=self.flow_engine_train.io.output,
                ),
                output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), f"output_{key}_eval.pq"),
                force_exact=True,
            )
            self.add_stage(the_eval)

        self.point_estimate_test = PointEstHistSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_point_estimate_test.hdf5"),
            single_NZ=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "single_NZ_point_estimate_test.hdf5"),
        )

        self.naive_stack_test = NaiveStackSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "output_naive_stack_test.hdf5"),
            single_NZ=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.nz), "single_NZ_naive_stack_test.hdf5"),
        )



if __name__ == '__main__':    
    pipe = GoldenspikePipeline()
    pipe.initialize(dict(model=flow_file), dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_goldenspike.yml')
