#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *

from rail.utils.name_utils import NameFactory
from rail.core.stage import RailStage, RailPipeline

import ceci


from rail.utils.path_utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class GoldenspikePipeline(RailPipeline):

    default_input_dict = dict(
        model=flow_file,
    )
    
    def __init__(self, namer=None):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        bands = ['u','g','r','i','z','y']
        band_dict = {band:f'mag_{band}_lsst' for band in bands}
        rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}

        if namer is None:
            namer = NameFactory()
        truth_catalog_dir = namer.resolve_path_template('truth_catalog_path')
        degraded_catalog_dir = namer.resolve_path_template('degraded_catalog_path', selection="default", flavor="baseline")
        path_kwargs = dict(
            selection="default",
            flavor="baseline",
        )
            
        self.flow_engine_train = FlowCreator.build(
            model=flow_file,
            n_samples=50,
            seed=1235,
            output=os.path.join(truth_catalog_dir, "output_flow_engine_train.pq"),
        )

        self.lsst_error_model_train = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_train.io.output),    
            renameDict=band_dict, seed=29,
            output=os.path.join(degraded_catalog_dir, "output_lsst_error_model_train.pq"),
        )

        self.inv_redshift = InvRedshiftIncompleteness.build(
            connections=dict(input=self.lsst_error_model_train.io.output),
            pivot_redshift=1.0,
            output=os.path.join(degraded_catalog_dir, "output_inv_redshift.pq"),
        )

        self.line_confusion = LineConfusion.build(
            connections=dict(input=self.inv_redshift.io.output),
            true_wavelen=5007., wrong_wavelen=3727., frac_wrong=0.05,
            output=os.path.join(degraded_catalog_dir, "output_line_confusion.pq"),
        )

        self.quantity_cut = QuantityCut.build(
            connections=dict(input=self.line_confusion.io.output),
            cuts={'mag_i_lsst': 25.0},
            output=os.path.join(degraded_catalog_dir, "output_quantity_cut.pq"),
        )

        self.col_remapper_train = ColumnMapper.build(
            connections=dict(input=self.quantity_cut.io.output),
            columns=rename_dict,
            output=os.path.join(degraded_catalog_dir, "output_col_remapper_train.pq"),
        )

        self.table_conv_train = TableConverter.build(
            connections=dict(input=self.col_remapper_train.io.output),
            output_format='numpyDict',
            output=os.path.join(degraded_catalog_dir, "output_table_conv_train.hdf5"),
        )

        self.flow_engine_test = FlowCreator.build(
            model=flow_file,
            n_samples=50,
            output=os.path.join(truth_catalog_dir, "output_flow_engine_test.pq"),
        )

        self.lsst_error_model_test = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_test.io.output),
            bandNames=band_dict,
            output=os.path.join(degraded_catalog_dir, "output_lsst_error_model_test.pq"),
        )

        self.col_remapper_test = ColumnMapper.build(
            connections=dict(input=self.lsst_error_model_test.io.output),
            columns=rename_dict,
            output=os.path.join(degraded_catalog_dir, "output_col_remapper_test.pq"),
        )

        self.table_conv_test = TableConverter.build(
            connections=dict(input=self.col_remapper_test.io.output),
            output_format='numpyDict',
            output=os.path.join(degraded_catalog_dir, "output_table_conv_test.hdf5"),
        )

        self.inform_knn = KNearNeighInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            nondetect_val=np.nan,
            model=namer.resolve_path_template('estimator_model_path', algorithm='knn', model_suffix='.pkl', **path_kwargs),
            hdf5_groupname=''
        )

        self.inform_fzboost = FlexZBoostInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            model=namer.resolve_path_template('estimator_model_path', algorithm='fzboost', model_suffix='.pkl', **path_kwargs),
            hdf5_groupname=''
        )

        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            model=namer.resolve_path_template('estimator_model_path', algorithm='bpz', model_suffix='.pkl', **path_kwargs),
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
            output=namer.resolve_path_template('pz_pdf_path', algorithm='bpz', **path_kwargs),
        )

        self.estimate_knn = KNearNeighEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_knn.io.model,
            ),
            hdf5_groupname='',
            nondetect_val=np.nan,
            output=namer.resolve_path_template('pz_pdf_path', algorithm='knn', **path_kwargs),
        )

        self.estimate_fzboost = FlexZBoostEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_fzboost.io.model,
            ),
            nondetect_val=np.nan,
            hdf5_groupname='',            
            output=namer.resolve_path_template('pz_pdf_path', algorithm='fzboost', **path_kwargs),
        )

        eval_dict = dict(bpz=self.estimate_bpz, fzboost=self.estimate_fzboost, knn=self.estimate_knn)
        for key, val in eval_dict.items():
            the_eval = DistToPointEvaluator.make_and_connect(
                name=f'{key}_dist_to_point',
                connections=dict(
                    input=val.io.output,
                    truth=self.flow_engine_train.io.output,
                ),
                output=namer.resolve_path_template('per_object_metrics_path', algorithm=key, **path_kwargs),
                force_exact=True,
            )
            self.add_stage(the_eval)

        self.point_estimate_test = PointEstHistSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=namer.resolve_path_template('nz_pdf_path', algorithm='bpz', nzmethod='point_estimate', **path_kwargs),
            single_NZ=namer.resolve_path_template('single_nz_pdf_path', algorithm='bpz', nzmethod='point_estimate', **path_kwargs),
        )

        self.naive_stack_test = NaiveStackSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
            output=namer.resolve_path_template('nz_pdf_path', algorithm='bpz', nzmethod='stack', **path_kwargs),
            single_NZ=namer.resolve_path_template('single_nz_pdf_path', algorithm='bpz', nzmethod='stack', **path_kwargs),
        )
