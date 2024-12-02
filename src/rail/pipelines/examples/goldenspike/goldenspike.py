#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.core.stage import RailStage, RailPipeline
from rail.creation.engines.flowEngine import FlowCreator
from rail.creation.degraders.photometric_errors import LSSTErrorModel
from rail.creation.degraders.quantityCut import QuantityCut
from rail.creation.degraders.spectroscopic_degraders import LineConfusion, InvRedshiftIncompleteness
from rail.tools.table_tools import ColumnMapper, TableConverter
from rail.estimation.algos.bpz_lite import BPZliteEstimator, BPZliteInformer
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.estimation.algos.k_nearneigh import KNearNeighEstimator, KNearNeighInformer
from rail.estimation.algos.naive_stack import NaiveStackSummarizer
from rail.estimation.algos.point_est_hist import PointEstHistSummarizer
from rail.evaluation.dist_to_point_evaluator import DistToPointEvaluator


from rail.utils.path_utils import RAILDIR
flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')


class GoldenspikePipeline(RailPipeline):

    default_input_dict = dict(
        model=flow_file,
    )

    def __init__(self) -> None:
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
        )

        self.lsst_error_model_train = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_train.io.output),
            renameDict=band_dict, seed=29,
        )

        self.inv_redshift = InvRedshiftIncompleteness.build(
            connections=dict(input=self.lsst_error_model_train.io.output),
            pivot_redshift=1.0,
        )

        self.line_confusion = LineConfusion.build(
            connections=dict(input=self.inv_redshift.io.output),
            true_wavelen=5007., wrong_wavelen=3727., frac_wrong=0.05,
        )

        self.quantity_cut = QuantityCut.build(
            connections=dict(input=self.line_confusion.io.output),
            cuts={'mag_i_lsst': 25.0},
        )

        self.col_remapper_train = ColumnMapper.build(
            connections=dict(input=self.quantity_cut.io.output),
            columns=rename_dict,
        )

        self.table_conv_train = TableConverter.build(
            connections=dict(input=self.col_remapper_train.io.output),
            output_format='numpyDict',
        )

        self.flow_engine_test = FlowCreator.build(
            model=flow_file,
            n_samples=50,
        )

        self.lsst_error_model_test = LSSTErrorModel.build(
            connections=dict(input=self.flow_engine_test.io.output),
            bandNames=band_dict,
        )

        self.col_remapper_test = ColumnMapper.build(
            connections=dict(input=self.lsst_error_model_test.io.output),
            columns=rename_dict,
        )

        self.table_conv_test = TableConverter.build(
            connections=dict(input=self.col_remapper_test.io.output),
            output_format='numpyDict',
        )

        self.inform_knn = KNearNeighInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            nondetect_val=np.nan,
            hdf5_groupname=''
        )

        self.inform_fzboost = FlexZBoostInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
            hdf5_groupname=''
        )

        self.inform_bpz = BPZliteInformer.build(
            connections=dict(input=self.table_conv_train.io.output),
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
        )

        self.estimate_knn = KNearNeighEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_knn.io.model,
            ),
            hdf5_groupname='',
            nondetect_val=np.nan,
        )

        self.estimate_fzboost = FlexZBoostEstimator.build(
            connections=dict(
                input=self.table_conv_test.io.output,
                model=self.inform_fzboost.io.model,
            ),
            nondetect_val=np.nan,
            hdf5_groupname='',
        )

        eval_dict = dict(bpz=self.estimate_bpz, fzboost=self.estimate_fzboost, knn=self.estimate_knn)
        for key, val in eval_dict.items():
            the_eval = DistToPointEvaluator.make_and_connect(
                name=f'{key}_dist_to_point',
                connections=dict(
                    input=val.io.output,
                    truth=self.flow_engine_train.io.output,
                ),
                force_exact=True,
            )
            self.add_stage(the_eval)

        self.point_estimate_test = PointEstHistSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
        )

        self.naive_stack_test = NaiveStackSummarizer.build(
            connections=dict(input=self.estimate_bpz.io.output),
        )
