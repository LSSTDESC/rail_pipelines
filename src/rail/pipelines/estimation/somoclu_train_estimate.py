#!/usr/bin/env python
# coding: utf-8

from rail.core.stage import RailStage, RailPipeline
from rail.estimation.algos.somoclu_som import SOMocluInformer, SOMocluSummarizer


class SOMTrainEstimatePipeline(RailPipeline):
    """Pipeline that trains a SOMoclu SOM and uses it to estimate n(z).

    Stages
    ------
    inform_som : SOMocluInformer
        Trains the self-organising map on the photometric input catalog.
    summarize_som : SOMocluSummarizer
        Uses the trained SOM together with a spectroscopic reference sample
        to produce an n(z) ensemble.
    """

    default_input_dict = {
        'input_train': 'dummy.in',
        'input_photo': 'dummy.in',
        'spec_input': 'dummy.in',
    }

    def __init__(self, inform_dict=None, summ_dict=None):
        RailPipeline.__init__(self)

        if inform_dict is None:
            inform_dict = {}
        if summ_dict is None:
            summ_dict = {}

        informer = SOMocluInformer.make_and_connect(
            name='inform_som',
            aliases=dict(input='input_train'),
            **inform_dict,
        )
        self.add_stage(informer)

        summarizer = SOMocluSummarizer.make_and_connect(
            name='summarize_som',
            aliases=dict(input='input_photo'),
            connections=dict(model=informer.io.model),
            **summ_dict,
        )
        self.add_stage(summarizer)
