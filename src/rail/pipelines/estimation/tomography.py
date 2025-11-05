#!/usr/bin/env python
# coding: utf-8

import ceci

# Various rail modules
from rail.core.stage import RailStage, RailPipeline
from rail.estimation.algos.true_nz import TrueNZHistogrammer

from rail.utils.algo_library import PZ_ALGORITHMS, CLASSIFIERS, SUMMARIZERS


class TomographyPipeline(RailPipeline):

    default_input_dict=dict(
        truth='dummy.in',
    )

    def __init__(
            self,
            algorithms: dict | None=None,
            classifiers: dict | None=None,
            summarizers: dict | None=None,
            n_tomo_bins: int=5,
        ):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = PZ_ALGORITHMS

        if classifiers is None:
            classifiers = CLASSIFIERS

        if summarizers is None:
            summarizers = SUMMARIZERS

        for pz_algo_name_ in algorithms:

            for classifier_name_, classifier_info_ in classifiers.items():

                classifier_class = ceci.PipelineStage.get_stage(
                    classifier_info_['Classify'],
                    classifier_info_['Module'],
                )
                the_classifier = classifier_class.make_and_connect(
                    aliases=dict(input=f"input_{pz_algo_name_}"),
                    name=f'classify_{pz_algo_name_}_{classifier_name_}',
                    nbins=n_tomo_bins,
                )
                self.default_input_dict[f"input_{pz_algo_name_}"] = 'dummy.in'
                self.add_stage(the_classifier)

                for ibin in range(1, n_tomo_bins+1):

                    true_nz = TrueNZHistogrammer.make_and_connect(
                        name=f"true_nz_{pz_algo_name_}_{classifier_name_}_bin{ibin}",
                        connections=dict(
                            tomography_bins=the_classifier.io.output,
                        ),
                        selected_bin=ibin,
                        aliases=dict(input='truth'),
                    )
                    self.add_stage(true_nz)

                    for summarizer_name_, summarize_info_ in summarizers.items():
                        summarizer_class = ceci.PipelineStage.get_stage(
                            summarize_info_['Summarize'],
                            summarize_info_['Module'],
                        )
                        the_summarizer = summarizer_class.make_and_connect(
                            name=f'summarize_{pz_algo_name_}_{classifier_name_}_bin{ibin}_{summarizer_name_}',
                            aliases=dict(input=f"input_{pz_algo_name_}"),
                            connections=dict(
                                tomography_bins=the_classifier.io.output,
                            ),
                            selected_bin=ibin,
                            nsamples=20,
                        )
                        self.add_stage(the_summarizer)
