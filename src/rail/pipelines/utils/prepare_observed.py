#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import ceci


# Various rail modules
from rail.core.stage import RailStage, RailPipeline
from rail.tools.photometry_tools import Dereddener, LSSTFluxToMagConverter
from rail.utils import catalog_utils


if 'PZ_DUSTMAP_DIR' not in os.environ:  # pragma: no cover
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


class PrepareObservedPipeline(RailPipeline):

    default_input_dict={'input':'dummy.parq'}

    def __init__(
        self,
        **kwargs,
    ):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog_config = catalog_utils.CatalogConfigBase.active_class()

        self.flux_to_mag = LSSTFluxToMagConverter.build(
            flux_name="{band}Flux",
            flux_err_name="{band}FluxErr",
            mag_name="{band}Mag",
            mag_err_name="{band}MagErr",
            bands=active_catalog_config.bandlist,
            copy_cols=dict(
                objectId='objectId',
                coord_ra='coord_ra',
                coord_dec='coord_dec',
            )
        )

        self.deredden = Dereddener.build(
            connections = dict(input=self.flux_to_mag.io.output),
            dustmap_dir=dustmap_dir,
            dustmap_name='sfd',
            ra_name='coord_ra',
            dec_name='coord_dec',
            # band_a_env=active_catalog_config.a_env_dict,
            mag_name="{band}Mag",
            copy_all_cols=True,
        )
