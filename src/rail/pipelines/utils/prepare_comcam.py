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

class PrepareComCamPipeline(RailPipeline):

    default_input_dict={'input':'dummy.parq'}

    def __init__(
        self,
        **kwargs,
    ):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog_config = catalog_utils.CatalogConfigBase.active_class()
        band_name_dict = active_catalog_config.band_name_dict()        
        
        self.flux_to_mag = LSSTFluxToMagConverter.build(
            flux_name="{band}_cModelFlux",
            flux_err_name="{band}_cModelFluxErr",
            mag_name=active_catalog_config.band_template,
            mag_err_name=active_catalog_config.band_err_template,
            copy_cols=dict(
                objectId='objectId',
                ra='coord_ra',
                dec='coord_dec',
            )
        )
            
        self.deredden = Dereddener.build(
            connections = dict(input=self.flux_to_mag.io.output),
            dustmap_dir=dustmap_dir,
            mag_name=active_catalog_config.band_template,
            copy_all_cols=True,
        )
