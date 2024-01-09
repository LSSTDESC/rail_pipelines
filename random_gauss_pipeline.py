import numpy as np
import os
import scipy.special

import ceci
from rail.core.algo_utils import one_algo
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR
from rail.estimation.algos import random_gauss
from rail.core.data import (
    TableHandle,
)

sci_ver_str = scipy.__version__.split(".")


from rail.pipelines.utils.name_factory import NameFactory, DataType, CatalogType, ModelType, PdfType
namer = NameFactory()


DS = RailStage.data_store
DS.__class__.allow_overwrite = True

validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')
validation_data = DS.read_file('validation_data', TableHandle, validdata)

class RandomGaussPipeline(RailPipeline):
    def __init__(self):
        RailPipeline.__init__(self)
        
        bands = ['u','g','r','i','z','y']
        #band_dict = {band:f'mag_{band}_lsst' for band in bands}
        #rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
        
        self.rg_estimate = random_gauss.RandomGaussEstimator.build(
            input=validation_data,
            rand_width = 0.025,
            rand_zmin = 0.0,
            rand_zmax = 3.0,
            nzbins = 301,
            hdf5_groupname = "photometry",
            model = None,
            seed = 42,
        )


if __name__ == "__main__":
    #make_gauss_pipeline()
    pipeline = RandomGaussPipeline()
    input_dict = dict(
        model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_randgauss.pkl"), 
        input=validation_data
    )
    pipeline.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipeline.save("tmp_random_gauss.yml")
    
# ---------------------------------------------------------------------------------------


def make_gauss_pipeline():
    # Config values
    train_config_dict = {}
    estim_config_dict = {
        "rand_width": 0.025,
        "rand_zmin": 0.0,
        "rand_zmax": 3.0,
        "nzbins": 301,
        "hdf5_groupname": "photometry",
        "model": "None",
        "seed": 42,
    }
    zb_expected = np.array([2.322, 1.317, 2.576, 2.092, 0.283, 2.927, 2.283, 2.358, 0.384, 1.351])
    
    # Set up
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    
    # Data
    traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
    validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)

    # Make informer stage and inform
    train_pz = random_gauss.RandomGaussInformer.make_stage(**train_config_dict)
    train_pz.inform(training_data)
    
    # Make estimator stage and estimate
    pz = random_gauss.RandomGaussEstimator.make_stage(name="RandomPZ", **estim_config_dict)
    estim = pz.estimate(validation_data)
    
    # Print config
    print("--> train_pz.config:")
    print(train_pz.config)
    print("--> pz.config:")
    print(pz.config)

    # Clean up
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
   
    # Check values
    assert np.isclose(estim.data.ancil['zmode'], zb_expected).all()
    
    # Now try as a pipeline instead of separate stages
    pipeline = RailPipeline()
    
    pipeline.bands = ['u','g','r','i','z','y']
    pipeline.band_dict = {band:f'mag_{band}_lsst' for band in pipeline.bands}
    pipeline.rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in pipeline.bands}
    
    pipeline.rg_estimate = random_gauss.RandomGaussEstimator.build(
        #name = "RandomPZ",
        input=validation_data,
        rand_width = 0.025,
        rand_zmin = 0.0,
        rand_zmax = 3.0,
        nzbins = 301,
        hdf5_groupname = "photometry",
        model = None,
        seed = 42,
    )
    
    pipeline.initialize(dict(model=None, input=validation_data), dict(output_dir='.', log_dir='.', resume=False), None)
    pipeline.save("tmp_random_gauss.yml")
        