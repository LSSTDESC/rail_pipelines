import numpy as np
import os
import scipy.special

import ceci
from rail.core.algo_utils import one_algo
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR
from rail.estimation.algos.uniform_binning import UniformBinningClassifier
from rail.core.data import (
    TableHandle,
    QPHandle,
    Hdf5Handle
)

sci_ver_str = scipy.__version__.split(".")

from rail.pipelines.utils.name_factory import NameFactory, DataType, CatalogType, ModelType, PdfType
namer = NameFactory()

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


input_file = os.path.join('short-output_BPZ_lite.hdf5')
input_data = DS.read_file('input_data', QPHandle, input_file)


class UniformBinningPipeline(RailPipeline):
    def __init__(self):
        RailPipeline.__init__(self)
        bands = ['u','g','r','i','z','y']
        
        self.ub_classify = UniformBinningClassifier.build(
            input=input_data,
            point_estimate = 'zmode',
            no_assign = -99,
            zmin = 0.0,
            zmax = 0.3,
            nbins = 1,
            zbin_edges = [0.0, 0.3], 
            id_name = "CATAID",
        )
        
        output_data = self.ub_classify.classify(input_data)
        

if __name__ == "__main__":
    pipeline = UniformBinningPipeline()
    
    input_dict = dict(
        model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_uniform_binning.pkl"), 
        input_data=input_file, 
#        input=input_file,
    )
    
    pipeline.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipeline.save("new_uniform_binning.yml", reduce_config=True)
    
    print("Done.")