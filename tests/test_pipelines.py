import os
from rail.pipelines.examples.goldenspike.goldenspike import GoldenspikePipeline
from rail.pipelines.utils.name_factory import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline
from rail.core.common_params import RAILDIR


def test_golden():
    namer = NameFactory()
    flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')
    output_dir = namer.get_project_dir('.', 'tmp_test', 'tmp_test')
    try:
        os.makedirs(output_dir)
    except OSError:  # pragma: no cover
        pass    
    pipe = GoldenspikePipeline()
    pipe.initialize(dict(model=flow_file), dict(output_dir=output_dir, log_dir=output_dir, resume=False), None)
    pipe.save('tmp_goldenspike.yml')
    os.system(f"\\rm -rf {output_dir}")
