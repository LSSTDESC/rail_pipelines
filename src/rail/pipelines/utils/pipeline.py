import ceci

from rail.core.stage import RailStageBuild

class RailPipeline(ceci.MiniPipeline):

    def __init__(self):
        ceci.MiniPipeline.__init__(self, [], dict(name='mini'))
        
    def __setattr__(self, name, value):
        if isinstance(value, RailStageBuild):
            stage = value.build(name)
            self.add_stage(stage)
            return stage
        return ceci.MiniPipeline.__setattr__(self, name, value)
    
