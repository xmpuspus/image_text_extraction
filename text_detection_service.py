
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.adapters import FileInput

@artifacts([PickleArtifact('model')])
@env(pip_dependencies=['easyocr'], 
     conda_channels=["conda-forge"], 
     conda_dependencies=["ruamel.yaml"])
class TextDetectionService(BentoService):
    
    @api(input=FileInput())
    def predict(self, image):
        result = self.artifacts.model.detect_text(image[0])
        return result
