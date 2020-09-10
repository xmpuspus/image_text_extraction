
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.adapters import ImageInput

@artifacts([PickleArtifact('model')])
@env(pip_dependencies=['pytesseract'], 
     requirements_txt_file='pytesseract',
     conda_channels=["conda-forge"], 
     conda_dependencies=["ruamel.yaml"])
class TextDetectionService(BentoService):
    
    @api(input=ImageInput())
    def predict(self, image):
        result = self.artifacts.model.detect_text(image)
        return result
