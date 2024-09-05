from enum import Enum
from typing import Optional
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
import torch
from torch.utils.data import DataLoader

from src.secmlware.zoo.model import Model


# error handling policy
class ErrorHandling(Enum):
    ERR_AS_GOODWARE = 0
    ERR_AS_MALWARE = 1
    RAISE_ERROR = -1


class SequentialPipelineClassifier(Model):

    def __init__(self, models_type: list, error_handling: Optional[ErrorHandling] = ErrorHandling.RAISE_ERROR):
        super().__init__('', '')
        self.models: list = []
        self.error_handling = error_handling
        self.load_pretrained_model(models_type=models_type)

    def load_pretrained_model(self, device="cpu", models_type: list = None):
        for model in models_type:
            if model.model_path is not None:
                self.models.append(model.create_model(model.model_path, device=device))
            else:
                self.models.append(model.create_model(device=device))

    @classmethod
    def create_model(
            cls,
            models: list[Model] = None,
            device: str = "cpu",
            error_handling: ErrorHandling = ErrorHandling.RAISE_ERROR,
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
    ) -> BaseModel:
        model = cls(models_type=models)
        return _SequentialPipelineModel(models=model.models, error_handling=error_handling)


class _SequentialPipelineModel(BaseModel):

    def __init__(self, models: list[Model], error_handling: ErrorHandling):
        super().__init__()
        self.models = models
        self.error_handling = error_handling

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self._decision_function(x)
        return torch.Tensor(y > 0).int()

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        predictions = []
        for i in range(n_samples):
            x_i = x[i, :]
            x_i = x_i.unsqueeze(0)
            # y = torch.Tensor([0])
            for i, model in enumerate(self.models):
                try:
                    y = model.predict(x_i)
                    if y.item() == 1:
                        predictions.append(torch.Tensor([1]))
                        # if a model detects malware, the pipeline stops
                        break
                except Exception as e:
                    # if exception raised at the last model, return the error handling value
                    if i == len(self.models)-1:
                        predictions.append(self.error_handling.value)
                        break
            # if no model detects malware, return 0
            if len(predictions) == 0:
                predictions.append(torch.Tensor([0]))
        return torch.Tensor(predictions).unsqueeze(1)

    def train(self, dataloader: DataLoader) -> "BaseModel":
        raise NotImplementedError("Trainer not implemented for SequentialPipeline.")

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError("No gradient for SequentialPipeline.")


