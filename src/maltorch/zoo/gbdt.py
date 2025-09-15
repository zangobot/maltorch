import lightgbm
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader


class GBDTModel(BaseModel):
    def __init__(self, tree_model: lightgbm.Booster, threshold: float = 0.82, preprocessing: DataProcessing = None,
                 postprocessing: DataProcessing = None):
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
        self.tree_model: lightgbm.Booster = tree_model
        self.threshold: float = threshold

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.decision_function(x)
        return torch.Tensor(y > self.threshold).int()

    def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        probabilities = self.tree_model.predict(x.numpy())
        probabilities = torch.Tensor(probabilities).unsqueeze(1)
        return probabilities

    def gradient(self, x: torch.Tensor, y: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("No gradient for GBDT model.")

    def train(self, dataloader: DataLoader) -> "BaseModel":
        raise NotImplementedError("Trainer not implemented yet.")
