from typing import Optional

import ember.features
import lightgbm
import numpy as np
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader

from maltorch.zoo.model import Model


class EmberGBDT(Model):
    def __init__(self, model_path: str = ""):
        super().__init__(
            name="ember_gbdt", gdrive_id="1MGR7l5c3XSH2dTj2oeefBlKig0bvH2_Z"
        )
        self.tree_model = None
        self.load_pretrained_model(model_path=model_path)

    def load_pretrained_model(self, device: str = "cpu", model_path: str = None):
        if model_path is None:
            self._fetch_pretrained_model()
            self.tree_model = lightgbm.Booster(model_file=self.model_path)
        else:
            self.tree_model = lightgbm.Booster(model_file=model_path)

    @classmethod
    def create_model(
            cls,
            model_path: Optional[str] = None,
            threshold: int = 0.82,
            device: str = "cpu",
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
    ) -> BaseModel:
        model = cls(model_path=model_path)
        return _GBDTModel(tree_model=model.tree_model, threshold=threshold)


class _GBDTModel(BaseModel):
    def __init__(self, tree_model: lightgbm.Booster, threshold: float = 0.82):
        super().__init__()
        self.tree_model: lightgbm.Booster = tree_model
        self.threshold: float = threshold

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self._decision_function(x)
        return torch.Tensor(y > self.threshold).int()

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        extractor = ember.features.PEFeatureExtractor(print_feature_warning=False)
        n_samples = x.shape[0]
        feat_x = np.zeros((x.shape[0], extractor.dim))
        for i in range(n_samples):
            x_i = x[i, :]
            x_bytes = bytes(x_i.type(torch.int).flatten().tolist())
            feat_x[i, :] = torch.Tensor(extractor.feature_vector(x_bytes))
        probabilities = self.tree_model.predict(feat_x)
        probabilities = torch.Tensor(probabilities).unsqueeze(1)
        return probabilities

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError("No gradient for GBDT model.")

    def train(self, dataloader: DataLoader) -> "BaseModel":
        raise NotImplementedError("Trainer not implemented yet.")
