from abc import ABC
from typing import Optional
import numpy as np
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
import yara  # TODO add to requirements ?
import os
from torch.utils.data import DataLoader
from src.secmlware.zoo.model import Model


class Yara(Model):

    def __init__(self, model_path: str = "", name: str = None, gdrive_id: Optional[str] = None):
        super().__init__('yara_rules', 'boh')
        self.yara = None
        self.load_pretrained_model(rules_path=model_path)

    def load_pretrained_model(self, device="cpu", rules_path=None):
        if rules_path is None:
            raise ValueError("Rules path must be provided and must be a directory.")
        else:
            # building rules dictionary
            file_paths = [os.path.join(rules_path, file) for file in os.listdir(rules_path) if
                          os.path.isfile(os.path.join(rules_path, file))]
            paths_dict = {key: key for key in file_paths}
            # compiling rules
            self.yara = yara.compile(filepaths=paths_dict)

    @classmethod
    def create_model(
            cls,
            rules_path: Optional[str] = None,
            device: str = "cpu",
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None
    ) -> BaseModel:
        model = cls(model_path=rules_path)
        return _YaraModel(model=model.yara)


class _YaraModel(BaseModel, ABC):
    def __init__(self, model: yara.Rules):
        super().__init__()
        self.model = model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self._decision_function(x)
        return torch.Tensor(y > 0).int()

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        matches = []
        for i in range(n_samples):
            x_i = x[i, :].data.to(torch.uint8).numpy().tobytes()
            match = self.model.match(data=x_i)
            # checking how many rules have been triggered
            triggered_rules = len(match)
            match = torch.Tensor([triggered_rules]).to(dtype=torch.long)
            if triggered_rules == 0:
                match = torch.Tensor([0.0])
            matches.append(match)
        matches = torch.stack(matches)
        return matches

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError("No gradient for YARA.")

    def train(self, dataloader: DataLoader) -> "BaseModel":
        raise NotImplementedError("Trainer not implemented for YARA.")
