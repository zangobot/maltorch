from abc import ABC
from typing import Optional
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from src.secmlware.zoo.model import Model
from src.secmlware.zoo.ember_gbdt import EmberGBDT
from src.secmlware.zoo.malconv import MalConv
from src.secmlware.zoo.yara_matching import Yara


#TODO: Create each model inside the pipeline as a Model class

class SliferClassifierOld(Model):

    def __init__(self, name: str = None, gdrive_id: Optional[str] = None, model_paths: Optional[list[str]] = None):
        super().__init__('TODO', 'boh')
        self.yara = None
        self.malconv = None
        self.gbdt = None
        self.load_pretrained_model(model_paths=model_paths)

    def load_pretrained_model(self, device="cpu", model_paths=None):
        if model_paths is None:
            #TODO fare meglio
            raise ValueError("Model paths must be provided.")
        else:
            self.yara = Yara.create_model(rules_path=model_paths[0])
            self.malconv = MalConv.create_model(model_path=model_paths[1], device=device)
            self.gbdt = EmberGBDT.create_model(model_path=model_paths[2])
            # self.yara = Yara.create_model(rules_path='/Users/bridge/PhD/Code/secml2malware/yara_rules')
            # self.malconv = MalConv.create_model(device='cpu')
            # self.gbdt = EmberGBDT.create_model(model_path='/Users/bridge/PhD/Code/secml2malware/src/secmlware/zoo/models'
            #                                                '/ember_model.txt')

    @classmethod
    def create_model(
            cls,
            model_paths: Optional[list[str]] = None,
            device: str = "cpu",
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None
    ) -> BaseModel:
        model = cls(model_paths=model_paths)
        return _SliferModel(yara=model.yara, malconv=model.malconv, gbdt=model.gbdt)


class _SliferModel(BaseModel):

    def __init__(self, yara: Yara, malconv: MalConv, gbdt: EmberGBDT):
        super().__init__()
        self.yara: Yara = yara
        self.malconv: MalConv = malconv
        self.gbdt: EmberGBDT = gbdt

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self._decision_function(x)
        return y

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        predictions = []
        for i in range(n_samples):
            x_i = x[i, :]
            # x_i_bytes = x_i.numpy().tobytes()
            x_i = x_i.unsqueeze(0)
            if self.yara.predict(x_i).item() == 1:
                predictions.append(torch.Tensor([1]))
            elif self.malconv.predict(x_i).item() == 1:
                predictions.append(torch.Tensor([1]))
            else:
                try:
                    if self.gbdt.predict(x_i).item() == 1:
                        predictions.append(torch.Tensor([1]))
                except Exception as e:
                    predictions.append(torch.Tensor([0]))
        predictions = torch.stack(predictions)
        return predictions

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        raise NotImplementedError("No gradient for Slifer model.")

    def train(self, dataloader: DataLoader) -> "BaseModel":
        raise NotImplementedError("Trainer not implemented yet.")
