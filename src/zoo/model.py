from abc import ABC, abstractmethod
from typing import Optional

import torch
from secml2.models.base_model import BaseModel
from secml2.models.base_trainer import BaseTrainer
from secml2.models.data_processing.data_processing import DataProcessing
from secml2.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from src.utils.config import Config
from src.utils.utils import download_gdrive


class Model(torch.nn.Module, ABC):
    def __init__(self, name: str, gdrive_id: Optional[str]):
        super().__init__()
        self.name = name
        self.gdrive_id = gdrive_id
        self.model_path = Config.MODEL_ZOO_FOLDER / self.name

    def _fetch_pretrained_model(self):
        if not Config.MODEL_ZOO_FOLDER.exists():
            Config.MODEL_ZOO_FOLDER.mkdir()
        if not self.model_path.exists():
            if self.gdrive_id is not None:
                download_gdrive(gdrive_id=self.gdrive_id, fname_save=self.model_path)

    def load_pretrained_model(self, device="cpu", model_path=None):
        ...

    @classmethod
    def create_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "cpu",
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
    ) -> BaseModel:
        ...


class PytorchModel(Model):
    def load_pretrained_model(self, device="cpu", model_path=None):
        path = self.model_path
        if model_path is None:
            self._fetch_pretrained_model()
        else:
            path = model_path
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @classmethod
    def create_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "cpu",
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
        **kwargs,
    ) -> BaseModel:
        net = cls(**kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net.eval()
        net = BasePytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
        )
        return net


class BaseEmbeddingPytorchClassifier(BasePytorchClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
    ):
        super().__init__(model, preprocessing, postprocessing, trainer)

    def embed(self, x: torch.Tensor):
        return self.model.embed(x)

    def embedding_matrix(self):
        return self.model.embedding_matrix()


class EmbeddingModel(PytorchModel, ABC):
    @classmethod
    def create_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "cpu",
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
        input_embedding: bool = False,
        **kwargs,
    ) -> BaseEmbeddingPytorchClassifier:
        net = cls(input_embedding=input_embedding, **kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net.eval()
        net = BaseEmbeddingPytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
        )
        return net

    def __init__(
        self, name: str, gdrive_id: Optional[str], input_embedding: bool = False
    ):
        super().__init__(name, gdrive_id)
        self.input_embedding = input_embedding

    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def embedding_matrix(self):
        pass

    @abstractmethod
    def _forward_embed_x(self, x):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            the sample to test
        Returns
        -------
        torch.Tensor
            the result of the forward pass
        """
        if not self.input_embedding:
            x = self.embed(x)
        output = self._forward_embed_x(x)
        return output
